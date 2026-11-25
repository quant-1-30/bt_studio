#! /usr/bin/env python3

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from bt_studio.constant import (
    FEATURE_DIR,
    LLM_RUN_DIR,
    TUNE_BASE_DIR,
)
from bt_studio.utils.io import read_json_safe

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL = float(os.environ.get("BT_STUDIO_WATCH_INTERVAL", "5"))


class ArtifactKind(str, Enum):
    MODEL = "model"
    SCORE = "score"
    COLLAPSE = "collapse"
    LLM_RUN = "llm_run"
    FEATURE = "feature"


WATCH_SPECS: Dict[ArtifactKind, Tuple[str, str]] = {
    ArtifactKind.MODEL: (TUNE_BASE_DIR, ".pkl"),
    ArtifactKind.SCORE: (TUNE_BASE_DIR, ".parquet"),
    ArtifactKind.COLLAPSE: (TUNE_BASE_DIR, ".json"),
    ArtifactKind.LLM_RUN: (LLM_RUN_DIR, ".json"),
    ArtifactKind.FEATURE: (FEATURE_DIR, ".parquet"),
}

# re match avoid heavy io
_RE_MODEL = re.compile(r"^model_(\d+)\.pkl$")
_RE_SCORE = re.compile(r"^scores_(\d+)\.parquet$")
_RE_COLLAPSE = re.compile(r"^collapse_(\d+)_.+\.json$")
_RE_FEATURE = re.compile(r"^hf_(.+)_\d{6}\.parquet$")


@dataclass
class ArtifactRecord:
    kind: ArtifactKind
    name: str
    path: str
    mtime: float
    size: int
    summary: Optional[Dict[str, Any]] = None

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "path": self.path,
            "mtime": self.mtime,
            "size": self.size,
            "summary": self.summary,
        }


# ============================================================================
# parser and scan background
# ============================================================================

def _parse_summary_sync(kind: ArtifactKind, name: str, path: str) -> Optional[Dict[str, Any]]:
    try:
        if kind == ArtifactKind.COLLAPSE:
            data = read_json_safe(path, default={}) or {}
            return {
                "verdict": data.get("verdict"),
                "feature_col": data.get("feature_col"),
                "model_id": data.get("model_id"),
                "n_trials": data.get("n_trials"),
            }

        elif kind == ArtifactKind.LLM_RUN:
            data = read_json_safe(path, default={}) or {}
            features = data.get("features", []) or []
            return {
                "hypothesis_id": data.get("hypothesis_id"),
                "target": data.get("target"),
                "step_index": data.get("step_index"),
                "n_features": len(features),
                "feature_stages": [
                    f.get("stage") for f in features if isinstance(f, dict)
                ][:10],
            }

        # avoid unpickle bin
        elif kind == ArtifactKind.MODEL:
            m = _RE_MODEL.match(name)
            if m:
                return {"train_end_month": int(m.group(1))}

        elif kind == ArtifactKind.SCORE:
            m = _RE_SCORE.match(name)
            if m:
                return {"model_id": int(m.group(1))}

        elif kind == ArtifactKind.FEATURE:
            m = _RE_FEATURE.match(name)
            if m:
                return {"feature_col": m.group(1)}

    except Exception as e:
        logger.warning(f"[Watcher] Parser({path}) Failure: {e}")
        return None

    return None


def _matches_kind(kind: ArtifactKind, filename: str) -> bool:
    if kind == ArtifactKind.MODEL:
        return bool(_RE_MODEL.match(filename))
    if kind == ArtifactKind.SCORE:
        return bool(_RE_SCORE.match(filename))
    if kind == ArtifactKind.COLLAPSE:
        return bool(_RE_COLLAPSE.match(filename))
    if kind == ArtifactKind.LLM_RUN:
        return filename.startswith("step_") and filename.endswith(".json")
    if kind == ArtifactKind.FEATURE:
        return bool(_RE_FEATURE.match(filename))
    return True


def _scan_root_multi_kinds_sync(
    directory: str,
    kinds: List[ArtifactKind],
    prev_registries: Dict[ArtifactKind, Dict[str, ArtifactRecord]],
) -> Dict[ArtifactKind, Tuple[Dict[str, ArtifactRecord], List[Tuple[str, ArtifactRecord]], List[str]]]:
    """os.walk and os.stat once; dispatch to different kinds; only scan the same physical root directory (e.g., TUNE_BASE_DIR) once for each kind,
    classify and distribute in memory, eliminate 3x redundant disk traversal when multiple kinds share directories."""
    if not os.path.isdir(directory):
        return {k: ({}, [], list(prev_registries.get(k, {}).keys())) for k in kinds}

    new_regs: Dict[ArtifactKind, Dict[str, ArtifactRecord]] = {k: {} for k in kinds}
    events_by_k: Dict[ArtifactKind, List[Tuple[str, ArtifactRecord]]] = {k: [] for k in kinds}
    seen_by_k: Dict[ArtifactKind, Set[str]] = {k: set() for k in kinds}

    try:
        walk_iter = list(os.walk(directory))
    except OSError as e:
        logger.debug(f"[Watcher] Scan ({directory}) Failure: {e}")
        return {k: (prev_registries.get(k, {}), [], []) for k in kinds}

    for dirpath, dirnames, filenames in walk_iter:
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "__pycache__"]
        for fname in filenames:
            if fname.startswith(".") or ".tmp" in fname:
                continue

            matched_kind: Optional[ArtifactKind] = None
            for k in kinds:
                suffix = WATCH_SPECS[k][1]
                if fname.endswith(suffix) and _matches_kind(k, fname):
                    matched_kind = k
                    break
            if matched_kind is None:
                continue

            path = os.path.join(dirpath, fname)
            rel_key = os.path.relpath(path, directory).replace(os.sep, "/")
            try:
                st = os.stat(path)
            except OSError:
                continue

            seen_by_k[matched_kind].add(rel_key)
            prev_reg = prev_registries.get(matched_kind, {})
            prev = prev_reg.get(rel_key)

            is_new = prev is None
            is_modified = (
                not is_new
                and (st.st_mtime > prev.mtime + 1e-6 or st.st_size != prev.size)
            )
            if is_new or is_modified:
                summary = _parse_summary_sync(matched_kind, fname, path)
                record = ArtifactRecord(
                    kind=matched_kind,
                    name=rel_key,
                    path=path,
                    mtime=st.st_mtime,
                    size=st.st_size,
                    summary=summary,
                )
                new_regs[matched_kind][rel_key] = record
                events_by_k[matched_kind].append(
                    ("artifact_created" if is_new else "artifact_updated", record)
                )
            else:
                new_regs[matched_kind][rel_key] = prev

    out = {}
    for k in kinds:
        prev_reg = prev_registries.get(k, {})
        deleted = [name for name in prev_reg if name not in seen_by_k[k]]
        out[k] = (new_regs[k], events_by_k[k], deleted)
    return out


# ============================================================================
# GateWay and WebSocket BroadCast
# ============================================================================

class ResultWatcher:

    def __init__(self, connection_manager: Any, interval: float = DEFAULT_INTERVAL):
        self._manager = connection_manager
        self._interval = interval
        self._registry: Dict[ArtifactKind, Dict[str, ArtifactRecord]] = {
            k: {} for k in WATCH_SPECS
        }
        self._running = False
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------ #
    # lifetime
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(f"ResultWatcher Poll start and interval={self._interval}s)")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("ResultWatcher Stop")

    # ------------------------------------------------------------------ #
    # Loop
    # ------------------------------------------------------------------ #
    async def _poll_loop(self) -> None:
        first_scan = True
        while self._running:
            try:
                await self._scan_all(quiet=first_scan)
            except Exception as e:
                logger.error(f"[Watcher] Scan Exception: {e}", exc_info=True)
            first_scan = False
            await asyncio.sleep(self._interval)

    async def _scan_all(self, quiet: bool = False) -> None:
        # 按物理目录分组: 同一目录 (如 TUNE_BASE_DIR) 每轮仅 walk 1 次
        by_dir: Dict[str, List[ArtifactKind]] = {}
        for kind, (directory, _suffix) in WATCH_SPECS.items():
            by_dir.setdefault(directory, []).append(kind)

        for directory, kinds in by_dir.items():
            prev_slice = {k: self._registry[k] for k in kinds}
            batch_res = await asyncio.to_thread(
                _scan_root_multi_kinds_sync, directory, kinds, prev_slice
            )
            for kind in kinds:
                updated_reg, event_records, deleted_names = batch_res[kind]
                await self._apply_scan_results(
                    kind, updated_reg, event_records, deleted_names, quiet=quiet
                )

    async def _apply_scan_results(
        self,
        kind: ArtifactKind,
        updated_reg: Dict[str, ArtifactRecord],
        event_records: List[Tuple[str, ArtifactRecord]],
        deleted_names: List[str],
        quiet: bool = False,
    ) -> None:
        self._registry[kind] = updated_reg

        if quiet:
            return

        now_iso = datetime.now().isoformat()

        # 1. broadcast and update
        for event_type, record in event_records:
            payload = {
                "event": event_type,
                **record.to_payload(),
                "timestamp": now_iso,
            }
            await self._safe_broadcast(payload)

            if kind == ArtifactKind.COLLAPSE and record.summary:
                alert_payload = {
                    "event": "collapse_report",
                    **record.to_payload(),
                    "timestamp": now_iso,
                }
                await self._safe_broadcast(alert_payload)

        # 2. broadcast delete event
        for name in deleted_names:
            del_payload = {
                "event": "artifact_deleted",
                "kind": kind.value,
                "name": name,
                "timestamp": now_iso,
            }
            await self._safe_broadcast(del_payload)

    async def _safe_broadcast(self, message: Dict[str, Any]) -> None:
        try:
            await self._manager.broadcast_to_all(message)
        except Exception as e:
            logger.debug(f"[Watcher] WebSocket 广播异常: {e}")

    # ------------------------------------------------------------------ #
    # REST 
    # ------------------------------------------------------------------ #
    def snapshot(self, kind: str) -> List[Dict[str, Any]]:
        try:
            k = ArtifactKind(kind)
        except ValueError:
            return []
        return [
            r.to_payload()
            for r in sorted(self._registry[k].values(), key=lambda r: -r.mtime)
        ]

    async def get_detail(self, kind: str, name: str) -> Optional[Dict[str, Any]]:
        try:
            k = ArtifactKind(kind)
        except ValueError:
            return None

        record = self._registry[k].get(name)
        if record is None:
            return None

        detail = record.to_payload()

        if k in (ArtifactKind.COLLAPSE, ArtifactKind.LLM_RUN):
            data = await asyncio.to_thread(read_json_safe, record.path)
            if data is not None:
                detail["data"] = data
            else:
                detail["data_error"] = "failed to read json"

        return detail

    def snapshot_all(self) -> Dict[str, List[Dict[str, Any]]]:
        return {k.value: self.snapshot(k.value) for k in WATCH_SPECS}
