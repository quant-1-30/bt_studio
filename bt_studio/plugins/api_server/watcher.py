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
    TUNE_COLLAPSE_DIR,
    TUNE_MODEL_DIR,
    TUNE_SCORE_DIR,
)

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL = float(os.environ.get("BT_STUDIO_WATCH_INTERVAL", "5"))


class ArtifactKind(str, Enum):
    MODEL = "model"
    SCORE = "score"
    COLLAPSE = "collapse"
    LLM_RUN = "llm_run"
    FEATURE = "feature"


WATCH_SPECS: Dict[ArtifactKind, Tuple[str, str]] = {
    ArtifactKind.MODEL: (TUNE_MODEL_DIR, ".pkl"),
    ArtifactKind.SCORE: (TUNE_SCORE_DIR, ".parquet"),
    ArtifactKind.COLLAPSE: (TUNE_COLLAPSE_DIR, ".json"),
    ArtifactKind.LLM_RUN: (LLM_RUN_DIR, ".json"),
    ArtifactKind.FEATURE: (FEATURE_DIR, ".parquet"),
}

# re match avoid heavy io
_RE_MODEL = re.compile(r"^model_(\d+)\.pkl$")
_RE_SCORE = re.compile(r"^scores_(\d+)\.parquet$")
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
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "verdict": data.get("verdict"),
                "feature_col": data.get("feature_col"),
                "model_id": data.get("model_id"),
                "n_trials": data.get("n_trials"),
            }

        elif kind == ArtifactKind.LLM_RUN:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
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


def _scan_directory_sync(
    kind: ArtifactKind,
    directory: str,
    suffix: str,
    prev_registry: Dict[str, ArtifactRecord],
) -> Tuple[Dict[str, ArtifactRecord], List[Tuple[str, ArtifactRecord]], List[str]]:

    if not os.path.isdir(directory):
        return {}, [], []

    new_registry: Dict[str, ArtifactRecord] = {}
    events: List[Tuple[str, ArtifactRecord]] = []
    seen: Set[str] = set()

    try:
        entries = os.listdir(directory)
    except OSError as e:
        logger.debug(f"[Watcher] Scan ({directory}) Failure: {e}")
        return prev_registry, [], []

    for name in entries:
        # 1. filter .tmp and .* 
        if name.startswith(".") or name.endswith(".tmp") or not name.endswith(suffix):
            continue

        path = os.path.join(directory, name)
        try:
            st = os.stat(path)
        except OSError:
            continue

        seen.add(name)
        prev = prev_registry.get(name)

        is_new = prev is None
        is_modified = (
            not is_new
            and (st.st_mtime > prev.mtime + 1e-6 or st.st_size != prev.size)
        )

        if is_new or is_modified:
            # 原子写入保证此时解析读取 100% 安全
            summary = _parse_summary_sync(kind, name, path)
            record = ArtifactRecord(
                kind=kind,
                name=name,
                path=path,
                mtime=st.st_mtime,
                size=st.st_size,
                summary=summary,
            )
            new_registry[name] = record

            if is_new:
                events.append(("artifact_created", record))
            else:
                events.append(("artifact_updated", record))
        else:
            # 状态未改变，直接复用
            new_registry[name] = prev

    # 检测已删除产物
    deleted_names = [name for name in prev_registry if name not in seen]

    return new_registry, events, deleted_names


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
        logger.info(f"ResultWatcher 启动完成 (轮询间隔={self._interval}s)")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("ResultWatcher 已安全停止")

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
        for kind, (directory, suffix) in WATCH_SPECS.items():
            await self._scan_single_kind(kind, directory, suffix, quiet=quiet)

    async def _scan_single_kind(
        self,
        kind: ArtifactKind,
        directory: str,
        suffix: str,
        quiet: bool = False,
    ) -> None:
        # avoid I/O and Parser Block
        updated_reg, event_records, deleted_names = await asyncio.to_thread(
            _scan_directory_sync,
            kind,
            directory,
            suffix,
            self._registry[kind],
        )

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
    # REST 查询接口
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
            def _read_json():
                with open(record.path, "r", encoding="utf-8") as f:
                    return json.load(f)

            try:
                detail["data"] = await asyncio.to_thread(_read_json)
            except Exception as e:
                detail["data_error"] = str(e)

        return detail

    def snapshot_all(self) -> Dict[str, List[Dict[str, Any]]]:
        return {k.value: self.snapshot(k.value) for k in WATCH_SPECS}
