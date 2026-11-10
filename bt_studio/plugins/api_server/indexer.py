#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import threading
import time
import urllib.request
from datetime import datetime
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

from bt_studio.constant import RAW_PIT_VERSION, RESULT_ROOT
from bt_studio.utils.io import atomic_save_json, evict_cache_lru, read_json_safe as _read_json
from bt_studio.utils.months import parse_ym_from_path

logger = logging.getLogger(__name__)

_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS artifacts(
  path TEXT PRIMARY KEY, kind TEXT, run_id TEXT, feature_id TEXT,
  model_id INTEGER, ym INTEGER, size INTEGER, mtime REAL, inode INTEGER,
  content_sha1 TEXT, meta_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_art_kind_mtime ON artifacts(kind, mtime DESC);
CREATE INDEX IF NOT EXISTS idx_art_run ON artifacts(run_id, mtime);
CREATE INDEX IF NOT EXISTS idx_art_feature ON artifacts(feature_id, ym);
CREATE TABLE IF NOT EXISTS usage_daily(
  day TEXT, provider TEXT, model TEXT, calls INTEGER,
  prompt_tokens INTEGER, completion_tokens INTEGER, total_tokens INTEGER,
  errors INTEGER, latency_sum_ms REAL, latency_max_ms REAL,
  PRIMARY KEY(day, provider, model)
);
CREATE TABLE IF NOT EXISTS usage_offsets(path TEXT PRIMARY KEY, offset INTEGER);
CREATE TABLE IF NOT EXISTS review_marks(
  fingerprint TEXT PRIMARY KEY, decision TEXT, note TEXT,
  created_at TEXT, operator TEXT
);
CREATE INDEX IF NOT EXISTS idx_review_decision ON review_marks(decision);
"""


def _iter_clean_files(root: str, limit: int) -> Iterable[str]:
    """Yield valid artifact file paths under root up to limit, skipping temp/hidden files."""
    if not root or not os.path.isdir(root) or limit <= 0:
        return
    n = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git")]
        for name in filenames:
            if name.startswith(".") or ".tmp" in name or name.endswith((".consumed", ".invalid")):
                continue
            n += 1
            if n > limit:
                return
            yield os.path.join(dirpath, name)


def _parse_meta(row: sqlite3.Row) -> dict:
    try:
        return json.loads(row["meta_json"] or "{}")
    except Exception:
        return {}


def _sum_usage_rows(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    total = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "errors": 0}
    for r in rows:
        for k in total:
            total[k] += int(r.get(k, 0) or 0)
    return total


class MonitorIndex:
    """SQLite-backed index for runs, features, token usage, raw_pit coverage, and alerts."""

    def __init__(self, result_root: Optional[str] = None, db_path: Optional[str] = None):
        self.result_root = os.path.abspath(result_root or RESULT_ROOT)
        self.llm_run_dir = os.path.join(self.result_root, "artifacts", "runs")
        self.LLM_USAGE_DIR = os.path.join(self.result_root, "artifacts", "usage")
        self.feature_agent_dir = os.path.join(self.result_root, "artifacts", "features")
        self.raw_pit_dir = os.path.join(self.result_root, "cache", "bars", RAW_PIT_VERSION)
        self.tune_base_dir = os.path.join(self.result_root, "artifacts", "tune")
        self.db_path = db_path or os.path.join(self.result_root, "cache", "monitor_index.sqlite")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._write_lock = threading.Lock()
        self._last_scan_ts = 0.0
        with self._write_lock, self._conn() as c:
            c.executescript(_SCHEMA_DDL)

    @contextlib.contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
            with conn:
                yield conn
        finally:
            conn.close()

    def _upsert(
        self,
        c: sqlite3.Connection,
        path: str,
        kind: str,
        meta: Optional[dict] = None,
        parse_sha: bool = False,
        **scope: Any,
    ) -> None:
        st = os.stat(path)
        size, mtime, inode = int(st.st_size), float(st.st_mtime), int(getattr(st, "st_ino", 0))
        old = c.execute("SELECT size,mtime,inode FROM artifacts WHERE path=?", (path,)).fetchone()
        if old and int(old["size"]) == size and float(old["mtime"]) == mtime and int(old["inode"]) == inode:
            return
        content_sha1 = ""
        if parse_sha and size <= 2_000_000:
            with contextlib.suppress(Exception), open(path, "rb") as f:
                content_sha1 = hashlib.sha1(f.read()).hexdigest()
        c.execute(
            """INSERT INTO artifacts(path,kind,run_id,feature_id,model_id,ym,size,mtime,inode,content_sha1,meta_json)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(path) DO UPDATE SET kind=excluded.kind,run_id=excluded.run_id,
                 feature_id=excluded.feature_id,model_id=excluded.model_id,ym=excluded.ym,
                 size=excluded.size,mtime=excluded.mtime,inode=excluded.inode,
                 content_sha1=excluded.content_sha1,meta_json=excluded.meta_json""",
            (
                path,
                kind,
                scope.get("run_id", ""),
                scope.get("feature_id", ""),
                scope.get("model_id"),
                scope.get("ym"),
                size,
                mtime,
                inode,
                content_sha1,
                json.dumps(meta or {}, ensure_ascii=False, default=str),
            ),
        )

    # ------------------------------------------------------------------ #
    # Unified Artifact & Usage Scanning
    # ------------------------------------------------------------------ #
    def scan_if_stale(self, min_interval: float = 2.0, max_files: int = 200_000) -> None:
        if time.time() - self._last_scan_ts < min_interval:
            return
        self.scan_once(max_files=max_files)

    def scan_once(self, max_files: int = 200_000) -> None:
        with self._write_lock:
            remaining = max_files
            with self._conn() as c:
                for domain, root in (
                    ("runs", self.llm_run_dir),
                    ("features", self.feature_agent_dir),
                    ("raw_pit", self.raw_pit_dir),
                    ("tune", self.tune_base_dir),
                ):
                    if remaining <= 0:
                        break
                    for p in _iter_clean_files(root, remaining):
                        if self._ingest_artifact_file(c, domain, p):
                            remaining -= 1
                self._scan_usage(c)
            self._last_scan_ts = time.time()

    def _ingest_artifact_file(self, c: sqlite3.Connection, domain: str, p: str) -> bool:
        base = os.path.basename(p)
        if domain == "runs":
            run_id = os.path.basename(os.path.dirname(p))
            if base == "state.json":
                data = _read_json(p) or {}
                self._upsert(c, p, "run_state", run_id=run_id, parse_sha=True, meta={
                    "step_counter": data.get("step_counter", 0),
                    "config_fingerprint": data.get("config_fingerprint", ""),
                    "target": data.get("target", ""),
                })
            elif base.startswith("step_") and base.endswith(".json"):
                data = _read_json(p) or {}
                rb = data.get("replay_buffer") or {}
                fps = [
                    str(f.get("fingerprint") or f.get("ast_fingerprint"))
                    for f in (data.get("features") or [])
                    if f.get("fingerprint") or f.get("ast_fingerprint")
                ]
                self._upsert(c, p, "run_step", run_id=run_id, parse_sha=True, meta={
                    "step_index": data.get("step_index", 0),
                    "hypothesis_id": data.get("hypothesis_id", ""),
                    "n_features": len(data.get("features", []) or []),
                    "n_success": len(rb.get("successful_history") or []),
                    "n_failure": len(rb.get("failed_history") or []),
                    "fingerprints": sorted(set(fps)),
                })
            elif base == "pending_step.json":
                self._upsert(c, p, "run_pending", run_id=run_id, meta=_read_json(p) or {}, parse_sha=True)
            else:
                return False
            return True

        if domain == "features":
            if base == "viable_features.json":
                data = _read_json(p) or {}
                for item in (data.get("features") or []):
                    fid = str(item.get("fingerprint") or item.get("feature_id") or "")
                    if fid:
                        self._upsert(c, p, "feature_manifest", feature_id=fid, parse_sha=True, meta={
                            "display_name": item.get("display_name") or item.get("feature_name") or fid,
                            "fingerprint": fid,
                            "final_score": item.get("final_score", -500.0),
                            "window_pass_ratio": item.get("window_pass_ratio", 0.0),
                        })
                return True
            if base.startswith("hf_") and base.endswith(".parquet"):
                self._upsert(
                    c, p, "feature_parquet",
                    feature_id=os.path.basename(os.path.dirname(p)),
                    ym=parse_ym_from_path(base),
                )
                return True
            return False

        if domain == "raw_pit":
            if not (base.startswith("ym=") and base.endswith(".parquet")):
                return False
            ym = int(base[3:-len(".parquet")]) if base[3:-len(".parquet")].isdigit() else None
            self._upsert(c, p, "raw_pit", feature_id="pack", ym=ym)
            return True

        if domain == "tune":
            if base.startswith("model_") and base.endswith(".manifest.json"):
                data = _read_json(p) or {}
                mid = data.get("model_id")
                mid_int = int(mid) if str(mid).isdigit() else None
                self._upsert(c, p, "model_manifest", feature_id=str(data.get("feature_id") or ""),
                             model_id=mid_int, parse_sha=True, meta={
                                 "source": data.get("source", ""),
                                 "ast_fingerprint": data.get("ast_fingerprint", ""),
                                 "task_id": data.get("task_id", ""),
                                 "label_basis": data.get("label_basis", {}),
                                 "gates": data.get("gates", {}),
                             })
                return True
            if base.startswith("model_") and base.endswith(".pkl"):
                stem = base[len("model_"):-len(".pkl")]
                if stem.isdigit():
                    self._upsert(c, p, "model_pickle", model_id=int(stem))
                    return True
            elif base.startswith("scores_") and base.endswith(".parquet"):
                stem = base[len("scores_"):-len(".parquet")]
                if stem.isdigit():
                    self._upsert(c, p, "score_parquet", model_id=int(stem))
                    return True
            elif base.startswith("collapse_") and base.endswith(".json"):
                data = _read_json(p) or {}
                self._upsert(c, p, "collapse", feature_id=str(data.get("feature_col") or ""),
                             model_id=data.get("model_id") if isinstance(data.get("model_id"), int) else None,
                             meta={"verdict": data.get("verdict", ""), "n_trials": data.get("n_trials", 0)}, parse_sha=True)
                return True
        return False

    def _scan_usage(self, c: sqlite3.Connection) -> int:
        if not os.path.isdir(self.LLM_USAGE_DIR):
            return 0
        n = 0
        for name in sorted(os.listdir(self.LLM_USAGE_DIR)):
            if not (name.startswith("usage_") and name.endswith(".jsonl")):
                continue
            day = name[len("usage_"):-len(".jsonl")]
            path = os.path.join(self.LLM_USAGE_DIR, name)
            offset = int((c.execute("SELECT offset FROM usage_offsets WHERE path=?", (path,)).fetchone() or {"offset": 0})["offset"])
            if offset > os.path.getsize(path):
                offset = 0
                c.execute("DELETE FROM usage_daily WHERE day=?", (day,))
            with open(path, "rb") as f:
                f.seek(offset)
                raw_chunk = f.read()
            if not raw_chunk:
                continue
            last_nl = raw_chunk.rfind(b"\n")
            if last_nl == -1:
                continue
            complete_bytes = raw_chunk[: last_nl + 1]
            new_offset = offset + len(complete_bytes)
            for line in complete_bytes.decode("utf-8", errors="replace").splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                u = rec.get("usage") or {}
                latency = float(rec.get("latency_ms") or 0.0)
                c.execute(
                    """INSERT INTO usage_daily(day,provider,model,calls,prompt_tokens,completion_tokens,total_tokens,errors,latency_sum_ms,latency_max_ms)
                       VALUES(?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(day,provider,model) DO UPDATE SET
                         calls=calls+1,
                         prompt_tokens=prompt_tokens+excluded.prompt_tokens,
                         completion_tokens=completion_tokens+excluded.completion_tokens,
                         total_tokens=total_tokens+excluded.total_tokens,
                         errors=errors+excluded.errors,
                         latency_sum_ms=latency_sum_ms+excluded.latency_sum_ms,
                         latency_max_ms=MAX(latency_max_ms, excluded.latency_max_ms)""",
                    (day, str(rec.get("provider") or "unknown"), str(rec.get("model") or "unknown"), 1,
                     int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0),
                     int(u.get("total_tokens") or 0), 1 if str(rec.get("status") or "ok") != "ok" else 0,
                     latency, latency),
                )
                n += 1
            c.execute(
                "INSERT INTO usage_offsets(path,offset) VALUES(?,?) ON CONFLICT(path) DO UPDATE SET offset=excluded.offset",
                (path, int(new_offset)),
            )
        return n

    # ------------------------------------------------------------------ #
    # Unified Read/Write Domain Queries
    # ------------------------------------------------------------------ #
    def runs(self, run_id: Optional[str] = None, limit: int = 50, cursor: Optional[str] = None) -> Any:
        """Query a single run by run_id or list paginated runs when run_id is None."""
        if run_id is not None:
            with self._conn() as c:
                rows = c.execute(
                    "SELECT kind,path,mtime,meta_json FROM artifacts WHERE run_id=? ORDER BY mtime",
                    (run_id,),
                ).fetchall()
                if not rows:
                    return None
                state, steps, pending = {}, [], None
                for r in rows:
                    meta = _parse_meta(r)
                    if r["kind"] == "run_state":
                        state = meta
                    elif r["kind"] == "run_step":
                        steps.append({"path": r["path"], "mtime": r["mtime"], **meta})
                    elif r["kind"] == "run_pending":
                        pending = {"path": r["path"], "mtime": r["mtime"], **meta}
                return {"run_id": run_id, "state": state, "steps": steps, "pending_step": pending}

        limit = max(1, min(int(limit), 200))
        where, params = "", []
        if cursor and "|" in cursor:
            with contextlib.suppress(ValueError):
                mtime_s, rid = cursor.split("|", 1)
                where = "AND (mtime < ? OR (mtime = ? AND run_id < ?))"
                params = [float(mtime_s), float(mtime_s), rid]
        sql = f"""
          SELECT run_id, MAX(mtime) AS mtime,
                 SUM(kind='run_step') AS n_steps,
                 SUM(kind='run_pending') AS has_pending
          FROM artifacts
          WHERE run_id != '' {where}
          GROUP BY run_id
          ORDER BY mtime DESC, run_id DESC
          LIMIT ?
        """
        with self._conn() as c:
            rows = c.execute(sql, (*params, limit + 1)).fetchall()
            out = []
            for r in rows[:limit]:
                rid = r["run_id"]
                state = c.execute(
                    "SELECT meta_json FROM artifacts WHERE run_id=? AND kind='run_state' ORDER BY mtime DESC LIMIT 1",
                    (rid,),
                ).fetchone()
                last_step = c.execute(
                    "SELECT meta_json,mtime FROM artifacts WHERE run_id=? AND kind='run_step' ORDER BY mtime DESC LIMIT 1",
                    (rid,),
                ).fetchone()
                sm = json.loads(state["meta_json"]) if state and state["meta_json"] else {}
                lm = json.loads(last_step["meta_json"]) if last_step and last_step["meta_json"] else {}
                out.append({
                    "run_id": rid,
                    "mtime": r["mtime"],
                    "step_counter": int(sm.get("step_counter", 0) or 0),
                    "config_fingerprint": sm.get("config_fingerprint", ""),
                    "target": sm.get("target", ""),
                    "n_steps": int(r["n_steps"] or 0),
                    "n_success": int(lm.get("n_success", 0) or 0),
                    "n_failure": int(lm.get("n_failure", 0) or 0),
                    "has_pending": bool(r["has_pending"]),
                    "last_step_index": int(lm.get("step_index", 0) or 0),
                })
            next_cursor = f"{rows[limit - 1]['mtime']:.6f}|{rows[limit - 1]['run_id']}" if len(rows) > limit else None
            return {"items": out, "next_cursor": next_cursor}

    def list_runs(self, limit: int = 50, cursor: Optional[str] = None) -> Dict[str, Any]:
        return self.runs(run_id=None, limit=limit, cursor=cursor)

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        return self.runs(run_id=run_id)

    def features(self, feature_id: Optional[str] = None, limit: int = 100, cursor: Optional[str] = None) -> Any:
        """Query a single feature by feature_id or list paginated features when feature_id is None."""
        if feature_id is not None:
            with self._conn() as c:
                man = c.execute(
                    "SELECT meta_json, content_sha1 FROM artifacts WHERE kind='feature_manifest' AND feature_id=? ORDER BY mtime DESC LIMIT 1",
                    (feature_id,),
                ).fetchone()
                if not man:
                    return None
                parquets = c.execute(
                    "SELECT path, ym, size, mtime FROM artifacts WHERE kind='feature_parquet' AND feature_id=? ORDER BY ym",
                    (feature_id[:12],),
                ).fetchall()
                rev = c.execute(
                    "SELECT decision, note, operator, created_at FROM review_marks WHERE fingerprint=?",
                    (feature_id,),
                ).fetchone()
                return {
                    "feature_id": feature_id,
                    **_parse_meta(man),
                    "parquets": [{k: r[k] for k in ("path", "ym", "size", "mtime")} for r in parquets],
                    "review": (dict(rev) if rev else None),
                }

        limit = max(1, min(int(limit), 200))
        where, params = "", []
        if cursor and "|" in cursor:
            with contextlib.suppress(ValueError):
                mtime_s, fid = cursor.split("|", 1)
                where = "AND (a.mtime < ? OR (a.mtime = ? AND a.feature_id < ?))"
                params = [float(mtime_s), float(mtime_s), fid]
        with self._conn() as c:
            rows = c.execute(
                f"""SELECT a.feature_id, a.mtime, a.meta_json,
                           r.decision AS review_decision, r.note AS review_note, r.operator AS review_operator
                    FROM artifacts a LEFT JOIN review_marks r ON r.fingerprint = a.feature_id
                    WHERE a.kind='feature_manifest' AND a.feature_id != '' {where}
                    ORDER BY a.mtime DESC, a.feature_id DESC LIMIT ?""",
                (*params, limit + 1),
            ).fetchall()
            items = []
            for row in rows[:limit]:
                m = _parse_meta(row)
                items.append({
                    "feature_id": row["feature_id"],
                    "display_name": m.get("display_name", row["feature_id"]),
                    "final_score": m.get("final_score"),
                    "window_pass_ratio": m.get("window_pass_ratio"),
                    "mtime": row["mtime"],
                    "review": ({"decision": row["review_decision"], "note": row["review_note"],
                                "operator": row["review_operator"]} if row["review_decision"] else None),
                })
            next_cursor = f"{rows[limit - 1]['mtime']:.6f}|{rows[limit - 1]['feature_id']}" if len(rows) > limit else None
            return {"items": items, "next_cursor": next_cursor}

    def list_features(self, limit: int = 100, cursor: Optional[str] = None) -> Dict[str, Any]:
        return self.features(feature_id=None, limit=limit, cursor=cursor)

    def get_feature(self, feature_id: str) -> Optional[Dict[str, Any]]:
        return self.features(feature_id=feature_id)

    def post_review(self, fingerprint: str, decision: str, note: str, operator: str) -> Dict[str, Any]:
        if decision not in ("reviewed", "rejected", "promote_candidate"):
            raise ValueError(f"invalid decision: {decision}")
        created_at = datetime.now().isoformat(timespec="seconds")
        with self._write_lock, self._conn() as c:
            c.execute(
                """INSERT INTO review_marks(fingerprint,decision,note,created_at,operator)
                   VALUES(?,?,?,?,?)
                   ON CONFLICT(fingerprint) DO UPDATE SET decision=excluded.decision,
                     note=excluded.note, operator=excluded.operator, created_at=excluded.created_at""",
                (fingerprint, decision, note, created_at, operator),
            )
        review_dir = os.path.join(self.result_root, "artifacts", "review")
        os.makedirs(review_dir, exist_ok=True)
        path = os.path.join(review_dir, f"{fingerprint}.json")
        atomic_save_json({"fingerprint": fingerprint, "decision": decision, "note": note,
                          "operator": operator, "created_at": created_at}, path)
        return {"fingerprint": fingerprint, "decision": decision, "path": path, "created_at": created_at}

    def review_mark(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            r = c.execute(
                "SELECT decision,note,operator,created_at FROM review_marks WHERE fingerprint=?",
                (fingerprint,),
            ).fetchone()
            return dict(r) if r else None

    def usage_daily(self, day_from: Optional[str] = None, day_to: Optional[str] = None) -> List[Dict[str, Any]]:
        where, params = "", []
        if day_from:
            where += " AND day >= ?"
            params.append(day_from)
        if day_to:
            where += " AND day <= ?"
            params.append(day_to)
        with self._conn() as c:
            rows = c.execute(
                f"""SELECT day,provider,model,calls,prompt_tokens,completion_tokens,total_tokens,errors,latency_sum_ms,latency_max_ms
                     FROM usage_daily WHERE 1=1 {where} ORDER BY day,provider,model""",
                params,
            ).fetchall()
            return [{
                **{k: r[k] for k in ("day", "provider", "model", "calls", "prompt_tokens", "completion_tokens", "total_tokens", "errors")},
                "latency_avg_ms": (r["latency_sum_ms"] / r["calls"]) if r["calls"] else 0.0,
                "latency_max_ms": r["latency_max_ms"],
            } for r in rows]

    def coverage(self) -> Dict[str, Any]:
        with self._conn() as c:
            r = c.execute(
                "SELECT COUNT(*) AS cells, COUNT(DISTINCT feature_id) AS sids, COUNT(DISTINCT ym) AS months, MAX(ym) AS latest_ym FROM artifacts WHERE kind='raw_pit'"
            ).fetchone()
            if not r or not r["cells"]:
                return {"status": "unknown", "requested_cells": 0, "existing_cells": 0,
                        "coverage": 0.0, "missing_preview": [], "note": "no raw_pit packs indexed"}
            raw_rows = c.execute("SELECT path FROM artifacts WHERE kind='raw_pit'").fetchall()
        total_cov = 0
        total_req = 0
        for row in raw_rows:
            meta_obj = _read_json(row["path"][:-len(".parquet")] + ".meta.json")
            if isinstance(meta_obj, dict):
                cov_n = len(meta_obj.get("covered_sids", []))
                total_cov += cov_n
                total_req += max(cov_n, int(meta_obj.get("requested_sids", cov_n) or cov_n))
        return {
            "status": "ok",
            "requested_cells": total_req,
            "existing_cells": total_cov,
            "coverage": (total_cov / total_req) if total_req else 1.0,
            "missing_preview": [],
            "note": f"raw_pit packs={r['cells']} months={r['months']} covered_sid_months={total_cov} latest_ym={r['latest_ym']}",
        }

    # ------------------------------------------------------------------ #
    # Alerts, Cache Eviction & Overview
    # ------------------------------------------------------------------ #
    def _evict_eval_cache_lru(self, max_remove: int = 500, include_bars: bool = True) -> int:
        return evict_cache_lru(result_root=self.result_root, max_remove=max_remove, include_bars=include_bars)

    def _notify_webhook_if_needed(self, alerts_list: List[Dict[str, Any]]) -> None:
        webhook_url = os.environ.get("BT_STUDIO_ALERT_WEBHOOK", "").strip()
        notable = [a for a in (alerts_list or []) if a.get("level") in ("warn", "critical")]
        if not webhook_url or not notable:
            return
        sig = tuple(sorted((a.get("code", ""), a.get("level", ""), a.get("message", "")) for a in notable))
        if getattr(self, "_last_webhook_sig", None) == sig:
            return
        try:
            payload = json.dumps({
                "service": "bt_studio",
                "ts": datetime.now().isoformat(timespec="seconds"),
                "alerts": notable,
            }, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(
                webhook_url, data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=3.0):
                pass
            self._last_webhook_sig = sig
        except Exception as e:
            logger.warning(f"[MonitorIndex] Alert webhook delivery failed ({webhook_url}): {e}")

    def _check_stage3_and_pending_alerts(self, out: List[Dict[str, Any]]) -> None:
        pending_dir = os.path.join(self.result_root, "artifacts", "pending")
        entries = os.listdir(pending_dir) if os.path.isdir(pending_dir) else []

        if os.environ.get("AGENT_STAGE3", "spool").lower() == "off":
            pending = sum(1 for n in entries if n.endswith(".json"))
            with self._conn() as c:
                tune = c.execute(
                    "SELECT COUNT(*) AS n FROM artifacts WHERE kind IN ('model_pickle','model_manifest','score_parquet','collapse') AND mtime >= ?",
                    (time.time() - 86400.0,),
                ).fetchone()["n"]
            if pending or tune:
                out.append({
                    "code": "forbidden_pending_tune",
                    "level": "critical",
                    "message": f"AGENT_STAGE3=off but recent pending/tune artifacts exist (pending={pending}, tune_24h={tune})",
                    "meta": {"pending": pending, "tune": tune},
                })

        invalids = sorted(n for n in entries if n.endswith(".invalid"))
        if invalids:
            out.append({
                "code": "pending_invalid_quarantined",
                "level": "warn",
                "message": f"{len(invalids)} invalid pending task(s) quarantined in pending/",
                "meta": {"files": invalids[:10]},
            })

    def _check_disk_usage_alert(self, out: List[Dict[str, Any]]) -> None:
        with contextlib.suppress(Exception):
            target_dir = self.result_root if os.path.exists(self.result_root) else os.path.dirname(self.result_root)
            if not os.path.exists(target_dir):
                return
            du = shutil.disk_usage(target_dir)
            used_ratio = du.used / du.total if du.total > 0 else 0.0
            warn_thr = float(os.environ.get("BT_STUDIO_DISK_WARN_RATIO", "0.85") or 0.85)
            crit_thr = float(os.environ.get("BT_STUDIO_DISK_CRIT_RATIO", "0.95") or 0.95)
            if used_ratio >= crit_thr:
                evicted = self._evict_eval_cache_lru(max_remove=500, include_bars=True)
                out.append({
                    "code": "disk_usage_critical",
                    "level": "critical",
                    "message": f"result volume disk usage {used_ratio:.1%} >= {crit_thr:.0%} (auto-evicted {evicted} cache files)",
                    "meta": {"used_gb": round(du.used / (1024**3), 2), "free_gb": round(du.free / (1024**3), 2), "evicted": evicted},
                })
            elif used_ratio >= warn_thr:
                out.append({
                    "code": "disk_usage_high",
                    "level": "warn",
                    "message": f"result volume disk usage {used_ratio:.1%} >= {warn_thr:.0%}",
                    "meta": {"used_gb": round(du.used / (1024**3), 2), "free_gb": round(du.free / (1024**3), 2)},
                })

    def _check_repeat_fingerprint_alert(self, out: List[Dict[str, Any]]) -> None:
        with contextlib.suppress(Exception):
            with self._conn() as c:
                rows = c.execute(
                    "SELECT meta_json, run_id FROM artifacts WHERE kind='run_step' AND mtime >= ?",
                    (time.time() - 86400.0,),
                ).fetchall()
            fp_runs: Dict[str, set] = {}
            for r in rows:
                for fp in (_parse_meta(r).get("fingerprints") or []):
                    fp_runs.setdefault(fp, set()).add(r["run_id"])
            repeats = {fp: sorted(v) for fp, v in fp_runs.items() if len(v) >= 2}
            if repeats:
                out.append({
                    "code": "repeat_fingerprint",
                    "level": "warn",
                    "message": f"{len(repeats)} fingerprints re-explored across runs in 24h (ReplayBuffer cross-run gap)",
                    "meta": {"examples": dict(list(repeats.items())[:5])},
                })

    def _check_index_lag_alert(self, out: List[Dict[str, Any]]) -> None:
        with self._conn() as c:
            newest = c.execute("SELECT MAX(mtime) AS m FROM artifacts").fetchone()["m"]
        disk_newest = 0.0
        for sub in (self.llm_run_dir, self.LLM_USAGE_DIR):
            if not os.path.isdir(sub):
                continue
            for dirpath, _dirs, files in os.walk(sub):
                for name in files:
                    with contextlib.suppress(OSError):
                        disk_newest = max(disk_newest, os.path.getmtime(os.path.join(dirpath, name)))
        idx_age = (time.time() - float(newest)) if newest else 0.0
        disk_age = (time.time() - disk_newest) if disk_newest else float("inf")
        if newest and disk_newest and idx_age - disk_age > 60.0:
            with contextlib.suppress(Exception):
                self.scan_once()
            out.append({
                "code": "index_lag",
                "level": "info",
                "message": f"index trailed disk by {int(idx_age - disk_age)}s (auto-rescan triggered)",
                "meta": {"newest_mtime": newest, "disk_newest": disk_newest},
            })

    def alerts(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        today = datetime.now().strftime("%Y%m%d")
        usage = _sum_usage_rows(self.usage_daily(today, today))
        budget_s = os.environ.get("LLM_DAILY_TOKEN_BUDGET", "")
        budget = int(budget_s) if budget_s.isdigit() else None
        if budget and usage["total_tokens"] >= int(budget * 0.8):
            out.append({
                "code": "token_budget_80",
                "level": "warn" if usage["total_tokens"] < budget else "critical",
                "message": f"today tokens {usage['total_tokens']}/{budget} >= 80%",
                "meta": usage,
            })

        runs = [r for r in self.list_runs(limit=5)["items"] if not r.get("has_pending")]
        if len(runs) >= 2 and all(r["n_success"] == 0 and r["n_failure"] > 0 for r in runs[:2]):
            out.append({
                "code": "stage2_zero_success_streak",
                "level": "warn",
                "message": "latest 2 completed runs have 0 success with failures recorded",
                "meta": {"runs": [r["run_id"] for r in runs[:2]]},
            })

        self._check_stage3_and_pending_alerts(out)
        self._check_disk_usage_alert(out)
        self._check_repeat_fingerprint_alert(out)

        with contextlib.suppress(Exception):
            cov = self.coverage()
            min_cov = float(os.environ.get("RAW_PIT_MIN_COVERAGE", "0.0") or 0.0)
            if cov.get("status") == "ok" and cov.get("coverage", 1.0) < min_cov:
                out.append({
                    "code": "raw_pit_coverage_low",
                    "level": "warn",
                    "message": f"raw_pit coverage {cov['coverage']:.1%} < {min_cov:.0%}",
                    "meta": cov,
                })

        self._check_index_lag_alert(out)
        self._notify_webhook_if_needed(out)
        return out

    def overview(self) -> Dict[str, Any]:
        today = datetime.now().strftime("%Y%m%d")
        usage = _sum_usage_rows(self.usage_daily(today, today))
        budget_s = os.environ.get("LLM_DAILY_TOKEN_BUDGET", "")
        budget = int(budget_s) if budget_s.isdigit() else None
        with self._conn() as c:
            active = c.execute(
                "SELECT COUNT(DISTINCT run_id) AS n FROM artifacts WHERE run_id != '' AND mtime >= ?",
                (time.time() - 86400.0,),
            ).fetchone()["n"]
            feats = c.execute(
                "SELECT COUNT(DISTINCT feature_id) AS n FROM artifacts WHERE kind='feature_manifest'"
            ).fetchone()["n"]
        return {
            "day": today,
            "usage": usage,
            "usage_all_time": _sum_usage_rows(self.usage_daily()),
            "token_budget": budget,
            "token_budget_used_ratio": (usage["total_tokens"] / budget) if budget else None,
            "active_runs_24h": int(active or 0),
            "viable_features": int(feats or 0),
            "raw_pit": self.coverage(),
            "alerts": self.alerts(),
        }

    def screener(self, model_id: Optional[int] = None, limit: int = 200) -> Dict[str, Any]:
        """Channel C: Return cross-sectional FSMPredictor trigger rows from the latest (or specified) score parquet."""
        import polars as pl

        with self._conn() as c:
            if model_id is not None:
                row = c.execute(
                    "SELECT path, model_id, mtime FROM artifacts WHERE kind='score_parquet' AND model_id=? ORDER BY mtime DESC LIMIT 1",
                    (int(model_id),),
                ).fetchone()
            else:
                row = c.execute(
                    "SELECT path, model_id, mtime FROM artifacts WHERE kind='score_parquet' ORDER BY model_id DESC, mtime DESC LIMIT 1"
                ).fetchone()
        if not row or not os.path.exists(row["path"]):
            return {"model_id": model_id, "items": []}

        try:
            df = pl.read_parquet(row["path"])
            if "sid" in df.columns and df["sid"].dtype == pl.Binary:
                df = df.with_columns(pl.col("sid").cast(pl.String))
            if "day" in df.columns:
                df = df.with_columns(pl.col("day").cast(pl.String))
            if "fsm_score" in df.columns:
                df = df.sort("fsm_score", descending=True)
            return {
                "model_id": row["model_id"],
                "path": row["path"],
                "mtime": row["mtime"],
                "items": df.head(max(1, min(int(limit), 1000))).to_dicts(),
            }
        except Exception:
            return {"model_id": row["model_id"], "items": []}

    @property
    def _focus_path(self) -> str:
        return os.path.join(self.result_root, "cache", "desk_focus.json")

    def get_focus(self) -> Dict[str, Any]:
        """Return the current cross-screen synchronized focus (sid, model_id, feature_id)."""
        data = _read_json(self._focus_path)
        return data if isinstance(data, dict) else {"sid": "600000", "model_id": None, "feature_id": ""}

    def set_focus(self, sid: str, model_id: Optional[int] = None, feature_id: str = "") -> Dict[str, Any]:
        """Persist and broadcast cross-screen focus selection across A/B/C screens."""
        payload = {
            "sid": str(sid).strip(),
            "model_id": int(model_id) if model_id is not None else None,
            "feature_id": str(feature_id).strip(),
            "updated_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        atomic_save_json(payload, self._focus_path)
        return payload
