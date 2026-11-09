#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections import deque
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .indexer import MonitorIndex

router = APIRouter()
ws_router = APIRouter()
_index: Optional[MonitorIndex] = None
_index_lock = threading.Lock()


def get_index() -> MonitorIndex:
    global _index
    if _index is None:
        with _index_lock:
            if _index is None:
                _index = MonitorIndex()
    return _index


def _scan() -> MonitorIndex:
    idx = get_index()
    idx.scan_if_stale(min_interval=2.0)
    return idx


@router.get("/overview")
def overview():
    return _scan().overview()


@router.get("/runs")
def list_runs(limit: int = Query(50, ge=1, le=200), cursor: Optional[str] = None):
    return _scan().list_runs(limit=limit, cursor=cursor)


@router.get("/runs/{run_id}")
def get_run(run_id: str):
    data = _scan().get_run(run_id)
    return data or {"run_id": run_id, "state": {}, "steps": [], "pending_step": None}


@router.get("/usage/daily")
def usage_daily(day_from: Optional[str] = Query(None, alias="from"),
                day_to: Optional[str] = Query(None, alias="to")):
    return {"items": _scan().usage_daily(day_from, day_to)}


@router.get("/coverage")
def coverage():
    return _scan().coverage()


@router.get("/alerts")
def alerts():
    return {"items": _scan().alerts()}


@router.get("/features")
def list_features(limit: int = Query(100, ge=1, le=200), cursor: Optional[str] = None):
    return _scan().list_features(limit=limit, cursor=cursor)


@router.get("/features/{feature_id}")
def get_feature(feature_id: str):
    data = _scan().get_feature(feature_id)
    return data or {"feature_id": feature_id, "detail": "not indexed"}


class ReviewIn(BaseModel):
    fingerprint: str
    decision: str          # reviewed | rejected | promote_candidate
    note: str = ""
    operator: str = "anonymous"


@router.post("/review")
def post_review(body: ReviewIn):
    """唯一写接口: 人工批注落盘, 不触发任何下游消费。"""
    try:
        out = _scan().post_review(body.fingerprint, body.decision, body.note, body.operator)
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail=str(e))
    return out


class FocusIn(BaseModel):
    sid: str
    model_id: Optional[int] = None
    feature_id: str = ""


@router.get("/screener")
def get_screener(model_id: Optional[int] = None, limit: int = Query(200, ge=1, le=1000)):
    """Channel C: Return cross-sectional FSMPredictor trigger radar rows."""
    return _scan().screener(model_id=model_id, limit=limit)


@router.get("/focus")
def get_focus():
    """Return current cross-screen synchronized focus (sid, model_id, feature_id)."""
    return get_index().get_focus()


@router.post("/focus")
def post_focus(body: FocusIn):
    """Update cross-screen focus and fan out to all connected A/B/C WebSocket screens."""
    payload = get_index().set_focus(body.sid, body.model_id, body.feature_id)
    _HUB.broadcast({"seq": _HUB.next_seq(), "type": "focus", "ts": time.time(), "detail": payload})
    return payload


# ---------------------------------------------------------------------------
# WS v2: 单后台 watcher (scan+签名对比+事件环) + 每连接独立队列 + 背压
#
# [重构修正] 旧版每连接各自对比全局 _LAST_SIG —— 多客户端下后观察到变更的
# 连接会因 sig==_LAST_SIG 跳过推送 (事件丢失), 且同步 sqlite/scan 跑在
# event loop 上阻塞全服务。现改为: 唯一 watcher task (to_thread 扫描) 推
# 事件环并 fanout; 连接只消费自己的 asyncio.Queue, 溢出(>200)被踢。
# ---------------------------------------------------------------------------

class _WsHub:
    def __init__(self) -> None:
        self.events: deque = deque(maxlen=1000)
        self.seq: int = 0
        self.last_sig: Optional[tuple] = None
        self.watcher_task: Optional[asyncio.Task] = None
        self.subs: Dict[int, "asyncio.Queue[Dict[str, Any]]"] = {}
        self.sub_id: int = 0

    def next_seq(self) -> int:
        self.seq += 1
        return self.seq

    def register(self, q: "asyncio.Queue[Dict[str, Any]]") -> int:
        self.sub_id += 1
        self.subs[self.sub_id] = q
        return self.sub_id

    def broadcast(self, event: Dict[str, Any]) -> None:
        self.events.append(event)
        for q in list(self.subs.values()):
            try:
                if q.qsize() < _QUEUE_MAX:
                    q.put_nowait(event)
                else:
                    while not q.empty():
                        try:
                            q.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    q.put_nowait({"type": "resync_required"})
            except Exception:
                continue


_HUB = _WsHub()
_QUEUE_MAX = 200


def _sig_of(idx: MonitorIndex) -> tuple:
    with idx._conn() as c:
        row = c.execute("SELECT COUNT(*) AS n, COALESCE(MAX(mtime),0) AS m FROM artifacts").fetchone()
    return (int(row["n"]), float(row["m"]))


def _kinds_of(idx: MonitorIndex) -> Dict[str, int]:
    with idx._conn() as c:
        rows = c.execute("SELECT kind, COUNT(*) AS n FROM artifacts GROUP BY kind").fetchall()
    return {r["kind"]: r["n"] for r in rows}


async def _watcher_loop() -> None:
    idx = get_index()
    poll_s = float(os.environ.get("BT_STUDIO_MONITOR_WS_POLL", "5") or 5)
    while True:
        await asyncio.sleep(poll_s)
        try:
            await asyncio.to_thread(idx.scan_if_stale, 2.0)   # 不阻塞 event loop
            sig = await asyncio.to_thread(_sig_of, idx)
            if sig == _HUB.last_sig:
                continue
            _HUB.last_sig = sig
            kinds = await asyncio.to_thread(_kinds_of, idx)
            event = {"seq": _HUB.next_seq(), "type": "changed", "ts": time.time(),
                     "detail": {"artifacts": sig[0], "max_mtime": sig[1], "kinds": kinds}}
            _HUB.broadcast(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            continue


def _ensure_watcher() -> None:
    if _HUB.watcher_task is None or _HUB.watcher_task.done():
        _HUB.watcher_task = asyncio.get_event_loop().create_task(_watcher_loop())


def _ws_authorized(ws: WebSocket) -> bool:
    token = os.environ.get("BT_STUDIO_API_TOKEN", "")
    if not token:
        return os.environ.get("BT_STUDIO_ENV", "").lower() != "prod"
    if ws.headers.get("authorization", "") == f"Bearer {token}":
        return True
    return ws.query_params.get("token") == token


@ws_router.websocket("/ws/v2")
async def monitor_ws(ws: WebSocket):
    if not _ws_authorized(ws):
        await ws.close(code=4401)
        return
    await ws.accept()
    idx = get_index()
    await asyncio.to_thread(idx.scan_if_stale, 0.0)
    _ensure_watcher()

    last_seq_param = ws.query_params.get("last_seq")
    q: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAX)
    try:
        if last_seq_param and last_seq_param.isdigit():
            replayed = [e for e in _HUB.events if e["seq"] > int(last_seq_param)]
            await asyncio.wait_for(ws.send_json(
                {"type": "replay", "events": replayed}), timeout=5.0)
        else:
            await asyncio.wait_for(ws.send_json(
                {"type": "snapshot", "seq": _HUB.seq,
                 "data": await asyncio.to_thread(idx.overview)}), timeout=5.0)

        my = _HUB.register(q)
        try:
            while True:
                event = await q.get()
                await asyncio.wait_for(ws.send_json(event), timeout=5.0)  # 慢消费者 → 踢
        finally:
            _HUB.subs.pop(my, None)
    except (asyncio.TimeoutError, WebSocketDisconnect, RuntimeError):
        try:
            await ws.send_json({"type": "resync_required"})
            await ws.close(code=1013)
        except Exception:
            pass
    except Exception:
        try:
            await ws.close()
        except Exception:
            pass
