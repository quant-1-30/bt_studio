#! /usr/bin/env python3

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Any, Deque, Dict, List, Optional, Set

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .monitor_api import router as monitor_v2_router, ws_router as monitor_ws_router
from .watcher import ArtifactKind, ResultWatcher, WATCH_SPECS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("api_server")


# =============================================================================
# WebSocket Connection Managerment and support Multi Client
# =============================================================================

class ConnectionManager:

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        await websocket.accept()

        async with self._lock:
            if client_id not in self.active_connections:
                self.active_connections[client_id] = []
            self.active_connections[client_id].append(websocket)
        logger.info(f"{client_id} connected and present Connections {len(self.active_connections[client_id])}")

    async def disconnect(self, websocket: WebSocket, client_id: str) -> None:
        async with self._lock:
            if client_id in self.active_connections:
                if websocket in self.active_connections[client_id]:
                    self.active_connections[client_id].remove(websocket)
                if not self.active_connections[client_id]:
                    del self.active_connections[client_id]
        logger.info(f"disconnected: {client_id}")

    async def broadcast_to_all(self, message: Dict[str, Any]) -> None:
        async with self._lock:
            targets: List[tuple[str, WebSocket]] = [
                (client_id, ws)
                for client_id, ws_list in self.active_connections.items()
                for ws in ws_list
            ]

        if not targets:
            return

        # async push
        async def _safe_send(cid: str, ws: WebSocket) -> Optional[tuple[str, WebSocket]]:
            try:
                await ws.send_json(message)
                return None
            except Exception as e:
                logger.debug(f"向客户端 [{cid}] 推送失败 (连接可能已断开): {e}")
                return (cid, ws)

        tasks = [_safe_send(cid, ws) for cid, ws in targets]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        # clean dead connection
        dead_connections = [r for r in results if r is not None]
        if dead_connections:
            async with self._lock:
                for cid, ws in dead_connections:
                    if cid in self.active_connections and ws in self.active_connections[cid]:
                        self.active_connections[cid].remove(ws)
                        if not self.active_connections[cid]:
                            del self.active_connections[cid]
            logger.info(f"has cleaned {len(dead_connections)} expired WebSocket Connection")


manager = ConnectionManager()
watcher = ResultWatcher(manager)

# =============================================================================
# FastAPI LifeTime
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("bt_studio API Server and Thin Gateway live")
    await watcher.start()
    yield
    logger.info("bt_studio API Server Stop...")
    await watcher.stop()


app = FastAPI(
    title="bt_studio API",
    description="FSM Strategy API GateWay and WebSocket Push Service",
    version="0.2.0",
    lifespan=lifespan,
)

# M0 shadow: index-first read-only monitor; legacy watcher endpoints remain untouched.
app.include_router(monitor_v2_router, prefix="/api/v2", tags=["monitor-v2"])
app.include_router(monitor_ws_router, tags=["monitor-v2-ws"])

# CORS / Auth (生产默认收紧; BT_STUDIO_ENV=prod 时强制要求 BT_STUDIO_API_TOKEN)
_API_TOKEN = os.environ.get("BT_STUDIO_API_TOKEN", "")
if os.environ.get("BT_STUDIO_ENV", "").lower() == "prod" and not _API_TOKEN:
    raise RuntimeError(
        "[api_server] BT_STUDIO_ENV=prod 模式下必须显式配置 BT_STUDIO_API_TOKEN 环境变量"
    )
_CORS_ORIGINS = [
    o.strip() for o in os.environ.get(
        "BT_STUDIO_CORS_ORIGINS", "http://localhost,http://127.0.0.1"
    ).split(",") if o.strip()
]
_ALLOW_CREDENTIALS = _CORS_ORIGINS != ["*"]  # credentials 与 wildcard 不共存

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _http_authorized(headers) -> bool:
    if not _API_TOKEN:
        return True
    return headers.get("authorization", "") == f"Bearer {_API_TOKEN}"


def _ws_authorized(websocket: WebSocket) -> bool:
    if _http_authorized(websocket.headers):
        return True
    # 浏览器/客户端不便设 header 时允许 ?token=; 仅在显式配置 token 时生效
    return bool(_API_TOKEN) and websocket.query_params.get("token") == _API_TOKEN


# 轻量内存滑动窗口限流器 (默认 120 req / 60s per client IP, 可通过 BT_STUDIO_RATE_LIMIT_PER_MIN 配置)
_RATE_LIMIT_PER_MIN = int(os.environ.get("BT_STUDIO_RATE_LIMIT_PER_MIN", "120") or 120)
_RATE_BUCKETS: Dict[str, Deque[float]] = defaultdict(deque)


def _check_rate_limit(client_ip: str) -> bool:
    if _RATE_LIMIT_PER_MIN <= 0:
        return True
    now = time.monotonic()
    q = _RATE_BUCKETS.get(client_ip)
    if q is not None:
        while q and now - q[0] > 60.0:
            q.popleft()
        if not q:
            _RATE_BUCKETS.pop(client_ip, None)
            q = None
    if len(_RATE_BUCKETS) > 4096:
        # 周期性全局清理过期空桶，防海量随机 IP 导致字典无限膨胀
        stale_ips = [ip for ip, dq in _RATE_BUCKETS.items() if not dq or now - dq[-1] > 60.0]
        for ip in stale_ips:
            _RATE_BUCKETS.pop(ip, None)
    if q is None:
        q = _RATE_BUCKETS[client_ip]
    if len(q) >= _RATE_LIMIT_PER_MIN:
        return False
    q.append(now)
    return True


def _extract_client_ip(request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        first_ip = xff.split(",")[0].strip()
        if first_ip:
            return first_ip
    return request.client.host if request.client else "unknown"


@app.middleware("http")
async def api_auth_middleware(request, call_next):
    path = request.url.path
    client_ip = _extract_client_ip(request)
    if path != "/api/health" and not _check_rate_limit(client_ip):
        return JSONResponse(status_code=429, content={"detail": "rate limit exceeded"})
    if (
        _API_TOKEN
        and (path.startswith("/api/") or path == "/metrics")
        and path != "/api/health"
        and not _http_authorized(request.headers)
    ):
        return JSONResponse(status_code=401, content={"detail": "unauthorized"})
    return await call_next(request)

# =============================================================================
# REST API 
# =============================================================================

@app.get("/")
async def root():
    return {
        "service": "bt_studio API Gateway",
        "version": "0.2.0",
        "status": "running",
        "mode": "results-gateway",
        "endpoints": {
            "health": "/api/health",
            "results": "/api/results",
            "results_by_kind": "/api/results/{kind}",
            "result_detail": "/api/results/{kind}/{name}",
            "ws": "/ws/{client_id}",
        },
        "monitored_kinds": [k.value for k in WATCH_SPECS],
    }


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "watcher_running": watcher._running,
        "watch_interval_sec": watcher._interval,
        "active_clients_count": sum(len(v) for v in manager.active_connections.values()),
        "registry_sizes": {k.value: len(v) for k, v in watcher._registry.items()},
    }


@app.get("/metrics")
def prometheus_metrics():
    """Prometheus 文本格式指标端点 (同步 def 由 FastAPI 线程池执行，不阻塞事件循环)。"""
    from fastapi.responses import PlainTextResponse
    from .monitor_api import get_index

    idx = get_index()
    idx.scan_if_stale(min_interval=2.0)
    ov = idx.overview()
    usage = ov.get("usage", {})
    all_time = ov.get("usage_all_time", {})
    alerts_list = ov.get("alerts", [])
    raw_pit = ov.get("raw_pit", {})

    lines = [
        "# HELP bt_studio_active_runs_24h Active LLM feature mining runs in the last 24h",
        "# TYPE bt_studio_active_runs_24h gauge",
        f"bt_studio_active_runs_24h {int(ov.get('active_runs_24h', 0))}",
        "# HELP bt_studio_viable_features_total Total viable features passing Stage 2",
        "# TYPE bt_studio_viable_features_total gauge",
        f"bt_studio_viable_features_total {int(ov.get('viable_features', 0))}",
        "# HELP bt_studio_llm_tokens_today Today LLM total tokens consumed",
        "# TYPE bt_studio_llm_tokens_today gauge",
        f"bt_studio_llm_tokens_today {int(usage.get('total_tokens', 0))}",
        "# HELP bt_studio_llm_calls_today Today LLM total API calls",
        "# TYPE bt_studio_llm_calls_today gauge",
        f"bt_studio_llm_calls_today {int(usage.get('calls', 0))}",
        "# HELP bt_studio_llm_tokens_all_time All-time LLM total tokens consumed",
        "# TYPE bt_studio_llm_tokens_all_time counter",
        f"bt_studio_llm_tokens_all_time {int(all_time.get('total_tokens', 0))}",
        "# HELP bt_studio_raw_pit_coverage Raw PIT minute bar cache coverage ratio",
        "# TYPE bt_studio_raw_pit_coverage gauge",
        f"bt_studio_raw_pit_coverage {float(raw_pit.get('coverage', 0.0)):.4f}",
        "# HELP bt_studio_alerts_active Active monitor alerts count",
        "# TYPE bt_studio_alerts_active gauge",
        f"bt_studio_alerts_active {len(alerts_list)}",
    ]
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")



# -----------------------------------------------------------------------------
# Artficat Registry
# -----------------------------------------------------------------------------

@app.get("/api/results")
async def list_result_kinds():
    return {
        "kinds": {
            k.value: {
                "directory": directory,
                "suffix": suffix,
                "count": len(watcher._registry[k]),
            }
            for k, (directory, suffix) in WATCH_SPECS.items()
        }
    }


@app.get("/api/results/{kind}")
async def list_results(kind: str):
    try:
        ArtifactKind(kind)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"未知产物类别: '{kind}'。可选类别: {[k.value for k in WATCH_SPECS]}",
        )
    return watcher.snapshot(kind)


@app.get("/api/results/{kind}/{name:path}")
async def get_result_detail(kind: str, name: str):
    detail = await watcher.get_detail(kind, name)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"产物不存在或尚未被索引: {kind}/{name}")
    return detail


# =============================================================================
# WebSocket E2E
# =============================================================================

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    if not _ws_authorized(websocket):
        await websocket.close(code=4401)
        return
    await manager.connect(websocket, client_id)

    try:
        await websocket.send_json({
            "event": "connected",
            "client_id": client_id,
            "message": "已成功连接至 bt_studio 结果监听网关",
        })

        await websocket.send_json({
            "event": "initial_state",
            "results": watcher.snapshot_all(),
        })

        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "ping":
                await websocket.send_json({"event": "pong"})

            elif action == "refresh":
                await websocket.send_json({
                    "event": "snapshot",
                    "results": watcher.snapshot_all(),
                })

            elif action == "subscribe":
                await websocket.send_json({
                    "event": "subscribed",
                    "kind": data.get("kind", "all"),
                })

    except WebSocketDisconnect:
        await manager.disconnect(websocket, client_id)
    except Exception as e:
        logger.warning(f"WebSocket 异常中断 [{client_id}]: {e}")
        await manager.disconnect(websocket, client_id)


# =============================================================================
# EntryPoint
# =============================================================================

def get_application() -> FastAPI:
    """ASGI (Gunicorn/Uvicorn Worker)"""
    return app


if __name__ == "__main__":
    is_dev = os.getenv("BT_STUDIO_ENV", "dev").lower() == "dev"
    bind_host = os.getenv("BT_STUDIO_BIND_HOST") or os.getenv("HOST") or "127.0.0.1"
    uvicorn.run(
        "bt_studio.plugins.api_server.main:app",
        host=bind_host,
        port=int(os.getenv("PORT", 8800)),
        reload=is_dev and os.getenv("UVICORN_RELOAD", "0") == "1",
        log_level="info",
    )
