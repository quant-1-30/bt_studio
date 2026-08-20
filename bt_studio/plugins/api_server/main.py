#! /usr/bin/env python3

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Set

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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

# CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.get("/api/results/{kind}/{name}")
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
    uvicorn.run(
        "bt_studio.plugins.api_server.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
        log_level="info",
    )
