from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, AsyncIterator, Optional

import zmq
import zmq.asyncio

try:
    from codec import EOF, decode_payload, encode_request
except ImportError:
    from .codec import EOF, decode_payload, encode_request

log = logging.getLogger(__name__)


class ZmqClient:
    """High-HWM ZeroMQ DEALER client with lossless background quote streaming.

    Transport-generic: it only speaks the ``codec`` request/response protocol and
    routes ``(rid, payload)`` frames to per-request queues. The concrete backend
    (XTP today, others in the future) lives in ``bt_playgroud``'s server side; the
    RPC method names (``trader.*`` / ``quote.*``) are opaque strings to this client.
    """

    def __init__(self, endpoint: str, timeout: float = 10.0, enable_ipv6: bool = False, rcvhwm: int = 65536):
        self.endpoint = endpoint
        self.timeout = timeout
        self.enable_ipv6 = enable_ipv6
        self.rcvhwm = max(4096, int(rcvhwm))
        self._ctx: Optional[zmq.asyncio.Context] = None
        self._socket: Optional[zmq.asyncio.Socket] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._queues: dict[bytes, asyncio.Queue] = {}
        self._connected = False

    async def connect(self) -> None:
        if self._connected:
            return
        self._ctx = zmq.asyncio.Context()
        self._socket = self._ctx.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.IDENTITY, f"bt-{uuid.uuid4()}".encode())
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.IPV6, 1 if self.enable_ipv6 else 0)
        self._socket.setsockopt(zmq.RCVHWM, self.rcvhwm)
        self._socket.setsockopt(zmq.SNDHWM, self.rcvhwm)
        self._socket.setsockopt(zmq.RCVBUF, 16 * 1024 * 1024)
        self._socket.connect(self.endpoint)
        self._recv_task = asyncio.create_task(self._recv_loop())
        self._connected = True

    async def close(self) -> None:
        if not self._connected:
            return
        self._connected = False
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
            self._recv_task = None
        if self._socket:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
            self._socket = None
        if self._ctx:
            try:
                self._ctx.term()
            except Exception:
                pass
            self._ctx = None

    async def _recv_loop(self) -> None:
        assert self._socket is not None
        while self._connected:
            try:
                frames = await self._socket.recv_multipart()
                if len(frames) < 2:
                    continue
                rid, payload = frames[0], frames[1]
                q = self._queues.get(rid)
                if q is None:
                    continue
                if payload == EOF:
                    q.put_nowait(None)
                    self._queues.pop(rid, None)
                else:
                    q.put_nowait(decode_payload(payload))
            except asyncio.CancelledError:
                break
            except zmq.ZMQError as ze:
                if ze.errno == zmq.ETERM:
                    break
            except Exception:
                continue

    async def _send(self, method: str, params: dict[str, Any]) -> bytes:
        assert self._socket is not None
        rid = uuid.uuid4().bytes
        self._queues[rid] = asyncio.Queue()
        await self._socket.send_multipart([rid, encode_request(method, params)])
        return rid

    async def call(self, method: str, params: Optional[dict[str, Any]] = None) -> Any:
        rid = await self._send(method, params or {})
        q = self._queues[rid]
        try:
            last = None
            while True:
                f = await asyncio.wait_for(q.get(), timeout=self.timeout)
                if f is None:
                    break
                last = f
            return last
        finally:
            self._queues.pop(rid, None)

    async def subscribe(
        self, method: str, params: Optional[dict[str, Any]] = None, idle_timeout: Optional[float] = None
    ) -> AsyncIterator[Any]:
        rid = await self._send(method, params or {})
        q = self._queues[rid]
        wait_s = (self.timeout * 6.0) if idle_timeout is None else idle_timeout
        try:
            while True:
                f = (await q.get()) if wait_s <= 0 else (await asyncio.wait_for(q.get(), timeout=wait_s))
                if f is None:
                    break
                yield f
        except asyncio.TimeoutError:
            log.warning("subscribe timeout for method=%s rid=%s", method, rid.hex()[:8])
            return
        finally:
            self._queues.pop(rid, None)

    async def ping(self) -> Any:
        return await self.call("ping", {})
