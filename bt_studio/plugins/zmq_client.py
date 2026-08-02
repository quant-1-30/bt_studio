from __future__ import annotations

import asyncio
import uuid
import zmq
import zmq.asyncio

from typing import Any, AsyncIterator, Optional

try:
    from codec import EOF, decode_payload, encode_request
except ImportError:
    from .codec import EOF, decode_payload, encode_request


class XtpClient:
    def __init__(self, endpoint: str, timeout: float = 10.0):
        self.endpoint = endpoint; self.timeout = timeout
        self._ctx = None; self._socket = None; self._recv_task = None
        self._queues = {}; self._connected = False

    async def connect(self):
        if self._connected: return
        self._ctx = zmq.asyncio.Context()
        self._socket = self._ctx.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.IDENTITY, f"bt-{uuid.uuid4()}".encode())
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(42, 1)
        self._socket.set_hwm(1000)
        self._socket.connect(self.endpoint)
        self._recv_task = asyncio.create_task(self._recv_loop())
        self._connected = True

    async def close(self):
        if not self._connected: return
        self._connected = False
        if self._recv_task:
            self._recv_task.cancel()
            try: await self._recv_task
            except asyncio.CancelledError: pass
        if self._socket: self._socket.close(linger=0)
        if self._ctx: self._ctx.term()

    async def _recv_loop(self):
        while self._connected:
            try:
                frames = await self._socket.recv_multipart()
                if len(frames) < 2: continue
                rid, payload = frames[0], frames[1]
                q = self._queues.get(rid)
                if q is None: continue
                if payload == EOF:
                    await q.put(None); self._queues.pop(rid, None)
                else:
                    await q.put(decode_payload(payload))
            except asyncio.CancelledError: break
            except zmq.ZMQError as ze:
                if ze.errno == zmq.ETERM: break
            except: continue

    async def _send(self, method, params):
        rid = uuid.uuid4().bytes
        q = asyncio.Queue(); self._queues[rid] = q
        await self._socket.send_multipart([rid, encode_request(method, params)])
        return rid

    async def call(self, method, params=None):
        params = params or {}
        rid = await self._send(method, params)
        q = self._queues[rid]
        try:
            last = None
            while True:
                f = await asyncio.wait_for(q.get(), timeout=self.timeout)
                if f is None: break
                last = f
            return last
        finally:
            self._queues.pop(rid, None)

    async def subscribe(self, method, params=None):
        params = params or {}
        rid = await self._send(method, params)
        q = self._queues[rid]
        try:
            while True:
                f = await asyncio.wait_for(q.get(), timeout=self.timeout * 6)
                if f is None: break
                yield f
        except asyncio.TimeoutError: return
        finally:
            self._queues.pop(rid, None)

    async def ping(self): 
        return await self.call("ping", {})
