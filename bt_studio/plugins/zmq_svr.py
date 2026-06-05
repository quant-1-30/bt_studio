#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import zlib
import warnings
import time
import zmq.asyncio
from collections import defaultdict, deque

from core.rpc.client import rpc_client
from bt_quote.constant import RateLimit, MAX_WORKERS, CONNECTION_TIMEOUT
from bt_quote.core.protocol import _DECODER


class ZmqServer:
    def __init__(self, host, port, backend_port, max_workers):
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.frontend_url = f"tcp://{self.host}:{self.port}"
        self.backend_url = f"tcp://{self.host}:{backend_port}"
        self.loop = asyncio.get_event_loop()
        self.context = zmq.asyncio.Context()
        self.frontend = None  # ROUTER
        self.backend = None   # DEALER
        self.worker_tasks = []
        self.shutdown_event = asyncio.Event()

        self.checksum = b"sentinel"
        self.shutdown_msg = b"shutdown"

        # Performance counters
        self.total_requests = 0
        self.dropped_requests = 0
        self.active_connections = defaultdict(lambda: {
            'last_seen': time.time(),
            'rate_limiter': deque(),
            'request_count': 0
        })
        self.cleanup_interval = 30
        self.connection_timeout = CONNECTION_TIMEOUT

    def _create_sockets(self):
        # Frontend for clients
        self.frontend = self.context.socket(zmq.ROUTER)
        self.frontend.set_hwm(1000)
        self.frontend.bind(self.frontend_url)

        # Backend for workers
        self.backend = self.context.socket(zmq.DEALER)
        self.backend.set_hwm(1000)
        self.backend.bind(self.backend_url)

    async def start(self): # 外部asyncio.run 会创建loop , 不能重复执行run_forever / run_until_complete 
        self._create_sockets()

        for i in range(self.max_workers):
            task = self.loop.create_task(self._worker(i))
            self.worker_tasks.append(task)
        
        cleanup_task = self.loop.create_task(self._cleanup_connections())
        self.worker_tasks.append(cleanup_task)
        
        # zmq.proxy 阻塞在单独的线程中启动代理
        proxy_task = self.loop.create_task(self._proxy_task())
        self.worker_tasks.append(proxy_task)

        print(f"PyZMQ server listening on {self.frontend_url} with {self.max_workers} workers")
        
        try:
            await self.shutdown_event.wait()
        except asyncio.CancelledError:
            print("服务器被取消，开始清理...")
            await self.stop()

    async def _proxy_task(self):
        """运行代理的异步任务"""
        try:
            print("Starting proxy...")
            # 使用 poller 来非阻塞地运行代理
            poller = zmq.asyncio.Poller()
            poller.register(self.frontend, zmq.POLLIN)
            poller.register(self.backend, zmq.POLLIN)
            
            while not self.shutdown_event.is_set():
                try:
                    events = await poller.poll(timeout=1000)  # 1秒超时
                    if events:
                        for socket, event in events:
                            if socket == self.frontend and event == zmq.POLLIN:
                                # 从前端接收消息并转发到后端
                                msg = await self.frontend.recv_multipart()
                                # print("msg from fronted :", msg)
                                await self.backend.send_multipart(msg)
                            elif socket == self.backend and event == zmq.POLLIN:
                                # 从后端接收消息并转发到前端
                                msg = await self.backend.recv_multipart()
                                await self.frontend.send_multipart(msg)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"Proxy polling error: {e}")
                    await asyncio.sleep(1)
        except Exception as e:
            print(f"Proxy task error: {e}")
    
    def _check_limit(self, addr, max_requests_per_minute=RateLimit):
        now = time.time()
        rate_limiter = self.active_connections[addr]['rate_limiter']
        
        while rate_limiter and rate_limiter[0] < now - 60:
            rate_limiter.popleft()
            
        if len(rate_limiter) >= max_requests_per_minute:
            self.dropped_requests += 1
            return False
        rate_limiter.append(now)
        return True

    async def _worker(self, worker_id):
        worker_socket = self.context.socket(zmq.DEALER)
        worker_socket.connect(self.backend_url)
        print(f"Worker {worker_id} started")

        while not self.shutdown_event.is_set():
            try:
                # The first frame is the identity of the original client
                identity, req_id, data = await worker_socket.recv_multipart()
                # print("receive ", identity, req_id, data)
                await self._handle_request(identity, req_id, data, worker_socket)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
        
        worker_socket.close()
        print(f"Worker {worker_id} stopped")

    async def _handle_request(self, identity: bytes, req_id: bytes, data: bytes, worker_socket):
        response_count = 0
        request_body = _DECODER.decode(data) # object body
        # print("unpack request_body :", identity, req_id, request_body)

        # if not self._check_limit(addr):
        #     return

        conn_info = self.active_connections[identity]
        conn_info['last_seen'] = time.time()
        conn_info['request_count'] += 1
        self.total_requests += 1

        response_iterator = rpc_client.on_request(request_body.topic, request_body.body)
        try:
            async for response in response_iterator: # response: pa.buffer
                # print("response :", response)
                await worker_socket.send_multipart([identity, req_id, response]) # compress on arrow MTU
                response_count += 1
                if response_count % 127 == 0:
                    await asyncio.sleep(0)
        except Exception as e:
            print(f"RPC call failed for {identity}: {e}")

        await worker_socket.send_multipart([identity, req_id, b'eof'])

    async def _cleanup_connections(self):
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.cleanup_interval)
                now = time.time()
                
                inactive_client = [
                    client_id for client_id, info in self.active_connections.items()
                    if now - info['last_seen'] > self.connection_timeout
                ]
                
                for client_id in inactive_client:
                    del self.active_connections[client_id]

                print(
                    f"Stats: Active connections: {len(self.active_connections)}, "
                    f"Processed: {self.total_requests}, "
                    f"Dropped: {self.dropped_requests}"
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Connection cleanup error: {e}")

    async def stop(self):
        print("Stopping server...")
        self.shutdown_event.set()

        for task in self.worker_tasks:
            if not task.done():
                task.cancel() # 同步 `asyncio` 中的任务取消是一个分为两步 请求**和**响应** 

        await asyncio.gather(*self.worker_tasks, return_exceptions=True)

        if self.frontend:
            self.frontend.close()
        if self.backend:
            self.backend.close()
        
        self.context.term()
