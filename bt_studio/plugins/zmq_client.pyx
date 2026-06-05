#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# cython: language_level=3, boundscheck=False, wraparound=False

import os
import socket
import asyncio
import uvloop
import reactivex
import uuid
import zmq
import zmq.asyncio
import threading
import pyarrow as pa
import pyarrow.compute as pc
import reactivex.operators as ops
import grpc
from reactivex import of
from reactivex.subject import Subject
from reactivex.scheduler.eventloop import AsyncIOScheduler
from concurrent.futures import Future, ThreadPoolExecutor

from bt_sdk.core.protocol import _ENCODER, _RespDECODER
from bt_sdk.core.rpc.client cimport RpcClient


asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

cdef int LENGTH_BYTES = 4
cdef int REQ_ID_SIZE = 16


cdef inline object _deserialize_to_table(bytes arrow_bytes): # inline function embed to reduce overhead when hundrends
    if not arrow_bytes:
        return None   
    # pa.py_buffer to wrap Python bytes
    # open_stream zero_copy parse 
    # read_all() ---> Table (underlying data point ZMQ bytes 
    # pc zero_copy and vectorize
    table = pa.ipc.open_stream(pa.py_buffer(arrow_bytes)).read_all()
    for col_name in ["name"]:
        if col_name in table.column_names:
            col = table.column(col_name)
            table = table.set_column(
                table.column_names.index(col_name),
                col_name,
                pc.cast(col, pa.string())
            )
    return table

cpdef object scale(dict data):
    """
        Table / RecordBatch scale
    """
    cdef object table = data["data"]
    cdef list columns = []
    cdef dict scale_configs = {
        **{col: 1e-5 for col in ["open", "high", "low", "close"]}, 
        **{col: 1e-3 for col in ["volume", "amount", "bonus_share", "transfer", "bonus", "price", "ratio"]}
    }
    cdef list field_names = table.schema.names
    cdef object col_data

    if table is None: return None
    
    for name in field_names:
        col_data = table.column(name) 
        if name in scale_configs:
            factor = scale_configs[name]
            col_data = pc.round(pc.multiply(col_data, factor), ndigits=2) # vectorize on C++ better than divide
        
        columns.append(col_data)
    return pa.Table.from_arrays(columns, names=field_names) # pa.RecordBatch.from_arrays


cdef class AsyncClient:

    def __init__(self):
        
        self._running = True
        self.listen_task = None
        
    cdef void _finalize_task(self, object future):
        try:
            ret = future.result()  # result or exception
        except Exception as e:
            print(f"_finalize_task: {e}")

    cdef object wrap_protocol(self, bytes req_id, object msg):
        """implement on protocol"""
        pass    

    async def send_request(self, bytes req_id, dict message):
        pass
    
    cpdef object run(self, bytes req_id, object msg):
        if not self._running:
            raise RuntimeError("client is not running")
        
        obs_or_fut = self.wrap_protocol(req_id, msg)
        return obs_or_fut

    async def _async_shutdown(self):
        if self.listen_task and not self.listen_task.done():
            self.listen_task.cancel()
            try:
                await self.listen_task
            except asyncio.CancelledError:
                pass

    cpdef void close(self):
        if not self._running: return
        self._running = False
        
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.create_task(self._async_shutdown())
        except RuntimeError:
            print("Close RuntimeError")
            pass # No loop running, nothing to clean up

        print(f"[{self.__class__.__name__}] Shutdown complete.")


cdef class AsyncZmqClient(AsyncClient):
    """
    High-performance asynchronous ZMQ client using a DEALER socket.
    This client is designed to communicate with a ZMQ ROUTER server.
    It maintains the singleton pattern and supports concurrent requests.
    """

    def __init__(self, tuple addr, int timeout=5):
        super().__init__()
        self.addr = f"tcp://{addr[0]}:{addr[1]}"
        self.timeout = timeout
        self._req_subject = {} # self._global_bus = Subject() # global bus to filter req is heavy cpu operation when max concurrent 
        self._connected = False
        self.loop = None
        
    cpdef void attach_loop(self, loop, bint is_background=False):
        self.loop = loop
        self.is_background_loop = is_background
        print(f"[{self.__class__.__name__}] Attached Loop: {id(loop)} (Background: {is_background})")
        
    async def _ensure_connection(self):
            """
                lazy initialize to ensure create zmq in loop
            """
            if self._connected:
                return
            try:
                print(f"[ZMQ] Initializing on Loop: {id(self.loop)}")
                self.context = zmq.asyncio.Context()
                self.socket = self.context.socket(zmq.DEALER)

                client_id = f"client-{uuid.uuid4()}".encode('utf-8')
                self.socket.setsockopt(zmq.IDENTITY, client_id)
                self.socket.setsockopt(zmq.LINGER, 0) # No Nagle
                # # zmq.SNDBUF` / `zmq.RCVBUF` kernal TCP Window 
                # self.socket.setsockopt(zmq.SNDBUF, 2*1024*1024)
                # self.socket.setsockopt(zmq.RCVBUF, 2*1024*1024)
                self.socket.setsockopt(42, 1) # zmq.constants.TCP_NODELAY / socket.TCP_NODELAY 1
                # Set high-water mark to prevent excessive memory usage
                self.socket.set_hwm(1000)
                # # heartbeat
                # self.socket.setsockopt(zmq.HEARTBEAT_IVL, 5000)      
                # self.socket.setsockopt(zmq.HEARTBEAT_TIMEOUT, 15000) 
                # self.socket.setsockopt(zmq.HEARTBEAT_TTL, 15000) 
                self.socket.connect(self.addr)
                self.listen_task = self.loop.create_task(self._listen_loop())
                self._connected = True
            except Exception as e:
                print(f"[ZMQ Init Error] {e}")
                raise e

    async def _listen_loop(self):
            """
            Sends a request via the ZMQ DEALER socket and asynchronously yields responses
            """
            cdef bytes payload
            cdef bytes r_id
            cdef object req_subject
            
            print("[ZMQ] Listener loop started.")
            while self._running:
                try:
                    # multi_frame [request_id, payload_bytes]
                    frames = await self.socket.recv_multipart()
                    if len(frames) < 2:
                        continue
                    
                    r_id = frames[0]
                    payload = frames[1]
                    req_subject = self._req_subject[r_id]
                    
                    if payload == b"eof":
                        req_subject.on_completed() # high efficient avoid take_while --- lambda 
                        self._req_subject.pop(r_id, None)
                        continue

                    # Table Zero-copy
                    table = _deserialize_to_table(payload)
                    
                    if table is not None:
                        req_subject.on_next({
                            "id": r_id, 
                            "data": table,
                            # "meta": table.schema.metadata 
                        })
                except zmq.ZMQError as ze:
                    if ze.errno == zmq.ETERM: break
                except asyncio.CancelledError:
                    print("Zmq asyncio.CancelledError")
                    break
                except Exception as e:
                    break 

    cdef object wrap_protocol(self, bytes req_id, object msg):
        cdef object req_subject = Subject()

        if not self._running:
            raise RuntimeError("client is not running")

        self._req_subject[req_id] = req_subject # avoid to ops.filter(lambda) --- global bus

        observable = req_subject.pipe(
            # ops.sample(0.1),  # 100ms abandon reset 
            # ops.buffer_with_time_or_count(timespan=1.0, count=500), # up to 500 / 1 second to list
            # ops.throttle_first(0.05), # on receive / 50ms not receive
            # ops.publish_replay(1), # cache 1 record 
            # ops.ref_count()
            ops.map(scale),
            ops.share() 
        ) 
        if self.is_background_loop: # avoid RuntimeError
            coro = self.send_request(req_id, msg) 
            future = asyncio.run_coroutine_threadsafe(coro, self.loop)
            future.add_done_callback(lambda f: self._finalize_task(f))
        else:
            self.loop.create_task(self.send_request(req_id, msg)) 
        return observable

    async def send_request(self, bytes req_id, object msg):
        await self._ensure_connection()
        # serialize_msg = pack(msg)
        serialize_msg = _ENCODER.encode(msg)
        await self.socket.send_multipart([req_id, serialize_msg]) # multi_frame

    async def _async_shutdown(self):
        await AsyncClient._async_shutdown(self) # 如果非await直接self.

        if self.socket is not None:
            self.socket.close(linger=0) # drop immediately
            self.socket = None 

        if self.context is not None:
            self.context.term()
            self.context = None

