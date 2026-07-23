#! /usr/bin/env python3
# -*- encondig: utf-8 -*-

import os
import queue
import numpy as np
import polars as pl
import reactivex.operators as ops
from typing import List, Any, Dict
from bt_sdk.utils.util import _merge2DataFrame


def _collect_stream_sync(observable) -> Dict[bytes, pl.DataFrame]:
    q = queue.Queue()
    observable.pipe(
        # ops.sample(0.1),  # 100ms abandon reset 
        # ops.buffer_with_time_or_count(timespan=1.0, count=500), # up to 500 / 1 second to list
        # ops.throttle_first(0.05), # on receive / 50ms not receive
        # ops.publish_replay(1), # cache 1 record 
        # ops.ref_count()
        ops.map(lambda data: data["data"]),
        ops.share()
    ).subscribe(
        on_next=q.put,
        on_error=q.put,
        on_completed=lambda: q.put(StopIteration)
    )
    
    tables = []
    while True:
        msg = q.get()
        if msg is StopIteration:
            break
        if isinstance(msg, Exception):
            raise msg
        tables.append(msg)
    data_df = _merge2DataFrame(tables)
    return data_df

