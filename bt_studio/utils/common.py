#! /usr/bin/env python3
# -*- encondig: utf-8 -*-

import os
import queue
import numpy as np
import polars as pl
import reactivex.operators as ops
from typing import List, Any, Dict
from bt_sdk.utils.util import _merge2DataFrame


def robust_z_normalize(window_data):
    """ mad z-norm replace (x - mean)/std """
    median_val = np.median(window_data)
    abs_dev = np.abs(window_data - median_val)
    
    mad = np.median(abs_dev)
    if mad == 0:
        mad = 1e-8
        
    robust_std = 1.4826 * mad
    robust_z = (window_data - median_val) / robust_std
    return robust_z


def calculate_dtw_params(config: dict):
    # L2 allowed_err(0.25 z-score) * sqrt(m)
    # max_dtw_dist = 0.5 * math.sqrt(m) 
    raw_window = int(config["m"] * config["dtw_window_frac"])
    # retricted and halfday
    max_intraday = int(60*2 / config["downsample"]) 
    dtw_window = max(1, min(max_intraday, raw_window))
    return dtw_window


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


def calculate_decay_weights(
    rets_window: dict[str, int], 
    half_life_minutes: float = 15.0
) -> dict[str, float]:
    """
    T+1 open Offset Targets exp weight

    Parameters
    ----------
    rets_window : dict[str, int]
        {"open_5m": 5, "open_15m": 15, "open_30m": 30}
    half_life_minutes : float, optional
        default: 30

    Returns
    -------
    dict[str, float]
    """
    if not rets_window:
        return {}

    decay_const = np.log(2) / half_life_minutes

    names = list(rets_window.keys())
    minutes = np.array(list(rets_window.values()), dtype=np.float64)

    raw_weights = np.exp(-decay_const * minutes)

    sum_w = np.sum(raw_weights)
    norm_weights = raw_weights / sum_w

    return {name: float(w) for name, w in zip(names, norm_weights)}
