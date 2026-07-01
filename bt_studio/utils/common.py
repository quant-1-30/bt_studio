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


def intercept(config):
    # ====================================================
    # Domain Knowledge Guardrails
    # ====================================================
    # 拦截 1 形态点数过少非有效博弈或过多无法匹配
    m = int(config["ndays"] * np.floor(240 / config["downsample"]))
    if m < 8 or m > 60:
        return {"status": "failed", "reason": f"m={m} 长度不合理", "metrics_score": -9999}
        
    # # 拦截 2 频率倒挂 采样频率比Beta频率还高噪音
    # if config["downsample"] < config["rolling_freq"]:
    #     return {"status": "failed", "reason": "频率倒挂", "metrics_score": -9999}
        
    # # 拦截 3 相关系数与长度木桶效应
    # if m > 30 and config["threshold_r"] > 0.90:
    #     return {"status": "failed", "reason": "长序列要求高r", "metrics_score": -9999}
    return {"status": "success"}


def get_latest_ckpt(target_year: int, model_dir: str) -> str:
    if not os.path.exists(model_dir): return None
    valid_models = []
    for f in os.listdir(model_dir):
        if f.startswith("model_") and f.endswith(".pkl"):
            try:
                y = int(f.replace("model_", "").replace(".pkl", ""))
                if y <= target_year:
                    valid_models.append((y, os.path.join(model_dir, f)))
            except ValueError:
                continue
                
    if not valid_models: return None
    valid_models.sort(key=lambda x: x[0], reverse=True)
    return valid_models[0][1]


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
    # import pdb; pdb.set_trace() 
    data_df = _merge2DataFrame(tables)
    return data_df
