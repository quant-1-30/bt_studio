#! /usr/bin/env python3
# -*- encondig: utf-8 -*-

import os
import numpy as np
from typing import List, Any, Dict
from bt_sdk.core.client import GetMdApi
from bt_core.execution.actor import AsyncRunner


def initialize_mdapi(timeout=1000):
    md_addr = os.getenv("MD_ADDR", "127.0.0.1:50051").split(":")
    mdapi = GetMdApi(addr=(md_addr[0], int(md_addr[1])), timeout=timeout)
    _runner = AsyncRunner()
    _runner.start() # new_event_loop
    _loop = _runner.get_loop()
    mdapi.start(_loop)
    return mdapi


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
