
#! /usr/bin/env python3
# -*- encondig: utf-8 -*-

import os
import ray
import math
import gc
import numpy as np
import stumpy
import pyarrow as pa
import pyarrow.compute as pc
import polars as pl
import contextlib
from collections import deque
from typing import List, Any, Dict
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from scipy.stats import chi2_contingency, ks_2samp, skew, genpareto

from workflow.common import *
from bt_sdk.core.protocol import QueryBody


def generate_quarter(start_date: int, end_date: int, overlap_days: int = 15):
    start_dt = datetime.strptime(str(start_date), "%Y%m%d")
    end_dt = datetime.strptime(str(end_date), "%Y%m%d")
    chunks =[]
    curr_start = start_dt
    
    while curr_start <= end_dt:
        curr_end = curr_start + relativedelta(months=3) - timedelta(days=1)
        if curr_end > end_dt: curr_end = end_dt
            
        req_start = curr_start - timedelta(days=overlap_days)
        chunks.append({
            "req_start": int(req_start.strftime("%Y%m%d")),
            "end_date": int(curr_end.strftime("%Y%m%d")),
            "valid_start": int(curr_start.strftime("%Y%m%d"))
        })
        curr_start = curr_end + timedelta(days=1)
    return chunks


def prepare_universe(start_date: int, end_date: int, market: str, frac: float = 0.1):
    md_api = initialize_mdapi()
    
    table = md_api.get_instrument()
    df = pl.from_arrow(table)  # PyArrow Table → Polars DataFrame
    
    mask = (
        (pl.col("first_trading") < end_date * 10000) &
        # (pl.col("delist") > start_date * 10000) &
        pl.col("sid").str.starts_with(market)
    )
    
    filter_df = df.filter(mask)
    num_rows = filter_df.height 
    
    num_samples = max(1, int(num_rows * frac))
    random_idx = np.random.choice(num_rows, size=num_samples, replace=False)
    sample_df = filter_df[random_idx]  
    
    samples = sample_df["sid"].cast(pl.Binary).to_list()
    universe = filter_df["sid"].cast(pl.Binary).to_list()
    return md_api, universe, samples


def prepare_daily(mdapi: object, universe: List[bytes], benchmark: bytes, start_date: int, end_date: int, stats_window: List[int], thres: float, loopback: int=252):
    warmup_start = start_date - 20000 
    
    # 1. macro_state
    bench_body = QueryBody(start_date=warmup_start, end_date=end_date, sid=[benchmark]) 
    bench_data = mdapi.get_benchmark(bench_body)
    macro_dict = compute_rolling_macro_states(bench_data[benchmark], loopback=loopback)

    # 2. universe close
    close_body = QueryBody(start_date=warmup_start, end_date=end_date, sid=universe)
    daily_dict = mdapi.get_close(close_body, 1)
    
    # LazyFrame and Concat
    lf = pl.concat([
        v.lazy() if isinstance(v, pl.DataFrame) else pl.from_arrow(v).lazy() 
        for _, v in daily_dict.items()
    ])

    lf = (
        lf.sort(["sid", "day"])
        .with_columns(
            day = pl.col("day").cast(pl.Int32), 
            daily_ret = pl.col("close").log().diff().fill_null(0.0).over("sid")
        )
        .with_columns(
            daily_vol = pl.col("daily_ret").rolling_std(window_size=20) # min_samples
                          .forward_fill() # .fill_null
                          .over("sid")
        ).drop_nulls(subset=["daily_vol"]) # avoid ipo between start_date and end_date
        .with_columns(
            daily_vol = pl.when(pl.col("daily_vol") < thres)
                          .then(thres)
                          .otherwise(pl.col("daily_vol"))
        )
        .with_columns(
            daily_ret_z = pl.col("daily_ret") / pl.col("daily_vol")
        )
    )
    
    # 4. Forward Returns
    exprs =[
        pl.col("daily_ret").shift(-1).over("sid").alias("fwd_ret_1"),
        pl.col("daily_ret_z").shift(-1).over("sid").alias("fwd_ret_1_z")
    ]
    
    for fw in stats_window:
        if fw == 1: 
            continue 
        expr = (
            pl.col("daily_ret")
            .rolling_sum(window_size=fw)
            .shift(-fw)                   
            .over("sid")
            .alias(f"fwd_ret_{fw}")
        )
        exprs.append(expr)
        
    lf = lf.with_columns(exprs)
    
    # 5. Polars C++ 
    panel_df = lf.collect()
    
    return panel_df, macro_dict


def prepare_ray_chunks(mdapi: object, universe: list, start_date: int, end_date: int, signal_type: str, adj=1):
    print(" loading minute data ...")
    chunks = generate_quarter(start_date, end_date, overlap_days=15)
    
    chunk_metas =[]  
    chunk_refs =[]   

    for idx, chunk in enumerate(chunks):
        print(f"📦 [Head Node 预加载] 正在拉取 {chunk['valid_start']}-{chunk['end_date']}...")
        
        body = QueryBody(start_date=chunk["req_start"], end_date=chunk["end_date"], sid=universe)
        tick_dict = mdapi.get_subscribe(body, adj)
        if not tick_dict: 
            continue

        # ========================================================
        # Memory Destructive Iteration)
        # ========================================================
        dfs =[]
        while tick_dict:
            sid_bytes, df = tick_dict.popitem() 
            if df.height > 0:
                dfs.append(df)
                
        del tick_dict 
        gc.collect()

        if not dfs:
            continue

        # ========================================================
        # Rust rechunk=True continus memory
        # ========================================================
        tick_df = pl.concat(dfs, how="vertical", rechunk=True)
        
        del dfs
        gc.collect()

        # ========================================================
        # put to Plasma (share memory)
        # ========================================================
        df_ref = ray.put(tick_df)
        
        del tick_df
        gc.collect()

        chunk_metas.append({
            "req_start": chunk["req_start"],
            "valid_start": chunk["valid_start"],
            "end_date": chunk["end_date"],
        })
        chunk_refs.append(df_ref) # objectRef ptr
        
        print(f"✅ Chunk {idx} Memory Transfer to Plasma and Head Node gc")
    return chunk_metas, chunk_refs


def process_to_residuals(panel_df: pl.DataFrame, signal_type: str) -> dict:
    """
        LazyFrame + Graph
    """
    lf = panel_df.lazy()

    lf = lf.with_columns(
        datetime = pl.from_epoch(pl.col("tick"), time_unit="s")
    ).with_columns(
        day = pl.col("datetime").dt.strftime("%Y%m%d").cast(pl.Int32)
    ).filter(
        pl.col("datetime").dt.hour() * 60 + pl.col("datetime").dt.minute() <= 14 * 60 + 55
    )

    if signal_type == "vwap":
        lf = lf.with_columns(
            signal_price = pl.when(pl.col("volume") > 0)
                             .then(pl.col("amount") / pl.col("volume"))
                             .otherwise(pl.col("close"))
        )
    else:
        lf = lf.with_columns(signal_price = pl.col("close"))

    # median 
    lf = lf.sort(["sid", "tick"]).with_columns(
        log_ret_raw = pl.col("signal_price").log().diff().fill_null(0.0).over("sid")
    ).with_columns(
        median_ret = pl.col("log_ret_raw").median().over("tick")
    ).with_columns(
        residual_ret = pl.col("log_ret_raw") - pl.col("median_ret")
    )

    # 
    if signal_type == "vpt":
        lf = lf.with_columns(
            daily_mean_vol = pl.col("volume").mean().over(["sid", "day"])
        ).with_columns(
            vol_weight = pl.when(pl.col("daily_mean_vol") > 0)
                           .then(pl.col("volume") / pl.col("daily_mean_vol"))
                           .otherwise(1.0)
        ).with_columns(
            residual_ret = pl.col("residual_ret") * pl.col("vol_weight")
        ).drop(["daily_mean_vol", "vol_weight"])

    
    lf = lf.with_columns(
        intraday_cum = pl.col("residual_ret").cum_sum().over("sid")
    ).drop(["signal_price", "median_ret"])

    # ==========================================
    # trigger
    # ==========================================
    df = lf.collect()

    hf_dfs = {}
    for sid_tuple, sub_df in df.partition_by("sid", as_dict=True).items():
        sid_val = sid_tuple[0] if isinstance(sid_tuple, tuple) else sid_tuple
        hf_dfs[sid_val] = sub_df.drop("sid")
        
    return hf_dfs


def compute_rolling_macro_states(bench_df: pl.DataFrame, loopback: int):
    """
        Lazy Chain
    """
    min_periods = int(loopback / 2)
    
    df = (
        bench_df.lazy()
        .with_columns(
            datetime = pl.from_epoch(pl.col("tick"), time_unit="s")
        )
        .filter(
            pl.col("datetime").dt.hour() * 60 + pl.col("datetime").dt.minute() <= 14 * 60 + 55
        )
        .with_columns(
            day = pl.col("datetime").dt.strftime("%Y%m%d").cast(pl.Int32)
        )
        .group_by("day").agg(
            close_1455 = pl.col("close").last() 
        )
        .sort("day")
        .with_columns(
            daily_ret_1455 = pl.col("close_1455").pct_change()
        )
        .with_columns(
            p20 = pl.col("daily_ret_1455").rolling_quantile(quantile=0.2, window_size=loopback, min_periods=min_periods),
            p80 = pl.col("daily_ret_1455").rolling_quantile(quantile=0.8, window_size=loopback, min_periods=min_periods)
        )
        .drop_nulls()
        .with_columns(
            macro_state = pl.when(pl.col("daily_ret_1455") < pl.col("p20")).then(0)  
                          .when(pl.col("daily_ret_1455") > pl.col("p80")).then(2)  
                          .otherwise(1)                                            
        )
        .select(["day", "macro_state"])
        .collect()
    )
    
    return dict(zip(df["day"].to_list(), df["macro_state"].to_list()))


def calculate_gpd(returns_series: np.array, quantiles: list): 
    """
    :param returns_series: np.array daily_returns
    :param quantiles: np.array  bins
    """
    returns = np.array(returns_series)
    returns = returns[np.isfinite(returns)]
    
    centers = np.zeros(len(quantiles) + 1)

    edges = np.quantile(returns, quantiles) # any np.nan return nan
    u_down = edges[0] 
    u_up = edges[-1]  
    
    # ==========================================
    # empritical
    # ==========================================
    centers[1] = np.mean(returns[(returns >= edges[0]) & (returns < edges[1])])
    centers[2] = np.mean(returns[(returns >= edges[1]) & (returns < edges[2])])
    centers[3] = np.mean(returns[(returns >= edges[2]) & (returns < edges[3])])
    
    # ==========================================
    # right GPD 
    # ==========================================
    right_tail = returns[returns > u_up] - u_up # loc = 0

    if len(right_tail) > 10:
        # scipy genpareto
        c_right, loc_right, scale_right = genpareto.fit(right_tail, floc=0) # evt and gpd
        
        # GPD E[X - u | X > u] = scale / (1 - c) 
        if c_right < 1:
            expected_excess_up = scale_right / (1 - c_right)
            centers[4] = u_up + expected_excess_up
        else:
            centers[4] = np.mean(returns[returns > u_up])
    else:
        centers[4] = np.mean(returns[returns > u_up])
        
    # ==========================================
    # left GPD
    # ==========================================
    left_tail = -returns[returns < u_down] - (-u_down) # min 
    if len(left_tail) > 10:
        c_left, loc_left, scale_left = genpareto.fit(left_tail, floc=0)
        if c_left < 1:
            expected_excess_down = scale_left / (1 - c_left)
            centers[0] = - (-u_down + expected_excess_down) 
        else:
            centers[0] = np.mean(returns[returns < u_down])
    else:
        centers[0] = np.mean(returns[returns < u_down])
    return edges, centers


def build_rolling_gpd(panel_df: pl.DataFrame, quantiles: list, loopback: int, freq_month: int):
    """
        LazyFrame  + deque $O(1)$ 
    """
    daily_rets_df = (
        panel_df.lazy()
        .select(["day", "daily_ret_z"])
        .drop_nulls()
        .group_by("day").agg(
            rets = pl.col("daily_ret_z") # list
        )
        .sort("day")
        .collect()
    )
    
    dates = daily_rets_df["day"].to_list()
    rets_list = daily_rets_df["rets"].to_list()
    
    gpd_dict = {}
    last_update_idx = -9999
    current_edges, current_centers = None, None
    
    # O(1)
    rolling_window = deque(maxlen=loopback)
    
    for d_int, day_rets in zip(dates, rets_list):
        rolling_window.append(day_rets)
        
        year = d_int // 10000
        month = (d_int % 10000) // 100
        curr_month_idx = year * 12 + month
        
        if current_edges is None or (curr_month_idx - last_update_idx >= freq_month):
            hist_rets = np.concatenate(rolling_window) if rolling_window else np.array([])
            # if len(hist_rets) >= 500:
            current_edges, current_centers = calculate_gpd(hist_rets, quantiles)
            last_update_idx = curr_month_idx
                    
        gpd_dict[d_int] = (current_edges, current_centers)
        
    return gpd_dict


def evaluate_objective(p_val: float, cond_mean: np.array, uncond_mean:np.array, m:int, penalty_m:int): 
    """
    1. **no direction**  abs(spread)
    2. **smooth** -log10(p_val)
    3. **length penalty**
    """
    spread = cond_mean - uncond_mean
    abs_spread = abs(spread)
    
    safe_pval = max(p_val, 1e-10)
    confidence = -np.log10(safe_pval) # penalty  log10(1.0) = 0 / log10(0.001) = -3 
    raw_score = confidence * abs_spread * 10000.0

    # Occam's Razor
    if m > penalty_m:
        penalty_factor = math.exp(- (m - penalty_m) / 50.0) 
        raw_score = raw_score * penalty_factor
    return raw_score
