
#! /usr/bin/env python3
# -*- encondig: utf-8 -*-

import os
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

from bt_studio.utils.common import initialize_mdapi, _collect_stream_sync
from bt_sdk.core.protocol import QueryBody
from bt_sdk.core.client.api import RpcTopic, FactorTopic
from bt_sdk.core.factor import apply_factor
from bt_core import external_mdapi_context


def downsample_universe(adj_close_dict: dict, n_layers=5) -> list:
    """
    a. top 80% based on avg_amount
    b. n_layer
    c. 1/5 each layer
    ---> 0.16
    """
    stats = []
    for sid, df in adj_close_dict.items():
        if df.height == 0:
            continue

        if "amount" in df.columns:
            mean_to = df["amount"].mean()
        elif "volume" in df.columns and "close" in df.columns:
            mean_to = (df["volume"] * df["close"]).mean()
        else:
            mean_to = 0.0

        stats.append({
            "sid": sid, 
            "mean_turnover": mean_to if mean_to is not None else 0.0
        })

    if not stats:
        return []

    sid_df = pl.DataFrame(stats).sort("mean_turnover", descending=True)

    chunk_size = int(sid_df.height * 0.8)
    if chunk_size == 0:
        return []

    chunk_df = sid_df.head(chunk_size)

    # revise missing data 
    segment_bounds = [int(i * chunk_size / n_layers) for i in range(n_layers+1)]
    
    samples = []
    for i in range(5):
        start_idx = segment_bounds[i]
        end_idx = segment_bounds[i+1]
        
        segment = chunk_df[start_idx:end_idx]

        if segment.height > 0:
            sample_size = max(1, int(segment.height * 0.2))
            
            sampled_sids = segment["sid"].sample(n=sample_size, seed=42).to_list()
            samples.extend(sampled_sids)
    return samples


def compress_snapshot(df: pl.DataFrame, hour=14, minute=55) -> pl.DataFrame:
    """
        Tick compress 14:55
    """
    if df.height == 0 or "tick" not in df.columns:
        return df
        
    return (
        df.lazy()
        .with_columns(datetime = pl.from_epoch(pl.col("tick"), time_unit="s"))
        .filter(pl.col("datetime").dt.hour() * 60 + pl.col("datetime").dt.minute() <= hour * 60 + minute)
        .with_columns(day = pl.col("datetime").dt.strftime("%Y%m%d").cast(pl.Int32))
        .group_by("day")
        .agg(pl.all().sort_by("tick").last())
        .drop("datetime")
        .sort("day")
        .collect()
    )


def compute_rolling_macro_states(bench_df: pl.DataFrame, loopback: int):
    """
        base on compress_snapshot
    """
    min_periods = int(loopback / 2)
    
    df = (
        bench_df.lazy()
        .sort("day")
        .with_columns(daily_ret = pl.col("close").pct_change().fill_null(0.0))
        .with_columns(
            p20 = pl.col("daily_ret").rolling_quantile(quantile=0.2, window_size=loopback, min_samples=min_periods),
            p80 = pl.col("daily_ret").rolling_quantile(quantile=0.8, window_size=loopback, min_samples=min_periods)
        )
        .drop_nulls(subset=["p20", "p80"])
        .with_columns(
            macro_state = pl.when(pl.col("daily_ret") < pl.col("p20")).then(0)  
                          .when(pl.col("daily_ret") > pl.col("p80")).then(2)  
                          .otherwise(1)                                            
        )
        .select(["day", "macro_state"])
        .collect()
    )
    return dict(zip(df["day"].to_list(), df["macro_state"].to_list()))


def prepare_macro(start_date: int, end_date: int, benchmark: str, stats_window: List[int], loopback: int, chunk_size=300):
    
    with external_mdapi_context() as mdapi:
        # =======================================================
        # 1. Universe and PIT 
        # =======================================================
        inst_df = mdapi.get_instrument()
        valid_meta = inst_df.filter(pl.col("delist") > start_date)
        universe = valid_meta["sid"].cast(pl.Binary).to_list() 

        first_trade_lazy = valid_meta.select([
            pl.col("sid").cast(pl.Binary), 
            pl.col("first_trading").cast(pl.Int32)
        ]).lazy()

        # =======================================================
        # 2. Benchmark 14:55 
        # =======================================================
        warmup_start = start_date - 10000 
        bench_bytes = benchmark.encode("utf-8") if isinstance(benchmark, str) else benchmark
        bench_body = QueryBody(start_date=warmup_start, end_date=end_date, sid=[bench_bytes]) 
        
        # Tick 
        bench_obs = mdapi.subscribe(bench_body, RpcTopic.Tick) 
        bench_raw = _collect_stream_sync(bench_obs)
        
        # compress 14:55 
        bench_df = compress_snapshot(bench_raw[bench_bytes])
        macro_dict = compute_rolling_macro_states(bench_df, loopback=loopback)
        
        # =======================================================
        # 3. DailyData Tick Stream Chunk and Compress
        # =======================================================
        adj_close_all = {}
        
        for i in range(0, len(universe), chunk_size):
            sub_uni = universe[i:i + chunk_size]
            body = QueryBody(start_date=warmup_start, end_date=end_date, sid=sub_uni)

            tick_obs = mdapi.subscribe(body, RpcTopic.Tick)
            raw_tick_dict = _collect_stream_sync(tick_obs)

            # compress to 14:55 and reduce 99%  
            snapshot_dict = {}
            for sid_bytes, tick_df in raw_tick_dict.items():
                snapshot_dict[sid_bytes] = compress_snapshot(tick_df)

            adj_factors = mdapi.get_factor(body, FactorTopic.Qfq)
            
            adj_close_chunk = apply_factor(snapshot_dict, adj_factors, FactorTopic.Qfq)
            
            adj_close_all.update(adj_close_chunk)

    if not adj_close_all:
        return universe, macro_dict, pl.DataFrame()
    
    # =======================================================
    # 4. Pre-Sampling
    # =======================================================
    samples = downsample_universe(adj_close_all) 
    
    lazy_frames = []
    for sid in samples:
        if sid in adj_close_all:
            df_lazy = (
                adj_close_all[sid]
                .lazy()
                .with_columns(pl.lit(sid).alias("sid"))
            )
            lazy_frames.append(df_lazy)
            
    if not lazy_frames:
        return samples, macro_dict, pl.DataFrame()
        
    lf = pl.concat(lazy_frames)
    # =======================================================
    # 5. Polars 
    # =======================================================
    lf = lf.join(first_trade_lazy, on="sid", how="left")

    lf = (
        lf.sort(["sid", "day"])
        .with_columns(day = pl.col("day").cast(pl.Int32))  
        
        # PIT --- Dynamic Filter 120天
        .with_columns(  
            date_day = pl.col("day").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d"),
            date_list = pl.col("first_trading").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d")
        )
        .filter((pl.col("date_day") - pl.col("date_list")).dt.total_days() >= 120) 
        .drop(["date_day", "date_list", "first_trading"]) 
        
        # 14:55 ret
        .with_columns(
            daily_ret = pl.col("close").log().diff().fill_null(0.0).over("sid")
        )
        .with_columns(
            dret_std = pl.col("daily_ret").rolling_std(window_size=20).forward_fill().over("sid")
        )
        .drop_nulls(subset=["dret_std"]) 
        .filter(pl.col("dret_std") >= 1e-4) # std < 1e-4  stands reach limit or no liquity | noise in fsm 
        .with_columns(
            daily_ret_z = pl.col("daily_ret") / pl.col("dret_std")
        )
    )

    # =======================================================
    # 6. Forward Returns 
    # =======================================================
    exprs = [
        pl.col("daily_ret").shift(-1).over("sid").alias("fwd_ret_1"),
        pl.col("daily_ret_z").shift(-1).over("sid").alias("fwd_ret_1_z")
    ]
    
    for fw in stats_window:
        if fw == 1: continue 
        exprs.append(
            pl.col("daily_ret")
            .rolling_sum(window_size=fw)
            .shift(-fw)                   
            .over("sid")
            .alias(f"fwd_ret_{fw}")
        )
        
    panel_df = lf.with_columns(exprs).collect()
    return samples, macro_dict, panel_df


def prepare_chunks(universe: list, start_date: int, end_date: int, adj:int=1):
    print(" loading minute data ...")
    with external_mdapi_context() as mdapi: 
        print(f"📦 [Head Node 预加载] 正在拉取 {start_date}-{end_date}...")

        # tick 
        body = QueryBody(start_date=start_date, end_date=end_date, sid=universe)
        tick_obs = mdapi.subscribe(body, RpcTopic.Tick)
        raw_tick = _collect_stream_sync(tick_obs)
        # factor
        adj_factors = mdapi.get_factor(body, FactorTopic.Qfq)
        # apply
        adj_tick = apply_factor(raw_tick, adj_factors, FactorTopic.Qfq)

        if not adj_tick: 
            return pl.DataFrame()

        # ========================================================
        # Memory Destructive Iteration)
        # ========================================================
        dfs =[]
        while adj_tick:
            sid_bytes, df = adj_tick.popitem() 
            if df.height > 0:
                dfs.append(df)

        # ========================================================
        # Rust rechunk=True continus memory
        # ========================================================
        tick_df = pl.concat(dfs, how="vertical", rechunk=True)
    return tick_df


def process_to_residuals(panel_df: pl.DataFrame, signal_type: str) -> dict:
    """
        LazyFrame + Graph
    """
    lf = panel_df.lazy()

    lf = lf.with_columns(
        datetime = pl.from_epoch(pl.col("tick"), time_unit="s")
    ).with_columns(
        day = pl.col("datetime").dt.strftime("%Y%m%d").cast(pl.Int32),
        # ensure to downsample minute and used for aggregate
        minute_bucket = pl.col("datetime").dt.truncate("1m")
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

    # median to extract markter beta 
    lf = lf.sort(["sid", "tick"]).with_columns(
        log_ret_raw = pl.col("signal_price").log().diff().fill_null(0.0).over("sid")
    ).with_columns(
        # median_ret = pl.col("log_ret_raw").median().over("tick")
        median_ret = pl.col("log_ret_raw").median().over("minute_bucket")
    ).with_columns(
        residual_ret = pl.col("log_ret_raw") - pl.col("median_ret")
    )

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
    ).drop(["signal_price", "median_ret", "minute_bucket"])

    # ==========================================
    # trigger
    # ==========================================
    df = lf.collect()

    hf_dfs = {}
    for sid_tuple, sub_df in df.partition_by("sid", as_dict=True).items():
        sid_val = sid_tuple[0] if isinstance(sid_tuple, tuple) else sid_tuple
        hf_dfs[sid_val] = sub_df.drop("sid")
        
    return hf_dfs


def calculate_gpd(returns_series: np.ndarray, quantiles: list): 
    """
    :param returns_series: np.array daily_returns
    :param quantiles: np.array  bins
    """
    returns = np.array(returns_series)
    returns = returns[np.isfinite(returns)]
    
    if len(returns) < 50: 
        return None, None
        
    centers = np.zeros(len(quantiles) + 1)
   # ==========================================================
    # z-score replace gpd when not enough data
    # Fallback avoid TypeError
    # ==========================================================
    if len(returns) < 15:
        theoretical = np.random.randn(10000)
        edges = np.quantile(theoretical, quantiles)
        for i in range(1, len(edges)):
            mask = (theoretical >= edges[i-1]) & (theoretical < edges[i])
            centers[i] = np.mean(theoretical[mask])

        centers[0] = np.mean(theoretical[theoretical < edges[0]])
        centers[-1] = np.mean(theoretical[theoretical > edges[-1]])
        return edges, centers

    # gpd calculation
    edges = np.quantile(returns, quantiles) 
    
    u_down = edges[0] 
    u_up = edges[-1]  
    
    for i in range(1, len(edges)):
        mask = (returns >= edges[i-1]) & (returns < edges[i])
        if np.any(mask):
            centers[i] = np.mean(returns[mask])
        else:
            centers[i] = (edges[i-1] + edges[i]) / 2.0
            
    # ==========================================
    # Right Tail GPD E[X | X > u] = u + scale / (1 - c)
    # ==========================================
    right_tail = returns[returns > u_up] - u_up 
    if len(right_tail) > 10:
        c_right, loc_right, scale_right = genpareto.fit(right_tail, floc=0) 
        if c_right < 1:
            centers[-1] = u_up + (scale_right / (1 - c_right))
        else:
            centers[-1] = np.mean(returns[returns > u_up])
    else:
        centers[-1] = np.mean(returns[returns > u_up]) if len(right_tail) > 0 else u_up
        
    # ==========================================
    # Left Tail GPD E[X | X < u] = u - scale / (1 - c)
    # ==========================================
    left_tail = -returns[returns < u_down] - (-u_down) 
    if len(left_tail) > 10:
        c_left, loc_left, scale_left = genpareto.fit(left_tail, floc=0)
        if c_left < 1:
            centers[0] = - (-u_down + (scale_left / (1 - c_left))) 
        else:
            centers[0] = np.mean(returns[returns < u_down])
    else:
        centers[0] = np.mean(returns[returns < u_down]) if len(left_tail) > 0 else u_down
        
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
