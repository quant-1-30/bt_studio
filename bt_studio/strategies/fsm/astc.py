#! /usr/bin/env python3
# -*- encondig: utf-8 -*-

import ray
import stumpy
import numpy as np
import polars as pl
import scipy.stats as stats
from typing import List, Any, Dict
from dtaidistance import dtw

from workflow.preprocess import *


def extract_asset_feature(hf_df: pl.DataFrame, downsample: int, m: int, amplify: int = 1000):
    """
        1. strict with 14:55 to eliminate future
        2. revise daily_ret from T-1 14:55 to T 14:55
        3. filter suspend and price limit
    """
    
    # ==========================================
    # 1. filter 14:55:00 
    # ==========================================
    lf_1455 = hf_df.lazy().filter(
        pl.col("datetime").dt.hour() * 60 + pl.col("datetime").dt.minute() <= 14 * 60 + 55
    ).with_columns(
        day = pl.col("datetime").dt.strftime("%Y%m%d").cast(pl.Int32)
    )
    
    # ==========================================
    # 2. daily ret 14:55 
    # ==========================================
    lf_daily = (
        lf_1455.group_by("day").agg(
            close_1455 = pl.col("close").last() 
        ).sort("day").with_columns(
            daily_ret = pl.col("close_1455").log().diff().fill_null(0.0) 
        )
    )
    
    # ==========================================
    # 3. downsample
    # ==========================================
    lf_sampled = (
        lf_1455.with_columns(
            intraday_cum_bps = pl.col("intraday_cum") * amplify
        ).group_by_dynamic(
            "datetime", 
            every=f"{downsample}m", 
            closed="right",
            label="right"
        ).agg(
            cum_val = pl.col("intraday_cum_bps").last(),
            day = pl.col("day").last(),
            last_tick_time = pl.col("datetime").last(),
            last_price = pl.col("close").last() 
        ).drop_nulls(subset=["cum_val"])
    )
    
    # ==========================================
    # 🚀 graph and calculate
    # =======================================================
    df_daily, df_sampled = pl.collect_all([lf_daily, lf_sampled])
    
    ret_dict = dict(zip(df_daily["day"].to_list(), df_daily["daily_ret"].to_list()))
    
    cum_vals = df_sampled["cum_val"].to_numpy()
    dates = df_sampled["day"].to_numpy()
    prices = df_sampled["last_price"].to_numpy() 
    tick_times = df_sampled["last_tick_time"].to_list()
    
    if len(cum_vals) < m:
        return[]
        
    is_eod = np.concatenate([dates[:-1] != dates[1:], [True]])
    eod_indices = np.where(is_eod)[0]
    
    records =[]
    for idx in eod_indices:
        if idx - m + 1 < 0:
            continue
            
        # filter suspend and reach limit eod eg 14:35, 14:45, 14:55  
        t_time = tick_times[idx]
        if t_time.hour < 14 or (t_time.hour == 14 and t_time.minute < 45):
            continue
            
        tail_prices = prices[idx - 2 : idx + 1] if idx >= 2 else prices[:idx+1]
        if np.std(tail_prices) < 1e-5: 
            continue
            
        curve_m = cum_vals[idx - m + 1 : idx + 1]
        curr_date = dates[idx]
        
        records.append({
            "day": curr_date,
            "curve": curve_m,
            "daily_ret": ret_dict.get(curr_date, 0.0),
        })
        
    return records


def build_stumpy_from_chunk(chunks_ref: List[ray.ObjectRef], samples: list, config: dict, signal_type: str):
    m = config["m"]
    padded =[]
    samples_str = [s.decode("utf-8") for s in samples]
    nan_buffer = np.full(m, np.nan)
    
    for df_ref in chunks_ref:
        df = ray.get(df_ref)
        df_samples = df.filter(pl.col("sid").is_in(samples_str))
        
        if df_samples.height == 0:
            continue
            
        hf_dfs_dict = process_to_residuals(df_samples, signal_type)
        
        for sid, hf_df in hf_dfs_dict.items():
            df_sampled = hf_df.filter(
                pl.col("datetime").dt.hour() * 60 + pl.col("datetime").dt.minute() <= 14 * 60 + 55
            ).with_columns(
                intraday_cum_bps = pl.col("intraday_cum") * 1000.0
            ).group_by_dynamic(
                "datetime", every=f"{config['downsample']}m", closed="right", label="right"
            ).agg(
                cum_val = pl.col("intraday_cum_bps").last()
            ).drop_nulls(subset=["cum_val"])
            
            valid_series = df_sampled["cum_val"].to_numpy()
            if len(valid_series) > m:
                padded.append(valid_series)
                padded.append(nan_buffer)
                
    return np.concatenate(padded) if padded else np.array([])


def build_panel_from_chunk(chunks_meta: list, chunks_ref: list, daily_ret_df: pl.DataFrame, config: dict, signal_type: str):
    m = config["m"]
    df_list =[] 
    
    for meta, chunk_data in zip(chunks_meta, chunks_ref):
        # ========================================================
        # Ray auto decrf ObjectRef
        # ========================================================
        if isinstance(chunk_data, ray.ObjectRef):
            chunk_df = ray.get(chunk_data)
        else:
            chunk_df = chunk_data
            
        if chunk_df.height == 0:
            continue

        # ========================================================
        # 🛡️ reuse median logic
        # ========================================================
        hf_dfs_dict = process_to_residuals(chunk_df, signal_type)
        # hf_dfs_dict = chunk_df.partition_by("sid", as_dict=True)
        
        chunk_records =[]
        
        # ========================================================
        # 🛡️ extract feature
        # ========================================================
        for sid, hf_df in hf_dfs_dict.items():
            records = extract_asset_feature(hf_df, config["downsample"], m, amplify=1000)
            
            for r in records:
                # ========================================================
                # 🌟 Burn-in Cut-off
                # ========================================================
                if meta["valid_start"] <= r["day"] <= meta["end_date"]:
                    r["sid"] = sid
                    chunk_records.append(r)
        
        del chunk_df
        del hf_dfs_dict
        
        if chunk_records:
            df_list.append(pl.DataFrame(chunk_records)) 
            
    if not df_list:
        return pl.DataFrame()
        
    # ========================================================
    # 🛡️ Concat Chunk and Daily ret
    # ========================================================
    snapshot_panel = pl.concat(df_list).sort(["sid", "day"])
    
    # panel_df = snapshot_panel.join(
    #     daily_ret_df, 
    #     on=["sid", "day"],
    #     how="inner" 
    # )

    panel_df = snapshot_panel.join(
        daily_ret_df, 
        on=["sid", "day"],
        how="left" 
    ).sort(["sid", "day"]).with_columns(
        daily_vol = pl.col("daily_vol").forward_fill().over("sid"), # fill_null
        daily_ret = pl.col("daily_ret").forward_fill().over("sid"),
    ).drop_nulls(subset=["curve", "daily_ret", "daily_vol"])  
    return panel_df


def get_atsc(raw_array: np.array, config):
    # filter by person d^2 = 2m(1-r)
    m = config["m"]
    threshold_d = config["threshold_d"]
    mp = stumpy.stump(raw_array, m=m)

    distances = np.copy(mp[:, 0])
    # np.argmin <= np.inf
    zero_mask = distances <= 1e-5
    if np.all(zero_mask):
        raise ValueError("exclude nearly horizonal vector")
    distances[zero_mask] = np.inf

    anchor_idx = np.argmin(distances)
    v_d = distances[anchor_idx]
    
    if v_d > threshold_d or np.isinf(v_d):
        print("v_d not effective: ", v_d)
        return None, np.array([])

    left_I = np.copy(mp[:, 2])   
    right_I = np.copy(mp[:, 3])  

    invalid_mask = (mp[:, 0] > threshold_d) | zero_mask
    left_I[invalid_mask] = -1  
    right_I[invalid_mask] = -1 

    backward_chain =[]
    curr_left = left_I[anchor_idx]

    while curr_left != -1 and curr_left not in backward_chain:
        backward_chain.append(curr_left)
        curr_left = left_I[curr_left]
        
    backward_chain.reverse() 
    atsc_chain = backward_chain + [anchor_idx]
    
    atsc_chain_v = np.array([raw_array[idx : idx + m] for idx in atsc_chain])
    return atsc_chain, atsc_chain_v


def evaluate_and_build_fsm(
    panel_df: pl.DataFrame, 
    motif: np.ndarray, 
    config: dict,
    macro_dict: dict,
    gpd_dict: dict,
    quantiles: list,
    stats_window: list 
): 
    num_bins = len(quantiles) + 1
    fsm_prior_matrix = np.ones((3, num_bins))
    
    # =========================================================
    # 1. Numpy + C 
    # =========================================================
    curves = np.stack(panel_df["curve"].to_numpy()) 
    z_curves = robust_z_normalize(curves)
    z_motif = robust_z_normalize(motif)

    z_curves_c = np.ascontiguousarray(z_curves, dtype=np.float64)
    z_motif_c = np.ascontiguousarray(z_motif, dtype=np.float64)
    
    dtw_window = calculate_dtw_params(config) 
    threshold_d = config["threshold_d"]
    
    distances = np.array([
        dtw.distance_fast(
            row, 
            z_motif_c, 
            window=dtw_window, 
            max_dist=threshold_d
        ) for row in z_curves_c
    ]) 
    
    # =========================================================
    # 2. Polars LazyFrame
    # =========================================================
    lf = panel_df.with_columns(pl.Series("distance", distances)).lazy()
    
    lf_eval = lf.with_columns(
        pl.col("day").replace(macro_dict, default=1).alias("macro_state")
    )
    
    lf_triggers = lf_eval.filter(pl.col("distance") < threshold_d)
    
    # Rust Graph
    eval_df, triggers = pl.collect_all([lf_eval, lf_triggers])
    
    if len(triggers) < 10:
        return {"status": "failed", "reason": "总触发次数过少 (<10)", "metrics_score": -np.inf}
         
    # =========================================================
    # 3. abandon iter_rows zip 100 times than iter_rows
    # =========================================================
    trigger_days = triggers["day"].to_numpy()
    trigger_macros = triggers["macro_state"].to_numpy()
    # Polars null auto to Numpy float np.nan
    # trigger_zrets = triggers["fwd_ret_1"].to_numpy() 
    trigger_zrets = triggers["fwd_ret_1_z"].to_numpy() 
    
    for d, m_state, r in zip(trigger_days, trigger_macros, trigger_zrets):
        if np.isnan(r):
            continue
            
        edges, _ = gpd_dict.get(d, (None, None))
        if edges is None: 
            continue

        bin_idx = np.digitize(r, edges)
        bin_idx = min(max(bin_idx, 0), num_bins - 1) 
        fsm_prior_matrix[m_state, bin_idx] += 1.0

    # =========================================================
    # 4. stats test Numpy vectorize
    # =========================================================
    ks_results = {}
    passed_alpha_test = False
    
    for fw in stats_window:
        col_name = f"fwd_ret_{fw}"
        
        cond_rets = triggers[col_name].drop_nulls().to_numpy()
        uncond_rets = eval_df[col_name].drop_nulls().to_numpy()
        
        if len(cond_rets) < 5:
            continue
            
        ks_stat, ks_pval = stats.ks_2samp(cond_rets, uncond_rets)
        
        cond_mean_val = np.mean(cond_rets) 
        uncond_mean_val = np.mean(uncond_rets) 
        
        if cond_mean_val > uncond_mean_val:
            u_stat, u_pval = stats.mannwhitneyu(cond_rets, uncond_rets, alternative='greater')
        else:
            u_stat, u_pval = stats.mannwhitneyu(cond_rets, uncond_rets, alternative='less')

        score = evaluate_objective(u_pval, cond_mean_val, uncond_mean_val, len(motif), config["penalty_m"])
        
        ks_results[f"T+{fw}"] = {
            "cond_mean": cond_mean_val,
            "uncond_mean": uncond_mean_val,
            "ks_pval": ks_pval,
            "u_pval": u_pval,
            "score": score
        }
        if u_pval < 0.05:
            passed_alpha_test = True
    
    raw_score = max([k["score"] for k in ks_results.values()]) if ks_results else -np.inf

    if not passed_alpha_test:
        return {"status": "failed", "reason": "KS/MW-U 检验不显著", "metrics_score": raw_score}

    return {
        "status": "success",
        "fsm_trigger_count": len(triggers), 
        "ks_results": ks_results,
        "fsm_prior_matrix": fsm_prior_matrix,
        "learned_motif": motif.tolist(),
        "metrics_score": raw_score
    }
