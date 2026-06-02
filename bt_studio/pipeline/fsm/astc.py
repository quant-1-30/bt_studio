#! /usr/bin/env python3
# -*- encondig: utf-8 -*-

import stumpy
import numpy as np
import polars as pl
import scipy.stats as stats
from typing import List, Any, Dict, Tuple, Optional
from dtaidistance import dtw

from bt_studio.pipeline.fsm.preprocess import *
from bt_studio.utils.common import *


def build_panel_from_chunk(hf_dfs: dict, daily_ret_df: pl.DataFrame, config: dict, signal_type: str) -> pl.DataFrame:
    if not hf_dfs:
        return pl.DataFrame()

    # ========================================================
    # 1. extract feature
    # ========================================================
    chunk_records = []
    for sid_bytes, hf_df in hf_dfs.items():
        records = extract_asset_feature(hf_df, config["downsample"], config["m"], amplify=1000)
        
        for r in records:
            r["sid"] = sid_bytes 
            chunk_records.append(r)
        
    if not chunk_records:
        return pl.DataFrame()
    
    # ========================================================
    # 2. force Schema avoid Polars Object type due to mixed data
    # ========================================================
    snapshot_panel = pl.DataFrame(
        chunk_records,
        schema={
            "sid": pl.Binary, 
            "day": pl.Int32, 
            "curve": pl.List(pl.Float64)
        }
    )
    
    # ========================================================
    # 3. Inner Join 
    # ========================================================
    panel_df = (
        snapshot_panel.join(
            daily_ret_df, 
            on=["sid", "day"],
            how="inner" 
        )
        .sort(["sid", "day"])
        # fwd_ret_1 / fwd_ret_1_z 
        .drop_nulls(subset=["curve", "daily_ret", "fwd_ret_1_z"])  
    )
    return panel_df


def build_stumpy_from_chunk(hf_dfs: dict, config: dict, signal_type: str):
    padded =[]
    m = config["m"]
    nan_buffer = np.full(m, np.nan)

    for sid, hf_df in hf_dfs.items():
            if hf_df.height == 0: continue
                
            df_sampled = (
                hf_df.filter(
                    pl.col("datetime").dt.hour() * 60 + pl.col("datetime").dt.minute() <= 14 * 60 + 55
                ).with_columns(
                    intraday_cum_bps = pl.col("intraday_cum") * 1000.0
                ).group_by_dynamic(
                    "datetime", every=f"{config['downsample']}m", closed="right", label="right"
                ).agg(
                    cum_val = pl.col("intraday_cum_bps").last()
                ).drop_nulls(subset=["cum_val"])
            )
            
            valid_series = df_sampled["cum_val"].to_numpy()
            if len(valid_series) > m:
                padded.append(valid_series)
                padded.append(nan_buffer)
            
    return np.concatenate(padded) if padded else np.array([])


def get_atsc(raw_array: np.ndarray, config: dict) -> Tuple[Optional[List[int]], np.ndarray]:
    # filter by person d^2 = 2m(1-r)
    m = config["m"]
    threshold_d = config["threshold_d"]
    
    # Matrix Profile 
    mp = stumpy.stump(raw_array, m=m)

    distances = np.copy(mp[:, 0])
    zero_mask = distances <= 1e-5
    if np.all(zero_mask):
        return None, np.array([])

    distances[zero_mask] = np.inf
    anchor_idx = int(np.argmin(distances))
    v_d = distances[anchor_idx]
    
    if v_d > threshold_d or np.isinf(v_d):
        return None, np.array([])

    left_I = np.copy(mp[:, 2])   
    invalid_mask = (mp[:, 0] > threshold_d) | zero_mask
    left_I[invalid_mask] = -1  

    # Cycle Detection
    backward_chain = []
    curr_left = left_I[anchor_idx]
    visited = set()

    while curr_left != -1 and curr_left not in visited:
        visited.add(curr_left)
        backward_chain.append(curr_left)
        curr_left = left_I[curr_left]
        
    backward_chain.reverse() 
    atsc_chain = backward_chain + [anchor_idx]
    
    atsc_chain_v = np.array([raw_array[idx : idx + m] for idx in atsc_chain])
    return atsc_chain, atsc_chain_v


# =================================================================================
# 2. State Machine Evaluation and Prior Construction
# =================================================================================

def evaluate_and_build_fsm(
    panel_df: pl.DataFrame, 
    motif: np.ndarray, 
    config: dict,
    macro_dict: dict,
    gpd_dict: dict,
    quantiles: list,
    stats_window: list 
) -> dict: 
    num_bins = len(quantiles) + 1

    # Laplace Smoothing
    fsm_prior_matrix = np.ones((3, num_bins), dtype=np.float64) 
    
    # 1. Z-Score 
    curves = np.stack(panel_df["curve"].to_numpy()) 
    z_curves_c = np.ascontiguousarray(robust_z_normalize(curves), dtype=np.float64)
    z_motif_c = np.ascontiguousarray(robust_z_normalize(motif), dtype=np.float64)
    
    dtw_window = max(1, int(config["m"] * config["dtw_window_frac"]))
    threshold_d = config["threshold_d"]
    
    # Motif DTW 
    distances = np.array([
        dtw.distance_fast(row, z_motif_c, window=dtw_window, max_dist=threshold_d) 
        for row in z_curves_c
    ]) 
    
    # 2. Polars Graph 
    lf = panel_df.with_columns(pl.Series("distance", distances)).lazy()
    
    # replace_strict
    lf_eval = lf.with_columns(
        pl.col("day").replace_strict(macro_dict, default=1).alias("macro_state")
    )
    lf_triggers = lf_eval.filter(pl.col("distance") < threshold_d)
    
    eval_df, triggers = pl.collect_all([lf_eval, lf_triggers])
    
    # avoid np.inf in ray , Continuous Reward is suitable
    if triggers.height < 10:
        return {"status": "failed", "reason": "触发次数过少<10", "metrics_score": 0.0}

    # FSM prior construction
    trigger_days = triggers["day"].to_numpy()
    trigger_macros = triggers["macro_state"].to_numpy()
    trigger_zrets = triggers["fwd_ret_1_z"].to_numpy() 
    
    for d, m_state, r in zip(trigger_days, trigger_macros, trigger_zrets):
        if np.isnan(r): continue
            
        edges, _ = gpd_dict.get(d, (None, None))
        if edges is None: continue

        bin_idx = min(max(np.digitize(r, edges), 0), num_bins - 1) 
        fsm_prior_matrix[m_state, bin_idx] += 1.0

    # 4. 统计检验 (KS-Test & MW-U Test)
    ks_results = {}
    any_window_passed_soft = False 
    any_window_passed_hard = False 
    
    for fw in stats_window:
        col_name = f"fwd_ret_{fw}"
        if col_name not in triggers.columns: continue
            
        cond_rets = triggers[col_name].drop_nulls().to_numpy()
        uncond_rets = eval_df[col_name].drop_nulls().to_numpy()
        
        if len(cond_rets) < 5 or np.std(cond_rets) < 1e-8:
            continue
            
        ks_stat, ks_pval = stats.ks_2samp(cond_rets, uncond_rets)
        cond_mean_val, uncond_mean_val = np.mean(cond_rets), np.mean(uncond_rets) 
        
        alt = 'greater' if cond_mean_val > uncond_mean_val else 'less'
        # U-Test
        try:
            u_stat, u_pval = stats.mannwhitneyu(cond_rets, uncond_rets, alternative=alt)
        except ValueError:
            continue

        score = evaluate_objective(u_pval, cond_mean_val, uncond_mean_val, config["m"], config["penalty_m"])

        # =================================================================
        # 🌟 业务规则：分数计算与显著性判定
        # =================================================================
        current_window_score = 0.0
        
        if u_pval <= 0.05:
            current_window_score = base_score
            any_window_passed_soft = True
            any_window_passed_hard = True
        elif u_pval <= 0.15:
            current_window_score = base_score * 0.1 # penalty for 0.05 < p <= 0.15
            any_window_passed_soft = True
        else:
            current_window_score = 0.0

        ks_results[f"T+{fw}"] = {
            "cond_mean": float(cond_mean_val),
            "uncond_mean": float(uncond_mean_val),
            "ks_pval": float(ks_pval),
            "u_pval": float(u_pval),
            "score": float(current_window_score)
        }

    # =================================================================
    # 🌟 decision
    # =================================================================
    if not any_window_passed_soft:
        return {
            "status": "failed", 
            "reason": "所有窗口均未达到显著性软门槛(p>0.15)", 
            "metrics_score": 0.0,
            "passed_strict_alpha": False
        }

    # 2. high score Ray Tune 
    raw_score = max([v["score"] for v in ks_results.values()])

    return {
        "status": "success",
        "passed_strict_alpha": any_window_passed_hard, 
        "fsm_trigger_count": len(triggers), 
        "ks_results": ks_results,
        "fsm_prior_matrix": fsm_prior_matrix.tolist(),
        "learned_motif": motif.tolist(),
        "metrics_score": raw_score   
    }


    return {
        "status": "success",
        "fsm_trigger_count": len(triggers), 
        "ks_results": ks_results,
        "fsm_prior_matrix": fsm_prior_matrix.tolist(), # JSON/XCom
        "learned_motif": motif.tolist(),
        "metrics_score": raw_score
    }

