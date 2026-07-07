import polars as pl
import numpy as np
import scipy.stats as stats

from typing import List, Dict, Any

from .m_astc import calc_min_subseq_dtw_md
from .astc import calc_min_subseq_dtw
from bt_studio.pipeline.metrics import calculate_hpo_score


def extract_fsm_matrix(triggers: pl.DataFrame, bin_cols: list) -> dict:
    """freq count with Laplace smoothing"""
    trans_t1 = np.ones((3, 4), dtype=np.float64) 
    trans_t1_t2 = np.ones((4, 4), dtype=np.float64) 
    trans_t2_t3 = np.ones((4, 4), dtype=np.float64) 

    select_cols = ["macro_state"] + bin_cols
    valid_chain = triggers.drop_nulls(subset=select_cols) 
    
    if valid_chain.height > 0:
        for row in valid_chain.select(select_cols).iter_rows():
            ms = row[0]
            actual_bins = row[1:] 
            if len(actual_bins) >= 1: trans_t1[ms, actual_bins[0]] += 1.0
            if len(actual_bins) >= 2: trans_t1_t2[actual_bins[0], actual_bins[1]] += 1.0
            if len(actual_bins) >= 3: trans_t2_t3[actual_bins[1], actual_bins[2]] += 1.0
            
    return {
        "P(T1|Macro)": (trans_t1 / trans_t1.sum(axis=1, keepdims=True)).tolist(),
        "P(T2|T1)": (trans_t1_t2 / trans_t1_t2.sum(axis=1, keepdims=True)).tolist(),
        "P(T3|T2)": (trans_t2_t3 / trans_t2_t3.sum(axis=1, keepdims=True)).tolist()
    }


def evaluate_and_build_fsm(
    panel_df: pl.DataFrame, 
    curves_2d: np.ndarray,
    motif: np.ndarray, 
    tune_config: dict,
    common_config: dict,
    skip_stats=False,
    ) -> dict:

    if panel_df["sid"].dtype != pl.Binary:
        panel_df = panel_df.with_columns(pl.col("sid").cast(pl.Binary))
    
    # =====================================================================
    # 1. Macro States) & Return Bins 0(flow in ) / 1(vibrate) / 2(flow out) 
    # =====================================================================
    daily_macro_lf = (
        panel_df.lazy()
        .select([
            "day", 
            "sid", 
            pl.col("lag_0").list.sum().alias("sid_ofi_sum")
        ])
        .group_by("day")
        .agg(
            pl.col("sid_ofi_sum").mean().alias("daily_ofi_mean")
        )
        .sort("day")
        # window mean replace daily
        .with_columns(
            pl.col("daily_ofi_mean").rolling_mean(window_size=common_config["macro_window"], min_periods=1).alias("smooth_macro")
        )
        .with_columns([
            pl.col("smooth_macro").quantile(1/3).alias("p33"),
            pl.col("smooth_macro").quantile(2/3).alias("p67"),
        ])
        .with_columns(
            pl.when(pl.col("smooth_macro") <= pl.col("p33")).then(0)
            .when(pl.col("smooth_macro") <= pl.col("p67")).then(1)
            .otherwise(2)
            .cast(pl.Int32)
            .alias("macro_state")
        )
        .with_columns(pl.col("macro_state").shift(1)) # avoid loopahead
        .drop(["p33", "p67", "daily_ofi_mean", "smooth_macro"])
        .drop_nulls()
    )

    daily_macro = daily_macro_lf.collect()
    
    eval_df = panel_df.join(daily_macro.select(["day", "macro_state"]), on="day", how="left")

    # =================================================================
    # 2. Time-Adjusted Zero-Anchored Bins Based on Rank not std
    # =================================================================
    edge_ratio = common_config["edge_ratio"]
    bin_cols = [] 

    for fw in common_config["stats_windows"]:
        col = f"fwd_ret_{fw}"
        if col not in eval_df.columns: continue

        bin_cols.append( f"bin_{fw}") 
        eval_df = eval_df.with_columns([
            # (pl.col(col) / (pl.col("vol_20d") * np.sqrt(fw))).alias(f"z_abs_{fw}")
            (pl.col(col).rank(method="average") / pl.len()).over("day").alias(f"rank_{fw}") # # average solve same ranke and normalize to [0,1]
        ]).with_columns([
            pl.when(pl.col(f"rank_{fw}") <= edge_ratio).then(0)                
            .when(pl.col(f"rank_{fw}") <= 0.50).then(1)                       
            .when(pl.col(f"rank_{fw}") <= (1.0 - edge_ratio)).then(2)         
            .otherwise(3).cast(pl.Int32).alias(f"bin_{fw}")                    
        ]).drop(f"rank_{fw}") 

    # =================================================================
    # 3. DTW Triggers
    # =================================================================
    m = tune_config["m"]
    threshold_d = tune_config["threshold_d"]
    dtw_w = max(3, int(m * common_config["dtw_window_frac"]))

    z_motif = np.ascontiguousarray((motif - np.mean(motif)) / (np.std(motif) + 1e-8), dtype=np.float64)

    distances = [
        calc_min_subseq_dtw(curve, z_motif, m, dtw_w, threshold_d) 
        for curve in curves_2d
    ]
    
    eval_df = eval_df.with_columns(pl.Series("distance", distances))
    triggers = eval_df.filter(pl.col("distance") <= threshold_d)
    
    if triggers.height < 5:
        return {"status": "failed", "reason": f"Matching Not enough (n={triggers.height})", "metrics_score": -9999.0}

    # =================================================================
    # 4. Markov Laplace and Bayesian Prior 
    # =================================================================
    fsm_matrix = extract_fsm_matrix(triggers, bin_cols)

    if skip_stats:
        return {
            "status": "success",
            "fsm_matrix": fsm_matrix,
            "trigger_count": triggers.height,
            "learned_motif": motif.tolist(),
        }

    # =================================================================
    # 5. Statistics Pval
    # =================================================================
    cond_rets = triggers["fwd_ret_1"].drop_nulls().to_numpy()
    uncond_rets = eval_df["fwd_ret_1"].drop_nulls().to_numpy() 
    
    # Clt theory ---> 30 bottom
    if len(cond_rets) < 30 or np.std(cond_rets) < 1e-8:
         return {"status": "failed", "reason": "ret Std 0 means supend or delist", "metrics_score": -9999.0}
    
    try:
        u_stat, u_pval = stats.mannwhitneyu(cond_rets, uncond_rets, alternative=common_config["alternative"])
    except ValueError:
        return {"status": "failed", "reason": "MW-U 检验数学越界", "metrics_score": -9999.0}
    
    # =================================================================
    # 6. Final Score 
    # =================================================================
    score = calculate_hpo_score(
        u_pval, len(cond_rets), cond_rets, uncond_rets, tune_config, common_config)

    if score <= -9990.0:
        return {"status": "failed", "reason": f"(P-val={u_pval:.4f})", "metrics_score": -9999.0}

    return {
        "status": "success",
        "fsm_matrix": fsm_matrix,
        "trigger_count": triggers.height,
        "learned_motif": motif.tolist(),
        "metrics_score": score,
        "u_pval": float(u_pval)
    }



def evaluate_and_build_fsm_md(
    panel_df: pl.DataFrame, 
    curves_md: np.ndarray, # 💡 Shape: (N, D, L)
    motif_md: np.ndarray,  # 💡 Shape: (D, m)
    tune_config: dict,
    common_config: dict,
    skip_stats: bool = False
) -> dict:

    if panel_df["sid"].dtype != pl.Binary:
        panel_df = panel_df.with_columns(pl.col("sid").cast(pl.Binary))
    
    # ======================================================================
    # 1. Macro States) & Return Bins 0(flow in ) / 1(vibrate) / 2(flow out) 
    # ======================================================================
    daily_macro_lf = (
        panel_df.lazy()
        .select([
            "day", 
            "sid", 
            pl.col("lag_0").list.sum().alias("sid_ofi_sum")
        ])
        .group_by("day")
        .agg(
            pl.col("sid_ofi_sum").mean().alias("daily_ofi_mean")
        )
        .sort("day")
        # window mean replace daily
        .with_columns(
            pl.col("daily_ofi_mean").rolling_mean(window_size=common_config["macro_window"], min_periods=1).alias("smooth_macro")
        )
        .with_columns([
            pl.col("smooth_macro").quantile(1/3).alias("p33"),
            pl.col("smooth_macro").quantile(2/3).alias("p67"),
        ])
        .with_columns(
            pl.when(pl.col("smooth_macro") <= pl.col("p33")).then(0)
            .when(pl.col("smooth_macro") <= pl.col("p67")).then(1)
            .otherwise(2)
            .cast(pl.Int32)
            .alias("macro_state")
        )
        .with_columns(pl.col("macro_state").shift(1)) # avoid loopahead
        .drop(["p33", "p67", "daily_ofi_mean", "smooth_macro"])
        .drop_nulls()
    )

    daily_macro = daily_macro_lf.collect()
    
    eval_df = panel_df.join(daily_macro.select(["day", "macro_state"]), on="day", how="left")

    # =======================================================================
    # 2. Time-Adjusted Zero-Anchored Bins Based on Rank not std
    # =======================================================================

    edge_ratio = common_config["edge_ratio"]
    bin_cols = [] 

    for fw in common_config["stats_windows"]:
        col = f"fwd_ret_{fw}"
        if col not in eval_df.columns: continue
        
        bin_cols.append( f"bin_{fw}") 
        eval_df = eval_df.with_columns([
            # (pl.col(col) / (pl.col("vol_20d") * np.sqrt(fw))).alias(f"z_abs_{fw}")
            (pl.col(col).rank(method="average") / pl.len()).over("day").alias(f"rank_{fw}") # # # average solve same ranke and normalize to [0,1]
        ]).with_columns([
            pl.when(pl.col(f"rank_{fw}") <= edge_ratio).then(0)                
            .when(pl.col(f"rank_{fw}") <= 0.50).then(1)                       
            .when(pl.col(f"rank_{fw}") <= (1.0 - edge_ratio)).then(2)         
            .otherwise(3).cast(pl.Int32).alias(f"bin_{fw}")                    
        ]).drop(f"rank_{fw}") 

    # =======================================================================
    # 3. DTW Triggers
    # =======================================================================
    m = tune_config["m"]
    threshold_d = tune_config["threshold_d"]
    dtw_w = max(3, int(m * common_config["dtw_window_frac"]))

    # Motif Z-Score
    m_means = np.mean(motif_md, axis=1, keepdims=True)
    m_stds = np.std(motif_md, axis=1, keepdims=True) + 1e-8
    z_motif_md = (motif_md - m_means) / m_stds

    # N 2D
    distances = [
        calc_min_subseq_dtw_md(curves_md[i], z_motif_md, dtw_w, threshold_d)
        for i in range(curves_md.shape[0])
    ]
    
    eval_df = eval_df.with_columns(pl.Series("distance", distances))
    triggers = eval_df.filter(pl.col("distance") <= threshold_d)
    
    if triggers.height < 5:
        return {"status": "failed", "reason": f"Matching Not enough (n={triggers.height})", "metrics_score": -9999.0}

    # =======================================================================
    # 4. Markov Laplace 
    # =======================================================================
    fsm_matrix = extract_fsm_matrix(triggers, bin_cols)

    if skip_stats:
        return {
            "status": "success",
            "fsm_matrix": fsm_matrix,
            "trigger_count": triggers.height,
            "learned_motif": motif_md.tolist(),
        }

    # =======================================================================
    # 5. Statistics Pval
    # =======================================================================
    cond_rets = triggers["fwd_ret_1"].drop_nulls().to_numpy()
    uncond_rets = eval_df["fwd_ret_1"].drop_nulls().to_numpy() 
    
    if len(cond_rets) < 30 or np.std(cond_rets) < 1e-8:
         return {"status": "failed", "reason": "ret Std 0 means supend or delist", "metrics_score": -9999.0}
    
    try:
        u_stat, u_pval = stats.mannwhitneyu(cond_rets, uncond_rets, alternative=common_config["alternative"])
    except ValueError:
        return {"status": "failed", "reason": "MW-U 检验数学越界", "metrics_score": -9999.0}

    # =======================================================================
    # 6. Final Score
    # =======================================================================
    score = calculate_hpo_score(u_pval, len(cond_rets), cond_rets, uncond_rets, tune_config, common_config)
    
    if score <= -9990.0:
        return {"status": "failed", "reason": f"(P-val={u_pval:.4f})", "metrics_score": -9999.0}

    return {
        "status": "success",
        "fsm_matrix": fsm_matrix,
        "trigger_count": triggers.height,
        "learned_motif": motif_md.tolist(),
        "metrics_score": score,
        "u_pval": float(u_pval)
    }
