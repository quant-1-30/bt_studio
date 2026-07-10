
import polars as pl
import numpy as np
import scipy.stats as stats

from typing import List, Dict, Any

from .mastc import calc_min_subseq_dtw_md
from bt_studio.pipeline.metrics import calculate_hpo_score


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
    rank_window = common_config["ranking_window"]
    daily_macro_lf = (
        panel_df.lazy()
        .select(["day", "sid", pl.col("lag_0").list.sum().alias("sid_ofi_sum")])
        .group_by("day")
        .agg(pl.col("sid_ofi_sum").mean().alias("daily_ofi_mean"))
        .sort("day") 
        .with_columns([
            pl.col("daily_ofi_mean")
              .rolling_quantile(quantile=0.33, window_size=rank_window, min_periods=5)
              .alias("p33"),
            pl.col("daily_ofi_mean")
              .rolling_quantile(quantile=0.67, window_size=rank_window, min_periods=5)
              .alias("p67")
        ])
        .with_columns(
            pl.when(pl.col("daily_ofi_mean") <= pl.col("p33")).then(0)
            .when(pl.col("daily_ofi_mean") <= pl.col("p67")).then(1)
            .otherwise(2)
            .cast(pl.Int32)
            .alias("macro_state")
        )
        .with_columns(pl.col("macro_state").shift(1))  # avoid loopahead
        .drop(["p33", "p67", "daily_ofi_mean"])
        .drop_nulls()
    )

    daily_macro = daily_macro_lf.collect()
    
    eval_df = panel_df.join(daily_macro.select(["day", "macro_state"]), on="day", how="left")

    # =======================================================================
    # 2. Time-Adjusted Zero-Anchored Bins Based on Rank not std
    # =======================================================================

    ranking_ratio = common_config["ranking_ratio"]
    bin_cols = [] 

    for fw in common_config["stats_windows"]:
        col = f"fwd_ret_{fw}"
        if col not in eval_df.columns: continue
        
        bin_cols.append( f"bin_{fw}") 
        eval_df = eval_df.with_columns([
            # (pl.col(col) / (pl.col("vol_20d") * np.sqrt(fw))).alias(f"z_abs_{fw}")
            (pl.col(col).rank(method="average") / pl.len()).over("day").alias(f"rank_{fw}") # # # average solve same ranke and normalize to [0,1]
        ]).with_columns([
            pl.when(pl.col(f"rank_{fw}") <= ranking_ratio).then(0)                
            .when(pl.col(f"rank_{fw}") <= 0.50).then(1)                       
            .when(pl.col(f"rank_{fw}") <= (1.0 - ranking_ratio)).then(2)         
            .otherwise(3).cast(pl.Int32).alias(f"bin_{fw}")                    
        ])# .drop(f"rank_{fw}") 

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
    # 5. Ranking Statistics Pval
    # =======================================================================
    cond_ranks = triggers["rank_1"].drop_nulls().to_numpy()
    uncond_ranks = eval_df["rank_1"].drop_nulls().to_numpy() 
    
    n_triggers = len(cond_ranks)
    if n_triggers < 30 or np.std(cond_ranks) < 1e-8:
         return {"status": "failed", "reason": f"Triggers ({n_triggers}) < 30", "metrics_score": -9999.0}
    
    try:
        u_stat, u_pval = stats.mannwhitneyu(cond_ranks, uncond_ranks, alternative=common_config["alternative"])
    except ValueError:
        return {"status": "failed", "reason": "MW-U 检验数学越界", "metrics_score": -9999.0} 
    
    # =======================================================================
    # 6. Final Score
    # =======================================================================
    cond_rets = triggers["fwd_ret_1"].drop_nulls().to_numpy()
    uncond_rets = eval_df["fwd_ret_1"].drop_nulls().to_numpy()

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
