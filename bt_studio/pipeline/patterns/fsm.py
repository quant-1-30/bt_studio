import polars as pl
import numpy as np
import scipy.stats as stats

from typing import List, Dict, Any

from .astc import calc_min_subseq_dtw
from bt_studio.pipeline.metrics import calculate_hpo_score
from bt_studio.utils.common import calculate_decay_weights


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
    
    # ==================================================================================
    # 1. Macro States Rolling Rank & Return Bins 0(flow in ) / 1(vibrate) / 2(flow out) 
    # ==================================================================================
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
        .with_columns(pl.col("macro_state").shift(1))
        .drop(["p33", "p67", "daily_ofi_mean"])
        .drop_nulls()
    )

    daily_macro = daily_macro_lf.collect()
    
    eval_df = panel_df.join(daily_macro.select(["day", "macro_state"]), on="day", how="left")

    # =================================================================
    # 2. Time-Adjusted Zero-Anchored Bins Based on Rank not std
    # =================================================================
    ranking_ratio = common_config["ranking_ratio"]
    bin_cols = [] 

    for fw in common_config["stats_windows"]:
        col = f"fwd_ret_{fw}"
        if col not in eval_df.columns: continue

        bin_cols.append( f"bin_{fw}") 
        eval_df = eval_df.with_columns([
            # (pl.col(col) / (pl.col("vol_20d") * np.sqrt(fw))).alias(f"z_abs_{fw}")
            (pl.col(col).rank(method="average") / pl.len()).over("day").alias(f"rank_{fw}") # # average solve same ranke and normalize to [0,1]
        ]).with_columns([
            pl.when(pl.col(f"rank_{fw}") <= ranking_ratio).then(0)                
            .when(pl.col(f"rank_{fw}") <= 0.50).then(1)                       
            .when(pl.col(f"rank_{fw}") <= (1.0 - ranking_ratio)).then(2)         
            .otherwise(3).cast(pl.Int32).alias(f"bin_{fw}")                    
        ])# .drop(f"rank_{fw}")

    # =======================================================================
    # 3. Route Rank And Route Ret Polars Expr
    # =======================================================================
    traj_weights = calculate_decay_weights(common_config["stats_windows"], half_life=common_config["decay"]) 
    
    traj_rank_expr = pl.lit(0.0) # 字面量
    traj_ret_expr = pl.lit(0.0)
    total_weight = 0.0
    
    for fw in common_config["stats_windows"]:
        w = traj_weights[fw]
        ret_col = f"fwd_ret_{fw}"
        rank_col = f"rank_{fw}"
        
        if rank_col in eval_df.columns:
            traj_rank_expr = traj_rank_expr + pl.col(rank_col) * w
            traj_ret_expr = traj_ret_expr + pl.col(ret_col) * w
            total_weight += w
            
    # Normalize
    traj_rank_expr = traj_rank_expr / total_weight
    traj_ret_expr = traj_ret_expr / total_weight

    eval_df = eval_df.with_columns([
            traj_rank_expr.alias("rank_trajectory"),
            traj_ret_expr.alias("ret_trajectory") 
        ])

    # =================================================================
    # 4. DTW Triggers Filter by Eval
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
    # 5. Bin Weights and Markov Laplace
    # =================================================================
    bin_weights = {}

    for fw in common_config["stats_windows"]:
        b_col, r_col = f"bin_{fw}", f"fwd_ret_{fw}"
        
        # prior median-ret
        global_grouped = (
            eval_df.group_by(b_col)
            .agg(pl.col(r_col).median().alias("global_ret"))
            .drop_nulls()
            .sort(b_col)
        )
        prior_map = {row[0]: row[1] for row in global_grouped.iter_rows()} # bin_col, median-ret
        
        unique_bins = sorted(list(prior_map.keys()))
        
        trigger_grouped = (
            triggers.group_by(b_col)
            .agg(pl.col(r_col).median().alias("local_ret"))
            .drop_nulls()
            .sort(b_col)
        )
        trigger_map = {row[0]: row[1] for row in trigger_grouped.iter_rows()}
        
        # mix global and trigger
        raw_weights = [trigger_map.get(b, prior_map.get(b, 0.0)) for b in unique_bins]
        bin_weights[fw] = np.round(raw_weights, 6).tolist()

    fsm_matrix = extract_fsm_matrix(triggers, bin_cols)
    # np.round 2D np.array clip
    for k in ["P(T1|Macro)", "P(T2|T1)", "P(T3|T2)"]:
        if k in fsm_matrix:
            fsm_matrix[k] = np.round(fsm_matrix[k], 6).tolist()
    
    fsm_matrix["bin_weights"] = bin_weights

    if skip_stats:
        return {
            "status": "success",
            "fsm_matrix": fsm_matrix,
            "trigger_count": triggers.height,
            "learned_motif": np.round(motif, 6).tolist()
        }

    # =================================================================
    # 6. Rank and Ret Statistics u_pval
    # =================================================================

    cond_ranks = triggers["rank_trajectory"].drop_nulls().to_numpy()
    uncond_ranks = eval_df["rank_trajectory"].drop_nulls().to_numpy() 
    
    n_triggers = len(cond_ranks)
    if n_triggers < 30 or np.std(cond_ranks) < 1e-8: # Clt theory 30
         return {"status": "failed", "reason": f"Triggers ({n_triggers}) < 30", "metrics_score": -9999.0}
    
    try:
        u_stat, u_pval = stats.mannwhitneyu(cond_ranks, uncond_ranks, alternative=common_config["alternative"])
    except ValueError:
        return {"status": "failed", "reason": "MW-U 检验数学越界", "metrics_score": -9999.0}  
    
    # =================================================================
    # 7. Calculate Hpo Score 
    # =================================================================
    cond_rets = triggers["ret_trajectory"].drop_nulls().to_numpy()
    uncond_rets = eval_df["ret_trajectory"].drop_nulls().to_numpy()

    score = calculate_hpo_score(
        u_pval, len(cond_rets), cond_rets, uncond_rets, tune_config, common_config)

    if score <= -9990.0:
        return {"status": "failed", "reason": f"(P-val={u_pval:.4f})", "metrics_score": -9999.0}

    return {
        "status": "success",
        "fsm_matrix": fsm_matrix,
        "trigger_count": triggers.height,
        "learned_motif": np.round(motif, 6).tolist(),
        "metrics_score": round(score, 6),             
        "u_pval": round(float(u_pval), 6)            
    }
