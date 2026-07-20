import polars as pl
import numpy as np
import scipy.stats as stats

from typing import List, Dict, Any

from .astc import calc_min_subseq_dtw
from bt_studio.pipeline.metrics import calculate_hpo_score
from bt_studio.utils.common import calculate_decay_weights


def extract_fsm_matrix(
    triggers: pl.DataFrame, 
    bin_cols: list, 
    n_macro_states: int = 3, 
    n_ret_states: int = 4
) -> dict:
    """
    Laplace Matrix
    - n_macro_states: (0, 1, 2 -> 3)
    - n_ret_states: (0, 1, 2, 3 -> 4)
    """
    select_cols = ["macro_state"] + bin_cols
    valid_chain = triggers.drop_nulls(subset=select_cols) 
    
    fsm_dict = {}
    if valid_chain.height == 0:
        return fsm_dict

    windows = [col.split("_")[-1] for col in bin_cols]
    num_windows = len(windows)
    
    trans_macro_t0 = np.ones((n_macro_states, n_ret_states), dtype=np.float64) 
    # 2. T_{i} -> T_{i+1}
    trans_t_t = [np.ones((n_ret_states, n_ret_states), dtype=np.float64) for _ in range(num_windows - 1)]

    for row in valid_chain.select(select_cols).iter_rows():
        ms = row[0]
        bins = row[1:] 
        
        if len(bins) >= 1: 
            trans_macro_t0[ms, bins[0]] += 1.0
            
        for i in range(len(bins) - 1):
            trans_t_t[i][bins[i], bins[i+1]] += 1.0
            
    key_macro = f"P(T{windows[0]}|Macro)"
    fsm_dict[key_macro] = np.round(trans_macro_t0 / trans_macro_t0.sum(axis=1, keepdims=True), 6).tolist()
    
    # num_windows
    for i in range(num_windows - 1):
        key = f"P(T{windows[i+1]}|T{windows[i]})"
        mat = trans_t_t[i]
        fsm_dict[key] = np.round(mat / mat.sum(axis=1, keepdims=True), 6).tolist()
            
    return fsm_dict


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
    
    stats_windows = common_config["stats_windows"]
    
    # =====================================================================================================================
    # 1. Macro States Rolling Rank & Return Bins avoid loopahead
    # =====================================================================================================================
    rank_window = common_config["ranking_window"]
    
    daily_macro_lf = (
        panel_df.lazy()
        .select(["day", "sid", pl.col("lag_0").list.sum().alias("sid_ofi_sum")]) # lag_0 ---> daily_curve
        .group_by("day")
        .agg(pl.col("sid_ofi_sum").mean().alias("daily_ofi_mean"))
        .sort("day") 
        .with_columns(pl.col("daily_ofi_mean").shift(1).alias("prev_ofi_mean"))
        .drop_nulls(subset=["prev_ofi_mean"])
        .with_columns([
            pl.col("prev_ofi_mean")
              .rolling_quantile(quantile=0.33, window_size=rank_window, min_periods=max(1, rank_window//2)) # half decay
              .alias("p33"),
            pl.col("prev_ofi_mean")
              .rolling_quantile(quantile=0.67, window_size=rank_window, min_periods=max(1, rank_window//2))
              .alias("p67")
        ])
        .with_columns(
            pl.when(pl.col("prev_ofi_mean") <= pl.col("p33")).then(0)
            .when(pl.col("prev_ofi_mean") <= pl.col("p67")).then(1)
            .otherwise(2)
            .cast(pl.Int32).alias("macro_state")
        )
        .drop(["p33", "p67", "daily_ofi_mean", "prev_ofi_mean"])
    )

    daily_macro = daily_macro_lf.collect()
    eval_df = panel_df.join(daily_macro.select(["day", "macro_state"]), on="day", how="inner")

    # =====================================================================================================================
    # 2. Time-Adjusted Zero-Anchored Bins 
    # =====================================================================================================================
    ranking_ratio = common_config["ranking_ratio"]
    bin_cols = [] 

    for fw in stats_windows:
        col = f"fwd_ret_{fw}"
        if col not in eval_df.columns: 
            continue

        bin_col = f"bin_{fw}"
        rank_col = f"rank_{fw}"
        bin_cols.append(bin_col) 
        
        eval_df = eval_df.with_columns([
            (pl.col(col).rank(method="average") / pl.len()).over("day").alias(rank_col)
        ]).with_columns([
            pl.when(pl.col(rank_col) <= ranking_ratio).then(0)                
            .when(pl.col(rank_col) <= 0.50).then(1)                        
            .when(pl.col(rank_col) <= (1.0 - ranking_ratio)).then(2)         
            .otherwise(3).cast(pl.Int32).alias(bin_col)                    
        ])

    if not bin_cols:
        return {"status": "failed", "reason": "not found Z-Score ret col", "metrics_score": -9999.0}

    # =====================================================================================================================
    # 3. Route Rank And Route Ret Polars Expr
    # =====================================================================================================================
    traj_weights = calculate_decay_weights(stats_windows, half_life=common_config["decay"]) 
    
    traj_rank_expr = pl.lit(0.0)
    traj_ret_expr = pl.lit(0.0)
    total_weight = 0.0
    
    for fw in stats_windows:
        w = traj_weights[fw]
        ret_col = f"fwd_ret_{fw}"  
        rank_col = f"rank_{fw}"
        
        if rank_col in eval_df.columns:
            traj_rank_expr = traj_rank_expr + pl.col(rank_col) * w
            traj_ret_expr = traj_ret_expr + pl.col(ret_col) * w
            total_weight += w
            
    traj_rank_expr = traj_rank_expr / total_weight
    traj_ret_expr = traj_ret_expr / total_weight

    eval_df = eval_df.with_columns([
        traj_rank_expr.alias("rank_trajectory"),
        traj_ret_expr.alias("ret_trajectory") 
    ])

    # =====================================================================================================================
    # 4. DTW Triggers Filter by Eval
    # =====================================================================================================================
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
    complementary_df = eval_df.filter(pl.col("distance") > threshold_d)
    
    if triggers.height < 5:
        return {"status": "failed", "reason": f"DTW (n={triggers.height}) not enough", "metrics_score": -9999.0}
    
    # =====================================================================================================================
    # 5. Valid Ratio and AutoCorr
    # =====================================================================================================================

    valid_sample_ratio = triggers.height / max(eval_df.height, 1)

    eval_df = (
        eval_df.with_columns(
            (pl.col("distance") <= threshold_d).cast(pl.Int8).alias("is_trigger")
        )
        .sort(["sid", "day"])
        .with_columns(
            pl.col("is_trigger").shift(1).over("sid").alias("prev_trigger") # shift.over
        )
    )
    
    autocorr = float(
        eval_df.select(
            pl.corr("is_trigger", "prev_trigger").fill_nan(0.0).fill_null(0.0)
        )
        .to_series()[0] 
    )

    # =====================================================================================================================
    # 5. Bin Weights and Markov Laplace 
    # =====================================================================================================================
    bin_weights = {}

    for fw in stats_windows:
        b_col, r_col = f"bin_{fw}", f"fwd_ret_{fw}"
        
        global_grouped = eval_df.group_by(b_col).agg(pl.col(r_col).median().alias("global_ret")).drop_nulls().sort(b_col)
        prior_map = {row[0]: row[1] for row in global_grouped.iter_rows()}
        unique_bins = sorted(list(prior_map.keys()))
        
        trigger_grouped = triggers.group_by(b_col).agg(pl.col(r_col).median().alias("median_ret")).drop_nulls().sort(b_col)
        trigger_map = {row[0]: row[1] for row in trigger_grouped.iter_rows()}
        
        raw_weights = [trigger_map.get(b, prior_map.get(b, 0.0)) for b in unique_bins]
        bin_weights[fw] = np.round(raw_weights, 6).tolist()

    fsm_matrix = extract_fsm_matrix(triggers, bin_cols)
    
    for k in list(fsm_matrix.keys()):
        if isinstance(fsm_matrix[k], (np.ndarray, list)):
            fsm_matrix[k] = np.round(fsm_matrix[k], 6).tolist()
    
    fsm_matrix["bin_weights"] = bin_weights

    if skip_stats:
        return {
            "status": "success",
            "fsm_matrix": fsm_matrix,
            "trigger_count": triggers.height,
            "learned_motif": np.round(motif, 6).tolist()
        }

    # =====================================================================================================================
    # 6. Rank and Ret Statistics u_pval
    # =====================================================================================================================
    cond_ranks = triggers["rank_trajectory"].drop_nulls().to_numpy()
    uncond_ranks = complementary_df["rank_trajectory"].drop_nulls().to_numpy() 
    
    n_triggers = len(cond_ranks)
    if n_triggers < 10 or np.std(cond_ranks) < 1e-8 or len(uncond_ranks) == 0:
         return {"status": "failed", "reason": f" Rank_trajectory trigger ({n_triggers}) <= 10", "metrics_score": -9999.0}
    
    try:
        u_stat, u_pval = stats.mannwhitneyu(cond_ranks, uncond_ranks, alternative=common_config["alternative"])
    except ValueError:
        return {"status": "failed", "reason": "MW-U stats ValueError", "metrics_score": -9999.0}  
    
    # =====================================================================================================================
    # 7. Calculate Hpo Score 
    # =====================================================================================================================
    cond_rets = triggers["ret_trajectory"].drop_nulls().to_numpy()
    uncond_rets = complementary_df["ret_trajectory"].drop_nulls().to_numpy()

    score = calculate_hpo_score(
        u_pval, len(cond_rets), cond_rets, uncond_rets, tune_config, common_config
    )

    if score <= -9990.0:
        return {"status": "failed", "reason": f"HPO Score Reach -9999.0 ", "metrics_score": -9999.0}

    return {
        "status": "success",
        "fsm_matrix": fsm_matrix,
        "trigger_count": triggers.height,
        "learned_motif": np.round(motif, 6).tolist(),
        "metrics_score": round(score, 6),     
        "valid_sample_ratio": valid_sample_ratio,
        "autocorr": autocorr,      
        "u_pval": round(float(u_pval), 6)            
    }
