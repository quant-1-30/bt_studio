import os
import polars as pl
import numpy as np
import scipy.stats as stats

from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from .astc import calc_min_subseq_dtw
from bt_studio.pipeline.metrics import calculate_hpo_score
from bt_studio.pipeline.utils import calculate_decay_weights


def extract_fsm_matrix(
    triggers: pl.DataFrame, 
    state_cols: list,
    target_names: list,  
    n_macro_states: int = 3, 
    n_ret_states: int = 5
) -> dict:
    """
    Laplace Matrix
    - n_macro_states: (0, 1, 2 -> 3)
    - n_ret_states: (0, 1, 2, 3, 4 -> 5)
    """
    select_cols = ["macro_state"] + state_cols
    valid_chain = triggers.drop_nulls(subset=select_cols) 
    
    fsm_dict = {}
    if valid_chain.height == 0:
        return fsm_dict

    num_windows = len(target_names)
    
    # Laplace Smoothing ---> np.ones
    trans_macro_t0 = np.ones((n_macro_states, n_ret_states), dtype=np.float64) 
    # T_{i} -> T_{i+1}
    trans_t_t = [np.ones((n_ret_states, n_ret_states), dtype=np.float64) for _ in range(num_windows - 1)]

    matrix_data = valid_chain.select(select_cols).to_numpy()
    for row in matrix_data:
        ms = int(row[0])
        bins = row[1:].astype(int)

        if len(bins) >= 1 and 0 <= ms < n_macro_states and 0 <= bins[0] < n_ret_states:
            trans_macro_t0[ms, bins[0]] += 1.0

        for i in range(len(bins) - 1):
            if 0 <= bins[i] < n_ret_states and 0 <= bins[i + 1] < n_ret_states:
                trans_t_t[i][bins[i], bins[i + 1]] += 1.0 
            
    key_macro = f"P({target_names[0]}|Macro)"
    fsm_dict[key_macro] = np.round(trans_macro_t0 / trans_macro_t0.sum(axis=1, keepdims=True), 6).tolist()
    
    # num_windows
    for i in range(num_windows - 1):
        key = f"P({target_names[i+1]}|{target_names[i]})"
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
    
    rets_window = common_config["T1_rets"] # {"open_5m":  5}   
    
    # =====================================================================================================================
    # 1. Macro States Rolling Rank & Return Bins avoid loopahead
    # =====================================================================================================================
    rank_window = common_config["ranking_window"]
    
    daily_macro_lf = (
        panel_df.lazy()
        .select(["day", "sid", pl.col("lag_0").list.sum().alias("sid_ofi_sum")])
        .group_by("day")
        .agg(pl.col("sid_ofi_sum").median().alias("daily_ofi_median"))
        .sort("day") 
        .with_columns(pl.col("daily_ofi_median").shift(1).alias("prev_ofi_median"))
        .with_columns([
            pl.col("prev_ofi_median")
            .rolling_quantile(quantile=0.33, window_size=rank_window, min_periods=max(1, rank_window//2))
            .alias("p33"),
            pl.col("prev_ofi_median")
            .rolling_quantile(quantile=0.67, window_size=rank_window, min_periods=max(1, rank_window//2))
            .alias("p67")
        ])
        .with_columns(
            pl.when(pl.col("prev_ofi_median") <= pl.col("p33")).then(0)
            .when(pl.col("prev_ofi_median") <= pl.col("p67")).then(1)
            .otherwise(2)
            .fill_null(1) 
            .cast(pl.Int32)
            .alias("macro_state")
        )
        .drop(["p33", "p67", "daily_ofi_median", "prev_ofi_median"])
    )    

    daily_macro = daily_macro_lf.collect(engine="streaming")
    eval_df = panel_df.join(daily_macro.select(["day", "macro_state"]), on="day", how="inner")

    # =====================================================================================================================
    # 2. fwd_z_ret Ranking State ---> [0,1,2,3] 
    # =====================================================================================================================
 
    target_names = list(common_config["T1_rets"].keys()) # e.g. ["open_15m", "open_30m"]
    target_states = [] 

    for target_name in target_names:
        z_score_col = f"fwd_z_{target_name}"
        
        if z_score_col not in eval_df.columns: 
            continue

        state_col = f"state_{target_name}"
        target_states.append(state_col) 
        rank_col = f"rank_{target_name}"
        
        # ranking_ratio = common_config["ranking_ratio"]
        # eval_df = (
        #     eval_df
        #     .with_columns([
        #         (pl.col(z_score_col).rank(method="average") / pl.len()).over("day").alias(rank_col)
        #     ])
        #     .with_columns([
        #         pl.when(pl.col(rank_col) <= ranking_ratio)
        #         .then(0)                
        #         .when(pl.col(rank_col) <= 0.50)
        #         .then(1)                        
        #         .when(pl.col(rank_col) <= (1.0 - ranking_ratio))
        #         .then(2)         
        #         .otherwise(3).
        #         cast(pl.Int32).alias(state_col)                    
        #     ])
        # )

        eval_df = (
            eval_df
            .with_columns([
                (pl.col(z_score_col).rank(method="average") / pl.len()).over("day").alias(rank_col)
            ])
            .with_columns([
                pl.when(pl.col(rank_col) <= 0.10).then(0)                
                  .when(pl.col(rank_col) <= 0.30).then(1)               
                  .when(pl.col(rank_col) <= 0.70).then(2)                
                  .when(pl.col(rank_col) <= 0.90).then(3)                
                  .otherwise(4)                                         
                  .cast(pl.Int32).alias(state_col)                    
            ])
        )

    # =====================================================================================================================
    # 3. Route Rank And Route Ret Polars Expr
    # =====================================================================================================================
    traj_weights = calculate_decay_weights(rets_window, half_life_minutes=common_config["decay_minutes"]) 
    
    traj_rank_expr, traj_ret_expr, traj_intra_expr = pl.lit(0.0), pl.lit(0.0), pl.lit(0.0)
    total_weight = 0.0
    
    for fw in rets_window:
        w = traj_weights[fw]
        ret_col = f"fwd_z_{fw}"  
        rank_col = f"rank_{fw}"
        intra_col = f"intra_{fw}"

        if rank_col in eval_df.columns and ret_col in eval_df.columns:
            traj_rank_expr = traj_rank_expr + pl.col(rank_col) * w
            traj_ret_expr = traj_ret_expr + pl.col(ret_col) * w

            if intra_col in eval_df.columns:
                traj_intra_expr = traj_intra_expr + pl.col(intra_col) * w

            total_weight += w
            
    traj_rank_expr = traj_rank_expr / total_weight
    traj_ret_expr = traj_ret_expr / total_weight

    eval_df = eval_df.with_columns([
        traj_rank_expr.alias("rank_trajectory"),
        traj_ret_expr.alias("ret_trajectory"),
        (traj_intra_expr / total_weight).alias("intra_trajectory") 
    ])

    # =====================================================================================================================
    # 4. DTW Triggers Filter by Eval
    # =====================================================================================================================
    m = tune_config["m"]
    threshold_d = tune_config["threshold_d"]
    dtw_w = max(3, int(m * common_config["dtw_window_frac"]))

    z_motif = np.ascontiguousarray((motif - np.mean(motif)) / (np.std(motif) + 1e-8), dtype=np.float64)

    # [FIX P1-P1] Parallelize DTW with ThreadPoolExecutor.
    # dtaidistance releases the GIL, so threads give true parallelism.
    # This was serial in training (bottleneck for topk candidates × N curves),
    # while predict.py already used ThreadPoolExecutor.
    max_workers = common_config.get("max_workers", os.cpu_count())
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        distances = list(executor.map(
            lambda i: calc_min_subseq_dtw(curves_2d[i], z_motif, m, dtw_w, threshold_d),
            range(curves_2d.shape[0])
        ))
    
    eval_df = eval_df.with_columns(pl.Series("distance", distances))
    triggers = eval_df.filter(pl.col("distance") <= threshold_d)
    complementary_df = eval_df.filter(pl.col("distance") > threshold_d)

    if triggers.height < common_config["trigger"]:
        return {
            "status": "failed", 
            "reason": f"DTW (n={triggers.height}) < trigger", 
            "metrics_score": -500.0
        }
    
    # =====================================================================================================================
    # 5. Autocorrelation
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
    # 6. State Value Vectors for Predictor
    # =====================================================================================================================
    state_return_vectors = {} 
    expected_states = [0, 1, 2, 3]

    for target_name in target_names:
        state_col = f"state_{target_name}"
        z_score_col = f"fwd_z_{target_name}"
        
        # Baseline Returns 
        baseline_df = (
            eval_df.group_by(state_col)
            .agg(pl.col(z_score_col).median().alias("baseline_ret"))
            .drop_nulls()
        )
        baseline_returns_map = {row[0]: row[1] for row in baseline_df.iter_rows()}
        
        # Triggered Returns (Motif)
        triggered_df = (
            triggers.group_by(state_col)
            .agg(pl.col(z_score_col).median().alias("triggered_ret"))
            .drop_nulls()
        )
        triggered_returns_map = {row[0]: row[1] for row in triggered_df.iter_rows()}
        
        value_vector = [
            triggered_returns_map.get(state, baseline_returns_map.get(state, 0.0)) 
            for state in expected_states
        ]
        
        state_return_vectors[target_name] = np.round(value_vector, 6).tolist()
    
    # =====================================================================================================================
    # 7. Markov Laplace Matrix
    # =====================================================================================================================

    fsm_matrix = extract_fsm_matrix(triggers, target_states, target_names)
    
    for k in list(fsm_matrix.keys()):
        if isinstance(fsm_matrix[k], (np.ndarray, list)):
            fsm_matrix[k] = np.round(fsm_matrix[k], 6).tolist()
    
    fsm_matrix["state_return_vectors"] = state_return_vectors

    if skip_stats:
        return {
            "status": "success",
            "fsm_matrix": fsm_matrix,
            "trigger_count": triggers.height,
            "learned_motif": np.round(motif, 6).tolist()
        }

    # =====================================================================================================================
    # 8. Mann-Whitney U Test
    # =====================================================================================================================
    cond_ranks = triggers["rank_trajectory"].drop_nulls().to_numpy()
    uncond_ranks = complementary_df["rank_trajectory"].drop_nulls().to_numpy() 
    
    # U Test at least >= 10
    min_test_samples = max(10, common_config["trigger"] // 3)
    n_triggers = len(cond_ranks)

    if n_triggers < min_test_samples or np.std(cond_ranks) < 1e-8 or len(uncond_ranks) == 0:
         return {
            "status": "failed", 
            "reason": f" Rank_trajectory trigger ({n_triggers}) <= {min_test_samples}", 
            "metrics_score": -500.0
        }
    
    try:
        _, u_pval = stats.mannwhitneyu(cond_ranks, uncond_ranks, alternative=common_config["alternative"])
    except ValueError:
        print("MW-U stats ValueError")
        u_pval = 1 
    
    # =====================================================================================================================
    # 9. Calculate Hpo Score 
    # =====================================================================================================================
    cond_data = triggers.select(["ret_trajectory", "z_gap", "intra_trajectory"]).drop_nulls().to_numpy()
    if len(cond_data) <= 2:
        return {
            "status": "failed", 
            "reason": "cond_data height 0", 
            "metrics_score": -500.0
        }
        
    cond_rets = cond_data[:, 0]
    cond_z_gaps = cond_data[:, 1]
    cond_intra = cond_data[:, 2]
    uncond_rets = complementary_df["ret_trajectory"].drop_nulls().to_numpy()

    score = calculate_hpo_score(
        u_pval, len(cond_rets), cond_rets, uncond_rets, cond_z_gaps, cond_intra, tune_config, common_config
    )
    
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