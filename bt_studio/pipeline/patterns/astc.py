import polars as pl
import numpy as np
import stumpy
import scipy.stats as stats

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="stumpy")

from dtaidistance import dtw
from typing import List, Dict, Any
from numpy.lib.stride_tricks import sliding_window_view

from bt_studio.pipeline.metrics import calculate_hpo_score


def prepare_curves(panel_df: pl.DataFrame, tune_config: dict, common_config: dict) -> np.ndarray:
    """DataFrame (N, L) tensor and NaN boarder"""
    cross_days = int(tune_config["cross_days"])

    lag_cols = [f"lag_{i}" for i in reversed(range(cross_days))]
    lag_arrays = [np.vstack(panel_df[col].to_list()) for col in lag_cols]
    
    # lag_0 today eg 14:55  np.nan！
    execlude_bars = common_config["exclude_bars"] // int(tune_config["downsample"])
    if execlude_bars > 0:
        lag_arrays[-1][:, -execlude_bars:] = np.nan

    curves_2d = np.hstack(lag_arrays) # Shape: (N, cross_days * bars_per_day)
    return curves_2d    # Shape: (N, L)


def get_candidate_motifs(raw_array: np.ndarray, config: dict, top_k: int = 5) -> List[np.ndarray]:
    if len(raw_array) == 0:
        return []

    m = config["m"]
    threshold_d = config["threshold_d"]
    
    try:
        mp = stumpy.stump(raw_array, m=m)
    except Exception as e:
        print(f"Stumpy failed to process array: {e}")
        return []

    distances = np.copy(mp[:, 0])
    # mp two different dtype --> object --> float64
    distances = np.copy(mp[:, 0]).astype(np.float64)
    distances[distances <= 1e-5] = np.inf

    if np.all(np.isinf(distances)):
        return []

    candidates = []
    for _ in range(top_k):
        anchor_idx = int(np.nanargmin(distances)) # argmin
        v_d = distances[anchor_idx]
        
        if v_d > threshold_d or np.isinf(v_d):
            break
            
        candidate_motif = raw_array[anchor_idx : anchor_idx + m]
        candidates.append(candidate_motif)
        
        # --- Exclusion Zone --- anchor around m keep isolate
        exclude_start = max(0, anchor_idx - m)
        exclude_end = min(len(distances), anchor_idx + m)
        distances[exclude_start:exclude_end] = np.inf
    return candidates


def calc_min_subseq_dtw(
    row_curve: np.ndarray, 
    z_motif: np.ndarray, 
    motif_len: int, 
    dtw_w: int, 
    threshold_d: float,
) -> float:

    L = len(row_curve)
    if L < motif_len:
        return np.inf

    # 1. Shape: (window, motif_len)
    windows = sliding_window_view(row_curve, window_shape=motif_len)
    
    # 2. vector mask
    is_valid_window = ~np.isnan(windows).any(axis=-1)
    
    min_dist = np.inf
    # 3. loop over valid windows
    for i in range(len(windows)):
        if not is_valid_window[i]:
            continue
            
        sub_seq = windows[i]
        
        std = np.std(sub_seq) + 1e-8
        z_sub = (sub_seq - np.mean(sub_seq)) / std
        
        z_sub = np.ascontiguousarray(z_sub, dtype=np.float64)
        
        d = dtw.distance_fast(z_sub, z_motif, window=dtw_w, max_dist=min(min_dist, threshold_d))
        if d < min_dist:
            min_dist = d
            
    return min_dist 


def evaluate_and_build_fsm(
    panel_df: pl.DataFrame, 
    curves_2d: np.ndarray,
    motif: np.ndarray, 
    tune_config: dict,
    common_config: dict,
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
            pl.col("lag_0").list.sum().alias("sid_ofi_sum") # list[f64]
        ])
        .group_by("day")
        .agg(
            pl.col("sid_ofi_sum").mean().alias("daily_ofi_mean")
        )
        .sort("day")
        .with_columns([
            pl.col("daily_ofi_mean").quantile(1/3).alias("p33"),
            pl.col("daily_ofi_mean").quantile(2/3).alias("p67"),
        ])
        .with_columns(
            pl.when(pl.col("daily_ofi_mean") <= pl.col("p33")).then(0)
            .when(pl.col("daily_ofi_mean") <= pl.col("p67")).then(1)
            .otherwise(2)
            .cast(pl.Int32)
            .alias("macro_state")
        )
        # avoid loopforward with shift
        .with_columns(pl.col("macro_state").shift(1))
        .drop(["p33", "p67", "daily_ofi_mean"])
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
            # # T +1 / T+2 / T+3 std ---> sqrt(T)
            # (pl.col(col) / (pl.col("vol_20d") * np.sqrt(fw))).alias(f"z_abs_{fw}")
            # method="average" to calculate Rank Ascending, then normalize to [0,1]
            (pl.col(col).rank(method="average") / pl.len()).over("day").alias(f"rank_{fw}")
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
    trans_t1 = np.ones((3, 4), dtype=np.float64) 
    trans_t1_t2 = np.ones((4, 4), dtype=np.float64) 
    trans_t2_t3 = np.ones((4, 4), dtype=np.float64) 

    select_cols = ["macro_state"] + bin_cols
    valid_chain = triggers.drop_nulls(subset=select_cols) 

    # if valid_chain.height > 0:
    #     for ms, b1, b2, b3 in valid_chain.select(select_cols).rows():
    #         trans_t1[ms, b1] += 1.0; trans_t1_t2[b1, b2] += 1.0; trans_t2_t3[b2, b3] += 1.0

    if valid_chain.height > 0:
        for row in valid_chain.select(select_cols).iter_rows():
            ms = row[0]
            actual_bins = row[1:] 
            
            if len(actual_bins) >= 1:
                b1 = actual_bins[0]
                trans_t1[ms, b1] += 1.0
            if len(actual_bins) >= 2:
                b2 = actual_bins[1]
                trans_t1_t2[b1, b2] += 1.0
            if len(actual_bins) >= 3:
                b3 = actual_bins[2]
                trans_t2_t3[b2, b3] += 1.0
            
    trans_t1 = (trans_t1 / trans_t1.sum(axis=1, keepdims=True)).tolist()
    trans_t1_t2 = (trans_t1_t2 / trans_t1_t2.sum(axis=1, keepdims=True)).tolist()
    trans_t2_t3 = (trans_t2_t3 / trans_t2_t3.sum(axis=1, keepdims=True)).tolist()

    # =================================================================
    # 5. Statistics Pval
    # =================================================================
    cond_rets = triggers["fwd_ret_1"].drop_nulls().to_numpy()
    uncond_rets = eval_df["fwd_ret_1"].drop_nulls().to_numpy() 
    
    if len(cond_rets) < 5 or np.std(cond_rets) < 1e-8:
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
        "fsm_network": {"P(T1|Macro)": trans_t1, "P(T2|T1)": trans_t1_t2, "P(T3|T2)": trans_t2_t3},
        "trigger_count": triggers.height,
        "learned_motif": motif.tolist(),
        "metrics_score": score,
        "u_pval": float(u_pval)
    }


def discover_fsm_pattern(
    panel_lf: pl.LazyFrame,  
    tune_config: dict, 
    common_config: dict
) -> Dict[str, Any]: 

    m = int(tune_config["motif_minutes"] // tune_config["downsample"])
    cross_days = int(tune_config["cross_days"])
    threshold_d = float(np.sqrt(2 * m * (1.0 - tune_config.get("threshold_r", 0.85))))

    # =========================================================================
    # 1. Filter Panel DataFrame
    # =========================================================================
    panel_df = panel_lf.collect(engine="streaming")
    if panel_df.height == 0:
        return {
            "status": "failed", 
            "reason": "HPO Panel_df Zero after filter", 
            "metrics_score": -9999.0
        }
        
    if panel_df.height <= m or m < 3:
        return {
            "status": "failed", 
            "reason": f"Not enough data (n={panel_df.height})", 
            "metrics_score": -9999.0
        }

    # =========================================================================
    # 2. Features Matrix (N,L)
    # =========================================================================
    tune_config["m"] = m
    tune_config["threshold_d"] = threshold_d

    curves_2d = prepare_curves(panel_df, tune_config, common_config)
    N, L = curves_2d.shape

    if curves_2d.size == 0:
        return np.array([])

    # =========================================================================
    # 3. Volatility-Driven Sampling for stumpy
    # =========================================================================
    # diff ---> mutation Shape -> diff (N, L-1) / abs ---> (N, L-1) / nanmax --> (N,)
    # mutation_scores = np.nanmax(np.nanvar(curves_2d, axis=1), axis=1)
    mutation_scores = np.nansum(np.abs(np.diff(curves_2d, axis=1)), axis=1)
    
    theory_points = common_config.get("max_points", 20000) # bug 
    sample_size = min(N, max(5, int(theory_points / L))) 
    
    active_idx = np.argsort(mutation_scores)[-sample_size:]
    sampled_curves = curves_2d[active_idx] # Shape: (Sample_N, L)
    
    # =========================================================================
    # 4. Nans between assets and Stumpy T_multi(Sample_N * L) For Candidates 
    # =========================================================================
    clean_curves = np.copy(sampled_curves)
    clean_curves[np.isinf(clean_curves)] = 0.0 # np.nan_to_num(sampled_curves, nan=0.0, posinf=0.0, neginf=0.0) 
    # m NaN as separate between assets
    nan_buffer = np.full((clean_curves.shape[0], m), np.nan)
    stumpy_1d_array = np.hstack([clean_curves, nan_buffer]).flatten()[:-m] # abundan last m np.nan

    candidate_motifs = get_candidate_motifs(stumpy_1d_array, tune_config, top_k=5)
    
    if not candidate_motifs: 
        return {"status": "failed", "reason": "Not Found Motif", "metrics_score": -9999.0}
    
    best_result, highest_score = None, -9999.0
    for motif in candidate_motifs:

        result = evaluate_and_build_fsm(
            panel_df, curves_2d, motif, tune_config, common_config)
        if result["status"] == "success" and result["metrics_score"] > highest_score:
            highest_score = result["metrics_score"]
            best_result = result
            
    return best_result if best_result else {"status": "failed", "reason": "(P-val > 0.1)", "metrics_score": -9999.0}
