import polars as pl
import numpy as np
import stumpy

from typing import List, Dict, Any


def build_fsm_panel(aligned_lfs: list[pl.LazyFrame], daily_lf: pl.LazyFrame, config: dict) -> pl.DataFrame:
    bars_per_day = 240 // config["downsample"]
    
    all_feat_lf = pl.concat(aligned_lfs)
    
    # std + ret
    daily_ret_lf = (
        daily_lf.sort(["sid", "day"])
        .with_columns([
            (pl.col("close") / pl.col("close").shift(1).over("sid") - 1.0).alias("daily_ret")
        ])
        .with_columns([
            pl.col("daily_ret").rolling_std(window_size=20, min_periods=5)
              .over("sid").fill_null(strategy="forward")
              .clip(lower_bound=0.005).alias("vol_20d"),
              
            (pl.col("close").shift(-1).over("sid") / pl.col("close") - 1.0).alias("fwd_ret_1"),
            (pl.col("close").shift(-2).over("sid") / pl.col("close") - 1.0).alias("fwd_ret_2"),
            (pl.col("close").shift(-3).over("sid") / pl.col("close") - 1.0).alias("fwd_ret_3")
        ])
    )

    curve_lf = (
        all_feat_lf
        .sort(["day", "sid", "bar_idx"])
        .group_by(["day", "sid"])
        .agg([
            pl.col("ofi_ratio").alias("daily_curve"),
            pl.col("ofi_ratio").count().alias("curve_len") 
        ])
        .filter(pl.col("curve_len") == bars_per_day) 
    )

    # cross Ndays concat
    shift_exprs = [
        pl.col("daily_curve").shift(i).over("sid").alias(f"lag_{i}") 
        for i in reversed(range(cross_days))
    ]

    curve_lf = (
        curve_lf.sort(["sid", "day"])
        .with_columns(shift_exprs)
        .drop_nulls(subset=[f"lag_{i}" for i in range(cross_days)]) 
    )

    panel_lf = curve_lf.join(
        daily_ret_lf.select(["day", "sid", "vol_20d", "fwd_ret_1", "fwd_ret_2", "fwd_ret_3"]),
        on=["day", "sid"], how="inner"
    )
    return panel_lf


def prepare_stumpy_array(curves_2d: np.ndarray, config: dict) -> np.ndarray:
    if curves_2d.size == 0:
        return np.array([])
        
    m = config["m"]
    # repair raw curve
    clean_curves = np.nan_to_num(curves_2d, nan=0.0, posinf=0.0, neginf=0.0)
    
    nan_buffer = np.full((clean_curves.shape[0], m), np.nan)
    
    stumpy_arr = np.hstack([clean_curves, nan_buffer]).flatten()
    
    # abundan last m np.nan
    return stumpy_arr[:-m] 


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
    
    zero_mask = distances <= 1e-5
    distances[zero_mask] = np.inf

    candidates = []
    
    for _ in range(top_k):
        anchor_idx = int(np.nanargmin(distances)) # argmin
        v_d = distances[anchor_idx]
        
        if v_d > threshold_d or np.isinf(v_d):
            break
            
        candidate_motif = raw_array[anchor_idx : anchor_idx + m]
        candidates.append(candidate_motif)
        
        # --- Exclusion Zone ---
        # anchor around m ---> inf incase pattern shift
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
        
    min_dist = np.inf
    
    for i in range(L - motif_len + 1):
        sub_seq = row_curve[i : i + motif_len]
        
        std = np.std(sub_seq) + 1e-8
        z_sub = (sub_seq - np.mean(sub_seq)) / std

        z_sub = np.ascontiguousarray(z_sub, dtype=np.float64)
        
        current_limit = min(min_dist, threshold_d) 
        d = dtw.distance_fast(z_sub, z_motif, window=dtw_w, max_dist=current_limit)
        
        if d < min_dist:
            min_dist = d
    return min_dist


def evaluate_and_build_fsm(
    panel_df: pl.DataFrame, 
    curves_2d: np.ndarray,
    motif: np.ndarray, 
    tune_config: dict,
    common_config: dict
    ) -> dict:

    base_score = 100.0
    
    # =====================================================================
    # 1. Macro States) & Return Bins 0(flow in ) / 1(vibrate) / 2(flow out) 
    # =====================================================================
    
    daily_macro_lf = (
        panel_df
        .group_by(["day", "sid"])
        .agg(
            pl.col("daily_curve").sum().alias("sid_ofi_sum")
        )
        .group_by("day")
        .agg(
            pl.col("sid_ofi_sum").mean().alias("daily_ofi_mean")
        )
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
        .drop(["p33", "p67"])
        .sort("day")
    )

    daily_macro = daily_macro_lf.collect()
    
    eval_df = panel_df.join(daily_macro.select(["day", "macro_state"]), on="day", how="left")

   # =================================================================
    # 2. Time-Adjusted Zero-Anchored Bins Based on std
    # =================================================================
    z_bound = tune_config["z_abs_bound"] # 默认为 1.0

    for fw in common_config["stats_windows"]:
        col = f"fwd_ret_{fw}"
        if col not in panel_df.columns: continue
            
        panel_df = panel_df.with_columns([
            # T +1 / T+2 / T+3 std ---> sqrt(T)
            (pl.col(col) / (pl.col("vol_20d") * np.sqrt(fw))).alias(f"z_abs_{fw}")
        ]).with_columns([
            pl.when(pl.col(f"z_abs_{fw}") < -z_bound).then(0)
            .when(pl.col(f"z_abs_{fw}") < 0.0).then(1)
            .when(pl.col(f"z_abs_{fw}") < z_bound).then(2)
            .otherwise(3).cast(pl.Int32).alias(f"bin_{fw}")
        ])

    # =================================================================
    # 3. DTW Triggers
    # =================================================================

    # curves = np.vstack(eval_df["curve"].to_list()).astype(np.float64)
    z_motif = np.ascontiguousarray((motif - np.mean(motif)) / (np.std(motif) + 1e-8), dtype=np.float64)

    distances = Parallel(n_jobs=-1)(
        delayed(calc_min_subseq_dtw)(curve, z_motif, m, dtw_w, threshold_d) 
        for curve in curves_2d
    )
    
    eval_df = eval_df.with_columns(pl.Series("distance", distances))
    triggers = eval_df.filter(pl.col("distance") <= threshold_d)
    
    if triggers.height < 5:
        return {"status": "failed", "reason": f"匹配样本太少 (n={triggers.height})", "metrics_score": 0.0}

    # === 构建马尔可夫转移矩阵  Laplace ===
    trans_t1 = np.ones((3, 4), dtype=np.float64) 
    trans_t1_t2 = np.ones((4, 4), dtype=np.float64) 
    trans_t2_t3 = np.ones((4, 4), dtype=np.float64) 
    
    valid_chain = triggers.drop_nulls(subset=["macro_state", "bin_1", "bin_2", "bin_3"])
    if valid_chain.height > 0:
        for ms, b1, b2, b3 in valid_chain.select(["macro_state", "bin_1", "bin_2", "bin_3"]).rows():
            trans_t1[ms, b1] += 1.0; trans_t1_t2[b1, b2] += 1.0; trans_t2_t3[b2, b3] += 1.0
            
    trans_t1 = (trans_t1 / trans_t1.sum(axis=1, keepdims=True)).tolist()
    trans_t1_t2 = (trans_t1_t2 / trans_t1_t2.sum(axis=1, keepdims=True)).tolist()
    trans_t2_t3 = (trans_t2_t3 / trans_t2_t3.sum(axis=1, keepdims=True)).tolist()

    # === 统计检验 two-sided or greater ===
    cond_rets = triggers["fwd_ret_1"].drop_nulls().to_numpy()
    uncond_rets = eval_df["fwd_ret_1"].drop_nulls().to_numpy() 
    
    if len(cond_rets) < 5 or np.std(cond_rets) < 1e-8:
         return {"status": "failed", "reason": "触发收益方差为0", "metrics_score": 0.0}
    
    try:
        u_stat, u_pval = stats.mannwhitneyu(cond_rets, uncond_rets, alternative=common_config["alternative"]) # 'two-sided' 
    except ValueError:
        return {"status": "failed", "reason": "MW-U 检验数学越界", "metrics_score": 0.0}
    
    score = 100.0 if u_pval <= 0.05 else (10.0 if u_pval <= 0.10 else 0.0)
    if score == 0.0:
        return {"status": "failed", "reason": f"缺乏统计显著性 (P-val={u_pval:.4f})", "metrics_score": 0.0}

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
    prior_config: dict,
    search_config: dict, 
) -> Dict[str, Any]:
    
    # config
    cross_days = int(search_config["cross_days"])
    m = int(search_config["motif_minutes"] // search_config["downsample"])
    threshold_d = float(np.sqrt(2 * m * (1.0 - search_config.get("threshold_r", 0.85))))
    
    tune_config = search_config.copy()
    tune_config["m"] = m
    tune_config["threshold_d"] = threshold_d
    
    # extract curves_2d
    panel_df = panel_lf.collect()

    lag_cols = [f"lag_{i}" for i in reversed(range(cross_days))]
    lag_arrays = [np.vstack(panel_df[col].to_list()) for col in lag_cols]
    curves_2d = np.hstack(lag_arrays) # Shape: (N, cross_days * bars_per_day)
    
    # extract stumpy feature
    stumpy_1d_array = prepare_stumpy_array(curves_2d, tune_config)
    candidate_motifs = get_candidate_motifs(stumpy_1d_array, tune_config, top_k=5)
    
    if not candidate_motifs: 
        return {"status": "failed", "reason": "未找到候选 Motif"}
    
    best_result, highest_score = None, -1.0
    for motif in candidate_motifs:

        result = evaluate_and_build_fsm(
            panel_df, curves_2d, motif, tune_config, prior_config)
        if result["status"] == "success" and result["metrics_score"] > highest_score:
            highest_score = result["metrics_score"]
            best_result = result
            
    return best_result if best_result else {"status": "failed", "reason": "未能通过统计学检验(P-val > 0.1)"}
