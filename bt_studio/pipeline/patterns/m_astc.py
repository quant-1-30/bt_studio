import stumpy
import numpy as np
from dtaidistance import dtw_ndim
from numpy.lib.stride_tricks import sliding_window_view

from bt_studio.pipeline.metrics import calculate_hpo_score


def prepare_mcurves(panel_df: pl.DataFrame, tune_config: dict, common_config: dict) -> np.ndarray:
    """DataFrame (N, D, L) tensor and NaN boarder"""
    cross_days = int(tune_config["cross_days"])
    feature_cols = common_config.get("features", ["ofi_ratio", "volatility"]) 
    
    curves_list = []
    for feat in feature_cols:
        lag_cols = [f"lag_{feat}_{i}" for i in reversed(range(cross_days))]
        feat_matrix = np.hstack([np.vstack(panel_df[col].to_list()) for col in lag_cols])
        curves_list.append(feat_matrix)
    
    curves_md = np.array(curves_list) # Shape: (D, N, L)
    
    # lag_0 today eg 14:55  np.nan！
    execlude_bars = common_config.get("exclude_bars", 10) // int(tune_config["downsample"])
    if execlude_bars > 0:
        curves_md[:, :, -execlude_bars:] = np.nan
    
    return np.swapaxes(curves_md, 0, 1) # Shape: (N, D, L)


def get_candidate_motifs_md(T_multi: np.ndarray, config: dict, top_k=5):
    # Motif
    m = config["m"]
    mps, indices = stumpy.mstump(T_multi, m=m)
    distances = np.copy(mps[D - 1, :]) 
    distances[distances <= 1e-5] = np.inf
    
    candidate_motifs = []
    for _ in range(top_k):
        anchor = int(np.nanargmin(distances))
        if distances[anchor] > config["threshold_d"] or np.isinf(distances[anchor]): break
        candidate_motifs.append(T_multi[:, anchor : anchor + m])
        distances[max(0, anchor - m) : min(len(distances), anchor + m)] = np.inf
        
    if not candidate_motifs: 
        return {"status": "failed", "reason": "Not Found Motif", "metrics_score": 0.0}
        
    return candidate_motifs


def calc_min_subseq_dtw_md(row_md: np.ndarray, z_motif_t: np.ndarray, dtw_w: int, threshold_d: float):
    D, L = row_md.shape  # row_md shape: (D, L)
    m = z_motif_t.shape[0] # z_motif_t shape: (m, D) 
    
    if L < m:
        return np.inf

    # 1. vectorize windows shape : (D, window, m)
    windows = sliding_window_view(row_md, window_shape=m, axis=1)
    
    # 2. Transpose -> (window, D, m)
    windows_iter = np.transpose(windows, (1, 0, 2))
    
    # 3. nan mask / axis=(1,2) (D, m) 
    is_valid_window = ~np.isnan(windows_iter).any(axis=(1, 2))

    min_dist = np.inf
    for i in range(len(windows_iter)):
        if not is_valid_window[i]:
            continue
            
        # Shape (D, m)
        sub_md = windows_iter[i]
        
        means = np.mean(sub_md, axis=1, keepdims=True) # keepdims to broadcast
        stds = np.std(sub_md, axis=1, keepdims=True) + 1e-8
        
        z_sub_md = (sub_md - means) / stds
        
        # dtaidistance C (m, D) 
        z_sub_t = np.ascontiguousarray(z_sub_md.T, dtype=np.float64)
        
        d = dtw_ndim.distance_fast(z_sub_t, z_motif_t, window=dtw_w, max_dist=min(min_dist, threshold_d))
        
        if d < min_dist: 
            min_dist = d
    return min_dist


def evaluate_and_build_fsm_md(
    panel_df: pl.DataFrame, 
    curves_md: np.ndarray, # 💡 Shape: (N, D, L)
    motif_md: np.ndarray,  # 💡 Shape: (D, m)
    tune_config: dict,
    common_config: dict
) -> dict:

    base_score = 100.0

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

    # =======================================================================
    # 3. DTW Triggers
    # =======================================================================
    m = tune_config["m"]
    threshold_d = tune_config["threshold_d"]
    dtw_w = int(m * tune_config.get("dtw_window_frac", 0.1))

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
        return {"status": "failed", "reason": f"Matching Not enough (n={triggers.height})", "metrics_score": 0.0}

    # =======================================================================
    # 4. Markov Laplace 
    # =======================================================================

    trans_t1 = np.ones((3, 4), dtype=np.float64) 
    trans_t1_t2 = np.ones((4, 4), dtype=np.float64) 
    trans_t2_t3 = np.ones((4, 4), dtype=np.float64) 
    
    select_cols = ["macro_state"] + bin_cols
    valid_chain = triggers.drop_nulls(subset=select_cols)

    if valid_chain.height > 0:
        # for ms, b1, b2, b3 in valid_chain.select(select_cols).rows():
        #     trans_t1[ms, b1] += 1.0; trans_t1_t2[b1, b2] += 1.0; trans_t2_t3[b2, b3] += 1.0

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

    # =======================================================================
    # 5. Statistics Pval
    # =======================================================================
    cond_rets = triggers["fwd_ret_1"].drop_nulls().to_numpy()
    uncond_rets = eval_df["fwd_ret_1"].drop_nulls().to_numpy() 
    
    if len(cond_rets) < 5 or np.std(cond_rets) < 1e-8:
         return {"status": "failed", "reason": "ret Std 0 means supend or delist", "metrics_score": 0.0}
    
    try:
        u_stat, u_pval = stats.mannwhitneyu(cond_rets, uncond_rets, alternative=common_config["alternative"])
    except ValueError:
        return {"status": "failed", "reason": "MW-U 检验数学越界", "metrics_score": 0.0}

    # =======================================================================
    # 6. Final Score
    # =======================================================================
    score = calculate_hpo_score(u_pval, len(cond_rets), cond_rets, uncond_rets, tune_config, common_config)
    
    if score == 0.0:
        return {"status": "failed", "reason": f"(P-val={u_pval:.4f})", "metrics_score": 0.0}

    return {
        "status": "success",
        "fsm_network": {"P(T1|Macro)": trans_t1, "P(T2|T1)": trans_t1_t2, "P(T3|T2)": trans_t2_t3},
        "trigger_count": triggers.height,
        "learned_motif": motif.tolist(),
        "metrics_score": score,
        "u_pval": float(u_pval)
    }


def discover_fsm_pattern_md(
    panel_lf: pl.LazyFrame,  
    tune_config: dict, 
    common_config: dict
) -> Dict[str, Any]:

    m = int(search_config["motif_minutes"] // search_config["downsample"])
    threshold_d = float(np.sqrt(2 * m * (1.0 - search_config.get("threshold_r", 0.85))))
    tune_config.update({"m": m, "threshold_d": threshold_d})

    # =========================================================================
    # 1. Filter Panel DataFrame
    # =========================================================================
    panel_df = panel_lf.collect(engine="streaming")
    if panel_df.height == 0:
        return {
            "status": "failed", 
            "reason": "HPO Panel_df Zero after filter", 
            "metrics_score": 0.0
        }
        
    if panel_df.height <= m or m < 3:
        return {
            "status": "failed", 
            "reason": f"Not enough data (n={panel_df.height})", 
            "metrics_score": 0.0
        }
    
    # =========================================================================
    # 2. Features Matrix (N,D,L)
    # =========================================================================
    curves_md = prepare_mcurves(panel_df, tune_config, common_config)
    N, D, L = curves_md.shape
    
    # =========================================================================
    # 3. Volatility-Driven Sampling for stumpy
    # =========================================================================
    # curves_md Shape: (N, D, L) and axis=2 --> abs diff 
    curves_asbdiff_sum = np.nansum(np.abs(np.diff(curves_md, axis=2)), axis=2) # Shape -> (N, D)
    # Cross-sectional Z-score For features eg. OFI and Volatility
    _mean = np.nanmean(curves_asbdiff_sum, axis=0, keepdims=True)
    _std = np.nanstd(curves_asbdiff_sum, axis=0, keepdims=True) + 1e-8
    z_md = (curves_asbdiff_sum - _mean) / _std  # Shape -> (N, D)
    
    #  Sum ofi + vol --->  Shape(N,)
    mutation_scores = np.nansum(z_md, axis=1)
    
    theory_points = common_config.get("max_discovery_points", 20000)
    sample_size = min(N, max(5, int(theory_points / L))) # D * L 
    
    active_idx = np.argsort(mutation_scores)[-sample_size:]
    sampled_curves = curves_md[active_idx] # Shape: (Sample_N, D, L)

    # =========================================================================
    # 4. Nans between assets and Stumpy T_multi(D, Sample_N * L) For Candidates 
    # =========================================================================
    clean_curves = np.copy(sampled_curves)
    clean_curves[np.isinf(clean_curves)] = 0.0
    
    flat_dims = []
    for d in range(D):
        dim_data = clean_curves[:, d, :] 
        nan_buf = np.full((sample_size, m), np.nan) 
        dim_flat = np.hstack([dim_data, nan_buf]).flatten()[:-m]
        flat_dims.append(dim_flat)
    T_multi = np.vstack(flat_dims) # Shape: (D, Sample_N * L)

    candidate_motifs = get_candidate_motifs_md(T_multi, tune_config, top_k=5)
    
    # =========================================================================
    # 5. Evaluate Motif and Build FSM
    # =========================================================================
    best_result, highest_score = None, -1.0

    for motif_md in candidate_motifs:
        result = evaluate_and_build_fsm_md(
            panel_df, curves_md, motif_md, tune_config, common_config
        )
        if result["status"] == "success" and result["metrics_score"] > highest_score:
            highest_score, best_result = result["metrics_score"], result
            
    return best_result if best_result else {"status": "failed", "reason": "(P-val > 0.1)", "metrics_score": 0.0}
