import polars as pl
import numpy as np
import stumpy
import scipy.stats as stats

from dtaidistance import dtw
from typing import List, Dict, Any


def build_fsm_panel(all_feat_lf: list[pl.LazyFrame], daily_lf: pl.LazyFrame, config: dict) -> pl.DataFrame:
    # =========================================================================
    # config 
    # =========================================================================
    ds = config["downsample"]
    bars_per_day = 240 // ds

    # =========================================================================
    # schema align with aligned_lf
    # =========================================================================
    daily_schema = daily_lf.collect_schema()

    if daily_schema["day"] in [pl.Int32, pl.Int64]:
        daily_lf = daily_lf.with_columns(pl.col("day").cast(pl.String).str.to_date("%Y%m%d"))
    elif daily_schema["day"] == pl.String:
        daily_lf = daily_lf.with_columns(pl.col("day").str.to_date("%Y%m%d"))
    elif daily_schema["day"] == pl.Datetime:
        daily_lf = daily_lf.with_columns(pl.col("day").cast(pl.Date))

    daily_lf = daily_lf.with_columns([
        pl.col("sid").cast(pl.String).str.strip_chars(" \x00\t\n"), 
        pl.col("day").cast(pl.Date)
    ])
    
    # =========================================================================
    # schema align with aligned_lf
    # =========================================================================

    # all_feat_lf = pl.concat(aligned_lfs)
    feat_schema = all_feat_lf.collect_schema()
    
    if feat_schema["day"] in [pl.Int32, pl.Int64]:
        all_feat_lf = all_feat_lf.with_columns(pl.col("day").cast(pl.String).str.to_date("%Y%m%d"))
    elif feat_schema["day"] == pl.String:
        all_feat_lf = all_feat_lf.with_columns(pl.col("day").str.to_date("%Y%m%d"))
    elif feat_schema["day"] == pl.Datetime:
        all_feat_lf = all_feat_lf.with_columns(pl.col("day").cast(pl.Date))

    all_feat_lf = all_feat_lf.with_columns([
        pl.col("sid").cast(pl.String).str.strip_chars(" \x00\t\n"), 
        pl.col("day").cast(pl.Date)
    ])

    # =========================================================================
    # daily_ret and vol
    # =========================================================================
    daily_ret_lf = (
        daily_lf.sort(["sid", "day"])
        .with_columns([
            (pl.col("close") / pl.col("close").shift(1).over("sid") - 1.0).alias("daily_ret")
        ])
        .with_columns([
            # pl.col("daily_ret").rolling_std(window_size=20, min_periods=5)
            #   .over("sid").fill_null(strategy="forward")
            #   .clip(lower_bound=0.005).alias("vol_20d"),
            (pl.col("close").shift(-1).over("sid") / pl.col("close") - 1.0).alias("fwd_ret_1"),
            (pl.col("close").shift(-2).over("sid") / pl.col("close") - 1.0).alias("fwd_ret_2"),
            (pl.col("close").shift(-3).over("sid") / pl.col("close") - 1.0).alias("fwd_ret_3")
        ])
    )
 
    # =========================================================================
    # downsample
    # =========================================================================
    if ds > 1:
        all_feat_lf = all_feat_lf.filter((pl.col("bar_idx") % ds) == 0)

    # =========================================================================
    # join 
    # =========================================================================
    curve_lf = (
        all_feat_lf
        .sort(["day", "sid", "bar_idx"])
        .group_by(["day", "sid"])
        .agg([
            pl.col("ofi_ratio").alias("daily_curve"),
            pl.len().alias("curve_len")  
        ])
        .filter(pl.col("curve_len") == bars_per_day) 
    )

    # =========================================================================
    # crossover concat 
    # =========================================================================
    shift_exprs = [
        pl.col("daily_curve").shift(i).over("sid").alias(f"lag_{i}") 
        for i in reversed(range(config["cross_days"]))
    ]

    curve_lf = (
        curve_lf.sort(["sid", "day"])
        .with_columns(shift_exprs)
        .drop_nulls(subset=[f"lag_{i}" for i in range(config["cross_days"])]) 
    )

    panel_lf = curve_lf.join(
        # daily_ret_lf.select(["day", "sid", "vol_20d", "fwd_ret_1", "fwd_ret_2", "fwd_ret_3"]),
        daily_ret_lf.select(["day", "sid", "fwd_ret_1", "fwd_ret_2", "fwd_ret_3"]),
        on=["day", "sid"], how="inner"
    )
    return panel_lf


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


# def get_candidate_mstump_motifs(raw_array_2d: np.ndarray, config: dict, top_k: int = 5):
#     # raw_array_2d shape: (2, N) -> 维度0: OFI, 维度1: VOL
#     m = config["m"]
    
#     # stumpy.mstump 返回两个核心矩阵：
#     # mps: 形状 (d, N-m+1), mps[0]是1D Motif, mps[1]是2D Motif 距离
#     # indices: 对应的最近邻索引
#     mps, indices = stumpy.mstump(raw_array_2d, m=m)
    
#     # 我们需要多维完全匹配的特征，所以取 d-1 (即 index 1)
#     distances = np.copy(mps[1, :]) 
    
#     zero_mask = distances <= 1e-5
#     distances[zero_mask] = np.inf

#     candidates = []
#     for _ in range(top_k):
#         anchor_idx = int(np.nanargmin(distances))
#         v_d = distances[anchor_idx]
        
#         if v_d > config["threshold_d"] or np.isinf(v_d):
#             break
            
#         # 截取二维候选 Motif，形状为 (2, m)
#         candidate_motif = raw_array_2d[:, anchor_idx : anchor_idx + m]
#         candidates.append(candidate_motif)
        
#         exclude_start = max(0, anchor_idx - m)
#         exclude_end = min(len(distances), anchor_idx + m)
#         distances[exclude_start:exclude_end] = np.inf
        
#     return candidates


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

        # skip if np.nan 
        if np.isnan(sub_seq).any():
            continue
        
        std = np.std(sub_seq) + 1e-8
        z_sub = (sub_seq - np.mean(sub_seq)) / std

        z_sub = np.ascontiguousarray(z_sub, dtype=np.float64)
        
        current_limit = min(min_dist, threshold_d) 
        d = dtw.distance_fast(z_sub, z_motif, window=dtw_w, max_dist=current_limit)
        
        if d < min_dist:
            min_dist = d
    return min_dist


# def calc_min_subseq_dtw_ndim(row_curve_2d: np.ndarray, z_motif_2d: np.ndarray, dtw_w: int):
#     # row_curve_2d shape: (N, 2), dtaidistance 需要 (N, d)
#     # Z-Score 标准化必须在多维上【独立】进行，防止量纲冲突
#     # 然后直接调用 dtaidistance 的多维 DTW
#     return dtw_ndim.distance_fast(row_curve_2d, z_motif_2d, window=dtw_w)


def evaluate_and_build_fsm(
    panel_df: pl.DataFrame, 
    curves_2d: np.ndarray,
    motif: np.ndarray, 
    tune_config: dict,
    common_config: dict
    ) -> dict:

    base_score = 100.0

    if panel_df["sid"].dtype != pl.Binary:
        panel_df = panel_df.with_columns(pl.col("sid").cast(pl.Binary))
    
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
        .with_columns(pl.col("macro_state").shift(1)) # avoid lookahead bias
        .drop(["p33", "p67"])
        .sort("day")
    )

    daily_macro = daily_macro_lf.collect()
    
    eval_df = panel_df.join(daily_macro.select(["day", "macro_state"]), on="day", how="left")

   # =================================================================
    # 2. Time-Adjusted Zero-Anchored Bins Based on std
    # =================================================================
    # z_bound = tune_config["z_abs_bound"] # 默认为 1.0

    # for fw in common_config["stats_windows"]:
    #     col = f"fwd_ret_{fw}"
    #     if col not in panel_df.columns: continue
            
    #     panel_df = panel_df.with_columns([
    #         # T +1 / T+2 / T+3 std ---> sqrt(T)
    #         (pl.col(col) / (pl.col("vol_20d") * np.sqrt(fw))).alias(f"z_abs_{fw}")
    #     ]).with_columns([
    #         pl.when(pl.col(f"z_abs_{fw}") < -z_bound).then(0)
    #         .when(pl.col(f"z_abs_{fw}") < 0.0).then(1)
    #         .when(pl.col(f"z_abs_{fw}") < z_bound).then(2)
    #         .otherwise(3).cast(pl.Int32).alias(f"bin_{fw}")
    #     ])

    edge_ratio = common_config["edge_ratio"]

    for fw in common_config["stats_windows"]:
        col = f"fwd_ret_{fw}"
        if col not in panel_df.columns: continue
        
        panel_df = panel_df.with_columns([
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
    dtw_w = int(m * tune_config.get("dtw_window_frac", 0.1))

    # curves = np.vstack(eval_df["curve"].to_list()).astype(np.float64)
    z_motif = np.ascontiguousarray((motif - np.mean(motif)) / (np.std(motif) + 1e-8), dtype=np.float64)

    distances = [
        calc_min_subseq_dtw(curve, z_motif, m, dtw_w, threshold_d) 
        for curve in curves_2d
    ]
    
    eval_df = eval_df.with_columns(pl.Series("distance", distances))
    triggers = eval_df.filter(pl.col("distance") <= threshold_d)
    
    if triggers.height < 5:
        return {"status": "failed", "reason": f"Matching Not enough (n={triggers.height})", "metrics_score": 0.0}

    # === Markov Laplace ===
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

    # ===  two-sided or greater ===
    cond_rets = triggers["fwd_ret_1"].drop_nulls().to_numpy()
    uncond_rets = eval_df["fwd_ret_1"].drop_nulls().to_numpy() 
    
    if len(cond_rets) < 5 or np.std(cond_rets) < 1e-8:
         return {"status": "failed", "reason": "ret Std 0 means supend or delist", "metrics_score": 0.0}
    
    try:
        u_stat, u_pval = stats.mannwhitneyu(cond_rets, uncond_rets, alternative=common_config["alternative"])
    except ValueError:
        return {"status": "failed", "reason": "MW-U 检验数学越界", "metrics_score": 0.0}
    
    score = 100.0 if u_pval <= 0.05 else (10.0 if u_pval <= 0.10 else 0.0)
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


def prepare_stumpy_array(curves_2d: np.ndarray, config: dict) -> np.ndarray:
    if curves_2d.size == 0:
        return np.array([])
        
    m = config["m"]
    clean_curves = np.copy(curves_2d)
    # clean_curves = np.nan_to_num(curves_2d, nan=0.0, posinf=0.0, neginf=0.0)
    clean_curves[np.isinf(clean_curves)] = 0.0 
    
    # m NaN as separate between assets
    nan_buffer = np.full((clean_curves.shape[0], m), np.nan)
    stumpy_arr = np.hstack([clean_curves, nan_buffer]).flatten()
    # abundan last m np.nan
    return stumpy_arr[:-m] 


def discover_fsm_pattern(
    search_config: dict, 
    panel_lf: pl.LazyFrame,  
    common_config: dict
) -> Dict[str, Any]:
    
    # config
    cross_days = int(search_config["cross_days"])
    m = int(search_config["motif_minutes"] // search_config["downsample"])
    threshold_d = float(np.sqrt(2 * m * (1.0 - search_config.get("threshold_r", 0.85))))
    
    tune_config = search_config.copy()
    tune_config["m"] = m
    tune_config["threshold_d"] = threshold_d
    
    # extract curves_2d
    panel_df = panel_lf.collect(engine="streaming")

    if panel_df.height == 0:
        return {
            "status": "failed", 
            "reason": "HPO Panel_df Zero after filter", 
            "metrics_score": 0.0
        }
        
    if panel_df.height <= m :
        return {
            "status": "failed", 
            "reason": f"Not enough data (n={panel_df.height})", 
            "metrics_score": 0.0
        }

    lag_cols = [f"lag_{i}" for i in reversed(range(cross_days))]
    lag_arrays = [np.vstack(panel_df[col].to_list()) for col in lag_cols]
    lag_arrays = [np.vstack(panel_df[col].to_list()) for col in lag_cols]
    
    # lag_0 today 14:55  np.nan！
    tail_bars = common_config.get("exclude_bars", 10) // int(search_config["downsample"])
    if tail_bars > 0:
        lag_arrays[-1][:, -tail_bars:] = np.nan

    curves_2d = np.hstack(lag_arrays) # Shape: (N, cross_days * bars_per_day)
    
    # extract stumpy feature
    stumpy_1d_array = prepare_stumpy_array(curves_2d, tune_config)
    candidate_motifs = get_candidate_motifs(stumpy_1d_array, tune_config, top_k=5)
    
    if not candidate_motifs: 
        return {"status": "failed", "reason": " Not Found Motif"}
    
    best_result, highest_score = None, -1.0
    for motif in candidate_motifs:

        result = evaluate_and_build_fsm(
            panel_df, curves_2d, motif, tune_config, common_config)
        if result["status"] == "success" and result["metrics_score"] > highest_score:
            highest_score = result["metrics_score"]
            best_result = result
            
    return best_result if best_result else {"status": "failed", "reason": "(P-val > 0.1)"}
