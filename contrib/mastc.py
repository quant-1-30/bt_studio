import stumpy
import numpy as np
import polars as pl

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="stumpy")

from dtaidistance import dtw_ndim
from typing import List, Dict, Any
from numpy.lib.stride_tricks import sliding_window_view


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


def get_candidate_motifs_md(T_multi: np.ndarray, config: dict, dimension: int, top_k=5):
    # Motif
    m = config["m"]
    mps, indices = stumpy.mstump(T_multi, m=m)

    distances = np.copy(mps[dimension - 1, :]).astype(np.float64) 
    distances[distances <= 1e-5] = np.inf
    
    if np.all(np.isinf(distances)):
        return []

    candidate_motifs = []
    for _ in range(top_k):
        anchor = int(np.nanargmin(distances))
        if distances[anchor] > config["threshold_d"] or np.isinf(distances[anchor]): break
        candidate_motifs.append(T_multi[:, anchor : anchor + m])
        distances[max(0, anchor - m) : min(len(distances), anchor + m)] = np.inf
        
    if not candidate_motifs: 
        return {"status": "failed", "reason": "Not Found Motif", "metrics_score": -100.0}
        
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




    # # cross_days
    # actual_cross_days = max(1, tune_config.get("cross_days", 1))
    
    # curve_lf = curve_lf.join(calendar_lf, on="day", how="left").join(
    #     daily_lf.select(["day", "sid", "regime_signal"]), on=["day", "sid"], how="left"
    # ).sort(["sid", "day"])

    # # trading_days continual
    # curve_lf = (
    #     curve_lf
    #     .with_columns((pl.col("trade_day_idx") - pl.col("trade_day_idx").shift(1).over("sid")).alias("day_diff"))
    #     .with_columns(
    #         (
    #             # 999 ensure skip
    #             (pl.col("day_diff").fill_null(999).rolling_max(window_size=actual_cross_days, min_periods=actual_cross_days).over("sid") == 1) &
    #             (pl.col("regime_signal") == 1)
    #         ).alias("is_valid_sequence")
    #     )
    # )

    # shift_exprs = [
    #     pl.when(pl.col("is_valid_sequence"))
    #     .then(pl.col("daily_curve").shift(i).over("sid") if i > 0 else pl.col("daily_curve"))
    #     .otherwise(None)
    #     .alias(f"lag_{i}")
    #     for i in reversed(range(actual_cross_days))
    # ]
    # curve_lf = curve_lf.with_columns(shift_exprs).drop(["trade_day_idx", "day_diff", "regime_signal"])

