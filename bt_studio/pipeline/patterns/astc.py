import polars as pl
import numpy as np
import stumpy

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="stumpy")

from dtaidistance import dtw
from typing import List, Dict, Any
from numpy.lib.stride_tricks import sliding_window_view


def prepare_curves(panel_df: pl.DataFrame, tune_config: dict, common_config: dict) -> np.ndarray:
    """
    DataFrame to (N, L) 2D tensor with Lookahead prevention and NaN Masking.
    
    Returns
    -------
    np.ndarray: Shape (N, L), where N is number of stock samples, L is bars_per_day.
    """
    if panel_df.height == 0:
        return np.array([], dtype=np.float64)

    bars_per_day = 240 // int(tune_config["downsample"])

    nan_pad = np.full(bars_per_day, np.nan, dtype=np.float64)
    raw_list = panel_df["lag_0"].to_list()
    filled_list = [nan_pad if x is None else x for x in raw_list]
    
    curves_2d = np.vstack(filled_list)  # Shape: (N, bars_per_day)

    exclude_bars = common_config.get("exclude_bars", 0) // int(tune_config["downsample"])
    if exclude_bars > 0:
        curves_2d[:, -exclude_bars:] = np.nan  

    return curves_2d


def get_candidate_motifs(raw_array: np.ndarray, config: dict, common_config: dict) -> List[np.ndarray]:
    m = config["m"]
    if raw_array.size < m or m < 3:
        return []

    threshold_d = config["threshold_d"]
    # # 理论随机距离: 两个不相关 z-norm 序列的期望欧氏距离
    # random_dist = float(np.sqrt(2 * m))
    
    mp = stumpy.stump(raw_array, m=m)
    distances = np.ascontiguousarray(mp[:, 0], dtype=np.float64)
    
    shape = (raw_array.size - m + 1, m)
    strides = (raw_array.strides[0], raw_array.strides[0])
    windows = np.lib.stride_tricks.as_strided(raw_array, shape=shape, strides=strides)
    
    has_nan = np.any(np.isnan(windows), axis=1)
    
    with np.errstate(invalid='ignore'):
        std_vals = np.std(windows, axis=1)
        is_even = std_vals < common_config["eps"]
        
    bad_mask = has_nan | is_even
    distances[bad_mask[:distances.size]] = np.inf
    distances[np.isnan(distances) | np.isinf(distances)] = np.inf

    candidates = []
    
    for _ in range(common_config["topk"]):
        anchor_idx = int(np.argmin(distances)) 
        v_d = distances[anchor_idx]
        
        if v_d > threshold_d or np.isinf(v_d):
            break
            
        candidates.append(raw_array[anchor_idx : anchor_idx + m])
        
        # --- Exclusion Zone ---
        exclude_start = max(0, anchor_idx - m)
        exclude_end = min(distances.size, anchor_idx + m)
        distances[exclude_start:exclude_end] = np.inf

    print(f"[DEBUG astc] candidates found: {len(candidates)}")
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

    # 1. Shape: (window, motif_len) and Zero_copy
    windows = sliding_window_view(row_curve, window_shape=motif_len)
    
    # 2. mask
    is_valid_window = ~np.isnan(windows).any(axis=-1)
    valid_windows = windows[is_valid_window]
    
    if len(valid_windows) == 0:
        return np.inf
        
    # 3. Z-Score Vectorize
    means = np.mean(valid_windows, axis=1, keepdims=True)
    stds = np.std(valid_windows, axis=1, keepdims=True) + 1e-8
    z_windows = (valid_windows - means) / stds
    
    min_dist = np.inf
    
    for i in range(len(z_windows)):
        z_sub = np.ascontiguousarray(z_windows[i], dtype=np.float64)

        d = dtw.distance_fast(
            z_sub, 
            z_motif, 
            window=dtw_w, 
            max_dist=min(min_dist, threshold_d)
        )
        if d < min_dist:
            min_dist = d
            if min_dist <= 1e-6:
                return float(min_dist)

    return float(min_dist)