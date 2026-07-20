import polars as pl
import numpy as np
import stumpy

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="stumpy")

from dtaidistance import dtw
from typing import List, Dict, Any
from numpy.lib.stride_tricks import sliding_window_view


def prepare_curves(panel_df: pl.DataFrame, tune_config: dict, common_config: dict) -> np.ndarray:
    """DataFrame to (N, L) tensor with Lookahead prevention and NaN Masking"""
    cross_days = int(tune_config["cross_days"])
    bars_per_day = 240 // int(tune_config["downsample"])

    lag_cols = [f"lag_{i}" for i in reversed(range(cross_days))]

    # np.nan list --> None 
    nan_pad = np.full(bars_per_day, np.nan, dtype=np.float64)
    
    lag_arrays = []
    for col in lag_cols:
        # Polars  None (Null) ---> nan_pad
        raw_list = panel_df[col].to_list()
        filled_list = [nan_pad if x is None else x for x in raw_list]
        lag_arrays.append(np.vstack(filled_list))
        
    execlude_bars = common_config["exclude_bars"] // int(tune_config["downsample"])
    if execlude_bars > 0:
        lag_arrays[-1][:, -execlude_bars:] = np.nan

    curves_2d = np.hstack(lag_arrays) # Shape: (N, cross_days * bars_per_day)
    return curves_2d    


def get_candidate_motifs(raw_array: np.ndarray, config: dict, common_config: dict) -> List[np.ndarray]:
    m = config["m"]
    if raw_array.size < m or m < 3:
        return []

    threshold_d = config["threshold_d"]
    
    try:
        mp = stumpy.stump(raw_array, m=m)
    except Exception:
        return []

    distances = np.ascontiguousarray(mp[:, 0], dtype=np.float64)
    
    shape = (raw_array.size - m + 1, m)
    strides = (raw_array.strides[0], raw_array.strides[0])
    windows = np.lib.stride_tricks.as_strided(raw_array, shape=shape, strides=strides) # row --> next row 8byte
    
    has_nan = np.any(np.isnan(windows), axis=1)
    
    # np.std  ---> np.nanstd and np.errstate supress NaN RuntimeWarning
    with np.errstate(invalid='ignore'):
        std_vals = np.std(windows, axis=1)
        is_even = std_vals < common_config["eps"]
        
    bad_mask = has_nan | is_even
    
    distances[bad_mask[:distances.size]] = np.inf
    distances[np.isnan(distances) | np.isinf(distances)] = np.inf

    if np.all(np.isinf(distances)):
        return []

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
        z_sub = np.ascontiguousarray(z_windows[i], dtype=np.float64) # row contiguous

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
