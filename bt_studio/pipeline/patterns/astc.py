import polars as pl
import numpy as np
import stumpy

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="stumpy")

from dtaidistance import dtw
from typing import List, Dict, Any
from numpy.lib.stride_tricks import sliding_window_view


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

