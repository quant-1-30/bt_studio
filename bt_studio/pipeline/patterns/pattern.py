import polars as pl
import numpy as np

from typing import List, Dict, Any

from .astc import prepare_curves, get_candidate_motifs
from .fsm import evaluate_and_build_fsm


def get_balanced_samples(curves: np.ndarray, max_points: int = 20000) -> np.ndarray:
    """
        no cross_days 50% + 50%
    
    :param curves: Shape (N, L) 1D /  (N, D, L) 
    :param max_points: Stumpy 
    """
    N = curves.shape[0]
    if N == 0:
        return curves

    # =========================================================================
    # Points Per Stock
    # =========================================================================
    if curves.ndim == 2:  # 1D: (N, L)
        points_per_stock = curves.shape[1]
    elif curves.ndim == 3:  # MD: (N, D, L)
        points_per_stock = curves.shape[1] * curves.shape[2]
    else:
        raise ValueError(f"Unsupported curves shape: {curves.shape}, expected 2D or 3D array.")

    sample_size = min(N, max(5, int(max_points / points_per_stock)))

    if N <= sample_size:
        return curves

    # =========================================================================
    # Mutation Score / Active Rank
    # =========================================================================
    if curves.ndim == 2:
        mutation_scores = np.nansum(np.abs(np.diff(curves, axis=1)), axis=1)
    else:
        diff_sum = np.nansum(np.abs(np.diff(curves, axis=2)), axis=2)  # Shape: (N, D)
        _mean = np.nanmean(diff_sum, axis=0, keepdims=True)
        _std = np.nanstd(diff_sum, axis=0, keepdims=True)
        _std = np.where(_std < 1e-8, 1e-8, _std)  # 防 0 划分
        
        z_md = (diff_sum - _mean) / _std  # Shape: (N, D)
        mutation_scores = np.nansum(z_md, axis=1)  # Shape: (N,)

    mutation_scores = np.nan_to_num(mutation_scores, nan=0.0)

    # =========================================================================
    # 50% Top + 50% Random
    # =========================================================================
    half_size = sample_size // 2
    sorted_idx = np.argsort(mutation_scores)

    top_active_idx = sorted_idx[-half_size:]

    remaining_idx = sorted_idx[:-half_size]
    
    random_size = min(sample_size - half_size, len(remaining_idx))
    random_idx = np.random.choice(remaining_idx, size=random_size, replace=False)

    final_sample_idx = np.sort(np.concatenate([top_active_idx, random_idx]))
    return curves[final_sample_idx]


def discover_fsm_pattern(
    panel_lf: pl.LazyFrame,  
    tune_config: dict, 
    common_config: dict
) -> Dict[str, Any]: 

    m = int(tune_config["motif_minutes"] // tune_config["downsample"])
    threshold_d = float(np.sqrt(2 * m * (1.0 - tune_config["threshold_r"])))

    # =========================================================================
    # 1. Filter Panel DataFrame
    # =========================================================================
    panel_df = panel_lf.collect(engine="streaming")
    if panel_df.height == 0:
        return {
            "status": "failed", 
            "reason": f"Panel_df height 0", 
            "metrics_score": -100.0
        }

    # =========================================================================
    # 2. Features Matrix (N,L)
    # =========================================================================
    tune_config["m"] = m
    tune_config["threshold_d"] = threshold_d

    curves_2d = prepare_curves(panel_df, tune_config, common_config)

    if curves_2d.size == 0:
        return {
            "status": "failed", 
            "reason": "Curves_2d Empty", 
            "metrics_score": -100.0
        }

    # =========================================================================
    # 3. Volatility-Driven Sampling for stumpy
    # =========================================================================
    theory_points = common_config.get("max_points", 20000) 
    sampled_curves = get_balanced_samples(curves_2d, max_points=theory_points) 
    
    # =========================================================================
    # 4. Nans between assets and Stumpy T_multi(Sample_N * L) For Candidates 
    # =========================================================================
    clean_curves = np.copy(sampled_curves)
    clean_curves[np.isinf(clean_curves)] = 0.0 # np.nan_to_num(sampled_curves, nan=0.0, posinf=0.0, neginf=0.0) 
    # m NaN as separate between assets
    nan_buffer = np.full((clean_curves.shape[0], m), np.nan)
    stumpy_1d_array = np.hstack([clean_curves, nan_buffer]).flatten()[:-m] # abundan last m np.nan

    candidate_motifs = get_candidate_motifs(stumpy_1d_array, tune_config, common_config)
    
    if not candidate_motifs: 
        return {
            "status": "failed", 
            "reason": "Not Found Motif", 
            "metrics_score": -100.0
        }
    
    best_result, highest_score = None, -100.0
    eps = common_config["eps"]

    for motif in candidate_motifs:

        if np.nanstd(motif) < eps: 
            continue

        result = evaluate_and_build_fsm(
            panel_df, curves_2d, motif, tune_config, common_config)

        if result["status"] == "success" and result["metrics_score"] > highest_score:
            highest_score = result["metrics_score"]
            best_result = result
            
    return best_result if best_result else {"status": "failed", "reason": "(P-val > 0.1)", "metrics_score": -100.0}
