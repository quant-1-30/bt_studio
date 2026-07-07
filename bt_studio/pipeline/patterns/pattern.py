import polars as pl
import numpy as np

from typing import List, Dict, Any

from .astc import prepare_curves, get_candidate_motifs
from .m_astc import prepare_mcurves, get_candidate_motifs_md
from .fsm import evaluate_and_build_fsm, evaluate_and_build_fsm_md


def get_balanced_samples(curves: np.ndarray, max_points: int = 20000) -> np.ndarray:
    """unify 1D and MD / balance 50% avoid distortion
    
    :param curves: Shape (N, L) / (N, D, L) 
    :param max_points: Stumpy one core process maxlength
    """
    N = curves.shape[0]
    
    # =========================================================================
    # Sample Size
    # =========================================================================
    if curves.ndim == 2: # 1D Shape (N, L)
        points_per_stock = curves.shape[1]
    else: # MD Shape (N, D, L)
        points_per_stock = curves.shape[1] * curves.shape[2] 
        
    # D * L
    sample_size = min(N, max(5, int(max_points / points_per_stock)))
    
    if N <= sample_size:
        return curves

    # =========================================================================
    # Mutation Score
    # =========================================================================
    if curves.ndim == 2: # 1D
        mutation_scores = np.nansum(np.abs(np.diff(curves, axis=1)), axis=1)
    else:
        # MD all feature scale to z-score / curves_md Shape: (N, D, L) and axis=2 --> abs diff 
        diff_sum = np.nansum(np.abs(np.diff(curves, axis=2)), axis=2) # Shape: (N, D)
        _mean = np.nanmean(diff_sum, axis=0, keepdims=True)
        _std = np.nanstd(diff_sum, axis=0, keepdims=True) + 1e-8
        z_md = (diff_sum - _mean) / _std # Shape: (N, D)
        mutation_scores = np.nansum(z_md, axis=1) # Shape: (N,)

    # =========================================================================
    # Balanced Sampling: 50% active  + 50% random
    # =========================================================================
    half_size = sample_size // 2

    # 1. sort by score and return idx
    sorted_idx = np.argsort(mutation_scores)
    
    # 2. Top Half 
    top_active_idx = sorted_idx[-half_size:]
    
    # 3. Random
    remaining_idx = sorted_idx[:-half_size] # avoid np.setdiff1d(np.arange(N), top_active_idx)
    
    random_size = sample_size - half_size
    random_idx = np.random.choice(remaining_idx, size=random_size, replace=False) # Sampling without replacement
    
    final_sample_idx = np.sort(np.concatenate([top_active_idx, random_idx]))
    return curves[final_sample_idx]


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
            "metrics_score": -9999.0
        }
        
    if panel_df.height <= m or m < 3:
        return {
            "status": "failed", 
            "reason": f"Not enough data (n={panel_df.height})", 
            "metrics_score": -9999.0
        }
    
    # =========================================================================
    # 2. Features Matrix (N,D,L)
    # =========================================================================
    curves_md = prepare_mcurves(panel_df, tune_config, common_config)
    N, D, L = curves_md.shape
    
    # =========================================================================
    # 3. Volatility-Driven Sampling for stumpy
    # =========================================================================
    theory_points = common_config.get("max_discovery_points", 20000)
    sampled_curves = get_balanced_samples(curves_md, max_points=theory_points) 

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

    candidate_motifs = get_candidate_motifs_md(T_multi, tune_config, dimension=D, top_k=5)
    
    # =========================================================================
    # 5. Evaluate Motif and Build FSM
    # =========================================================================
    best_result, highest_score = None, -9999.0

    for motif_md in candidate_motifs:
        result = evaluate_and_build_fsm_md(
            panel_df, curves_md, motif_md, tune_config, common_config
        )
        if result["status"] == "success" and result["metrics_score"] > highest_score:
            highest_score, best_result = result["metrics_score"], result
            
    return best_result if best_result else {"status": "failed", "reason": "(P-val > 0.1)", "metrics_score": -9999.0}
