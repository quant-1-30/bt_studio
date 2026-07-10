import polars as pl
import numpy as np

from typing import List, Dict, Any

from .mastc import prepare_mcurves, get_candidate_motifs_md
from .mfsm import evaluate_and_build_fsm_md


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


