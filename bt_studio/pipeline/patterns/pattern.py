import polars as pl
import numpy as np

from typing import List, Dict, Any

from .astc import prepare_curves, get_candidate_motifs
from .fsm import evaluate_and_build_fsm


def get_balanced_samples(curves: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if curves.ndim != 2 or curves.size == 0:
        return np.empty((0, curves.shape[1] if curves.ndim == 2 else 0))

    N, L = curves.shape
    sample_size = min(N, max(5, int(max_points / L)))

    valid_mask = ~np.isnan(curves).all(axis=1)
    valid_curves = curves[valid_mask]

    if len(valid_curves) <= sample_size:
        return valid_curves

    # [FIX P2-P6] Use a local RNG with fixed seed for reproducible HPO.
    # Without this, the same tune_config produces different scores across
    # trials due to random sampling, adding noise to Optuna TPE optimization.
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(valid_curves), size=sample_size, replace=False)
    return valid_curves[idx]


def discover_fsm_pattern(
    panel_lf: pl.LazyFrame,
    tune_config: dict,
    common_config: dict
) -> Dict[str, Any]:

    m = int(tune_config["motif_minutes"] // tune_config["downsample"])
    threshold_d = float(np.sqrt(2 * m * (1.0 - tune_config["threshold_r"])))
    random_dist = float(np.sqrt(2 * m))

    # =========================================================================
    # 1. Drop DataFrame lag_0 null
    # =========================================================================
    try:
        panel_df = panel_lf.collect(engine="streaming")
    except Exception:
        panel_df = panel_lf if isinstance(panel_lf, pl.DataFrame) else panel_lf.collect()

    panel_df = panel_df.filter(pl.col("lag_0").is_not_null())
    if panel_df.height <= m or m < 3:
        return {
            "status": "failed",
            "reason": f"Data not enough after mask (n={panel_df.height})",
            "metrics_score": -500.0
        }

    # =========================================================================
    # Tensor Matrix
    # =========================================================================
    tune_config["m"] = m
    tune_config["threshold_d"] = threshold_d

    curves_2d = prepare_curves(panel_df, tune_config, common_config)

    if curves_2d.size == 0:
        return {
            "status": "failed",
            "reason": "Curves_2d Empty",
            "metrics_score": -500.0
        }

    # =========================================================================
    # 3. Balanced Sample
    # =========================================================================
    theory_points = common_config.get("max_points", 20000)
    sampled_curves = get_balanced_samples(curves_2d, max_points=theory_points, seed=common_config.get("seed", 42))

    # =========================================================================
    # 4. Padding and NaN
    # =========================================================================
    clean_curves = np.copy(sampled_curves)
    clean_curves[np.isinf(clean_curves)] = 0.0

    nan_buffer = np.full((clean_curves.shape[0], m), np.nan)
    stumpy_1d_array = np.hstack([clean_curves, nan_buffer]).flatten()[:-m]

    candidate_motifs = get_candidate_motifs(stumpy_1d_array, tune_config, common_config)
    if not candidate_motifs:
        return {
            "status": "failed",
            "reason": "STUMPY returned no valid motifs",
            "metrics_score": -500.0
        }

    # =========================================================================
    # 5. Estimate
    # =========================================================================
    best_result, highest_score = None, -500.0

    for motif in candidate_motifs:
        if np.nanstd(motif) < 1e-4:
            continue

        result = evaluate_and_build_fsm(
            panel_df, curves_2d, motif, tune_config, common_config)

        if result["status"] == "success" and result["metrics_score"] > highest_score:
            highest_score = result["metrics_score"]
            best_result = result

    return best_result if best_result else {
        "status": "failed",
        "reason": "All candidates failed statistical tests",
        "metrics_score": -500.0
    }