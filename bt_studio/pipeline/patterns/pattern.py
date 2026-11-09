#! /usr/bin/env python3
"""Supervised positive-pool motif discovery and full-panel causal FSM evaluation."""

import logging
from typing import Any, Dict, Optional

import numpy as np
import polars as pl

from .astc import get_candidate_motifs, prepare_curves
from .fsm import evaluate_and_build_fsm, prepare_fsm_base_df
from bt_studio.utils.io import collect_lazy_safe
from bt_studio.utils.math_func import resolve_search_hyperparams

logger = logging.getLogger(__name__)


def _select_positive_motif_curves(
    curves_2d: np.ndarray,
    base_df: pl.DataFrame,
    m: int,
    common_config: dict,
) -> np.ndarray:
    """Select training curves from directional T+1 outperformers (with T+1 intraday continuation guard)."""
    topk = max(1, int(common_config.get("topk", 5)))
    max_curves = max(m + 1, topk * 30)

    raw_traj = base_df["raw_trajectory"].fill_null(np.nan).to_numpy()
    rank_traj = base_df["rank_trajectory"].fill_null(0.5).to_numpy()
    mom_traj = (
        base_df["momentum_trajectory"].fill_null(0.0).to_numpy()
        if "momentum_trajectory" in base_df.columns
        else np.zeros_like(raw_traj)
    )
    pos_quantile = float(common_config.get("positive_motif_quantile", 0.70))

    sign = -1.0 if common_config.get("alternative", "greater") == "less" else 1.0
    dir_raw = sign * raw_traj
    dir_mom = sign * mom_traj
    dir_rank = (1.0 - rank_traj) if sign < 0 else rank_traj

    # A-share T+1 continuation filter: require cross-sectional top quantile, positive holding return,
    # and non-negative T+1 open-to-exit continuation (with graceful fallback on tiny samples).
    base_pos_mask = np.isfinite(dir_raw) & (dir_rank >= pos_quantile) & (dir_raw > 0.0)
    pos_idx = np.empty(0, dtype=np.int64)
    for tier_idx, tier_mask in enumerate((base_pos_mask & (dir_mom >= 0.0), base_pos_mask, np.isfinite(dir_raw))):
        pos_idx = np.flatnonzero(tier_mask)
        if len(pos_idx) > m:
            if tier_idx > 0:
                logger.debug("[pattern] Positive pool relaxed to tier %d (n=%d, m=%d)", tier_idx, len(pos_idx), m)
            break
    if len(pos_idx) == 0:
        return curves_2d[:max_curves]

    if len(pos_idx) > max_curves:
        top_order = np.argsort(-dir_raw[pos_idx], kind="stable")[:max_curves]
        pos_idx = np.sort(pos_idx[top_order])

    return curves_2d[pos_idx]


def discover_fsm_pattern(
    panel_lf: pl.LazyFrame,
    tune_config: dict,
    common_config: dict,
) -> Dict[str, Any]:
    # 1. Derive m and threshold_d from scalar search hyperparameters
    local_tune_cfg = resolve_search_hyperparams(tune_config)
    m = local_tune_cfg["m"]

    if m < 3:
        return {
            "status": "failed",
            "reason": f"Motif length m={m} too short (< 3)",
            "metrics_score": -500.0,
        }

    # 2. Materialize LazyFrame and sort deterministically by (day, sid)
    panel_df = collect_lazy_safe(panel_lf).sort(["day", "sid"]).filter(pl.col("lag_0").is_not_null())
    if panel_df.height <= m:
        return {
            "status": "failed",
            "reason": f"Insufficient valid curves (n={panel_df.height} <= m={m})",
            "metrics_score": -500.0,
        }

    # 3. Build 2D curve matrix (with right-tail NaN causal masking) and precompute base_df
    curves_2d = prepare_curves(panel_df, local_tune_cfg, common_config)
    base_df = prepare_fsm_base_df(panel_df, common_config)
    if base_df is None:
        return {
            "status": "failed",
            "reason": "panel has no raw_* target columns",
            "metrics_score": -500.0,
        }

    # 4. Stage A: Extract Top-K candidate motifs from positive T+1 outperformers via STUMPY
    pos_curves = _select_positive_motif_curves(curves_2d, base_df, m, common_config)
    candidate_motifs = get_candidate_motifs(pos_curves, local_tune_cfg, common_config)
    if not candidate_motifs:
        return {
            "status": "failed",
            "reason": "STUMPY returned no valid candidate motifs",
            "metrics_score": -500.0,
        }

    # 5. Stage V: Evaluate candidate motifs on the full unbiased panel (base_df + curves_2d)
    best_result: Optional[Dict[str, Any]] = None
    highest_score: float = -500.0

    for motif in candidate_motifs:
        result = evaluate_and_build_fsm(
            panel_df=base_df,
            curves_2d=curves_2d,
            motif=motif,
            tune_config=local_tune_cfg,
            common_config=common_config,
        )
        if result.get("status") == "success" and result.get("metrics_score", -500.0) > highest_score:
            highest_score = result["metrics_score"]
            best_result = result

    if best_result is not None:
        return best_result

    return {
        "status": "failed",
        "reason": "All candidate motifs failed statistical tests or economic gates",
        "metrics_score": -500.0,
    }
