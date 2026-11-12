#! /usr/bin/env python3

from typing import Any, Dict, Optional
import numpy as np
import polars as pl

from bt_studio.utils.math_func import resolve_search_hyperparams
from .fanova import validate_parameter_plateau_fanova


def find_pareto_front(df_results: pl.DataFrame) -> pl.DataFrame:
    """Extract the non-dominated Pareto front across (efficiency, valid_sample_ratio, momentum_persistence)."""
    valid_df = (
        df_results
        .filter(pl.col("metrics_score") > -500.0)
        .drop_nulls(subset=["metrics_score", "valid_sample_ratio", "autocorr"])
        .filter(
            pl.col("metrics_score").is_not_nan()
            & pl.col("valid_sample_ratio").is_not_nan()
            & pl.col("autocorr").is_not_nan()
        )
    )
    if valid_df.height <= 1:
        return valid_df

    score_floor = float(valid_df["metrics_score"].min())
    valid_df = valid_df.with_columns([
        (pl.col("metrics_score") - score_floor + 1.0).alias("efficiency"),
        (1.0 - pl.col("autocorr").abs().clip(0.0, 1.0)).alias("momentum_persistence"),
    ])

    target_cols = ["efficiency", "valid_sample_ratio", "momentum_persistence"]
    obj_matrix = valid_df.select(target_cols).to_numpy()

    diffs = obj_matrix[None, :, :] - obj_matrix[:, None, :]
    j_dominates_i = np.all(diffs >= 0.0, axis=2) & np.any(diffs > 0.0, axis=2)
    is_dominated = np.any(j_dominates_i, axis=1)

    return valid_df.filter(~is_dominated)


def select_best_model_from_pareto(pareto_df: pl.DataFrame) -> Optional[Dict[str, Any]]:
    """Select the highest geometric utility trial from the Pareto front."""
    if pareto_df.height == 0:
        return None
    if pareto_df.height == 1:
        return pareto_df.to_dicts()[0]

    utility_df = (
        pareto_df
        .with_columns(
            (
                pl.col("efficiency").log1p()
                * pl.col("valid_sample_ratio").sqrt()
                * pl.col("momentum_persistence")
            ).alias("final_utility")
        )
        .sort("final_utility", descending=True)
        .limit(1)
    )
    return utility_df.to_dicts()[0]


def _extract_trial_config(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k.replace("config/", ""): v for k, v in row.items() if k.startswith("config/")}


def select_plateau_verified_config(
    df_results: pl.DataFrame,
    stats_valid_trials: pl.DataFrame,
    search_spec: dict,
) -> Optional[Dict[str, Any]]:
    """Select a Pareto-optimal configuration verified by fANOVA neighborhood plateau stability."""
    pareto_front_df = find_pareto_front(stats_valid_trials)
    best_model_dict = select_best_model_from_pareto(pareto_front_df)
    if not best_model_dict:
        return None

    best_config = _extract_trial_config(best_model_dict)
    pareto_score = float(best_model_dict.get("metrics_score", -500.0))

    best_valid_row = stats_valid_trials.sort("metrics_score", descending=True).row(0, named=True)
    best_valid_score = float(best_valid_row["metrics_score"])
    best_valid_config = _extract_trial_config(best_valid_row)

    is_plateau = validate_parameter_plateau_fanova(
        df_results=df_results,
        best_config=best_config,
        best_score=pareto_score,
        search_spec=search_spec,
    )
    if not is_plateau and best_config != best_valid_config:
        is_plateau = validate_parameter_plateau_fanova(
            df_results=df_results,
            best_config=best_valid_config,
            best_score=best_valid_score,
            search_spec=search_spec,
        )
        if is_plateau:
            best_config = best_valid_config

    return resolve_search_hyperparams(best_config) if is_plateau else None
