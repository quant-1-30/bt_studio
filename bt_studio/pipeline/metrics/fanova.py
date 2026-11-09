#! /usr/bin/env python3

import optuna
import numpy as np
import polars as pl
from typing import Any, Dict, List
from optuna.importance import FanovaImportanceEvaluator

from bt_studio.default_config import build_optuna_distributions


def _build_neighborhood_conditions(
    valid_df: pl.DataFrame,
    sorted_params: List[tuple[str, float]],
    search_spec: Dict[str, Dict[str, Any]],
    norm_best_cfg: Dict[str, Any],
) -> List[pl.Expr]:
    """Build local parameter neighborhood filters without arbitrary radius_ratio hyperparameters."""
    top_n = min(2, len(sorted_params))
    nearby_conds: List[pl.Expr] = []

    for pname, _ in sorted_params[:top_n]:
        meta = search_spec[pname]
        val = norm_best_cfg.get(pname)
        if val is None:
            continue

        p_col = f"config/{pname}"
        p_type = meta["type"]
        if p_type == "choice":
            vals = meta["values"]
            if len(vals) > 1 and all(isinstance(x, (int, float)) for x in vals):
                sorted_vals = sorted(vals)
                idx = (
                    sorted_vals.index(val)
                    if val in sorted_vals
                    else int(np.argmin([abs(x - val) for x in sorted_vals]))
                )
                allowed_vals = sorted_vals[max(0, idx - 1) : min(len(sorted_vals), idx + 2)]
                nearby_conds.append(pl.col(p_col).is_in(allowed_vals))
            else:
                nearby_conds.append(pl.col(p_col) == val)
        elif p_type == "int_range":
            step = int(meta.get("step", 1))
            nearby_conds.append(pl.col(p_col).is_between(int(val) - step, int(val) + step))
        elif p_type == "uniform":
            # Scale-free empirical standard deviation of sampled trials along p_col
            emp_std = float(valid_df[p_col].std() or 0.0)
            delta = max(emp_std, 1e-4)
            nearby_conds.append(pl.col(p_col).is_between(float(val) - delta, float(val) + delta))

    if not nearby_conds:
        return []

    for pname, _ in sorted_params[top_n:]:
        lock_val = norm_best_cfg.get(pname)
        if lock_val is not None:
            nearby_conds.append(pl.col(f"config/{pname}") == lock_val)

    return nearby_conds


def validate_parameter_plateau_fanova(
    df_results: pl.DataFrame,
    best_config: Dict[str, Any],
    best_score: float,
    search_spec: Dict[str, Dict[str, Any]],
) -> bool:
    """fANOVA Plateau test to reject isolated hyperparameter spikes."""
    norm_best_cfg = {k.removeprefix("config/"): v for k, v in best_config.items()}

    cfg_cols = [
        f"config/{k}" for k in search_spec.keys() if f"config/{k}" in df_results.columns
    ]
    if not cfg_cols or "metrics_score" not in df_results.columns:
        return False

    valid_df = (
        df_results.filter(pl.col("metrics_score") > -500.0)
        .drop_nulls(subset=cfg_cols + ["metrics_score"])
        .filter(pl.col("metrics_score").is_not_nan())
        .filter(pl.all_horizontal([pl.col(c).is_not_nan() for c in cfg_cols]))
    )
    if valid_df.height < 20:
        return False

    distributions, active_params = build_optuna_distributions(valid_df, search_spec)
    if not distributions:
        return True

    study = optuna.create_study(direction="maximize")
    for row in valid_df.iter_rows(named=True):
        study.add_trial(
            optuna.trial.create_trial(
                params={p: row[f"config/{p}"] for p in active_params},
                distributions=distributions,
                value=float(row["metrics_score"]),
            )
        )

    try:
        evaluator = FanovaImportanceEvaluator(seed=0)
        importances = optuna.importance.get_param_importances(study, evaluator=evaluator)
    except Exception:
        return False

    sorted_params = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    if not sorted_params:
        return False

    top_params = sorted_params[: min(2, len(sorted_params))]
    nearby_conds = _build_neighborhood_conditions(
        valid_df, sorted_params, search_spec, norm_best_cfg
    )
    if not nearby_conds:
        return True

    unique_neighbors = valid_df.filter(pl.all_horizontal(nearby_conds)).unique(subset=cfg_cols, keep="first")
    n_neighbors = max(0, unique_neighbors.height - 1)
    if n_neighbors < 2:
        return False

    scores = valid_df["metrics_score"].to_numpy()
    q75, q25 = np.percentile(scores, [75, 25])
    iqr_scale = max(float(q75 - q25), 0.05 * abs(best_score), 1e-4)

    worst_neighbor = float(unique_neighbors["metrics_score"].min())
    drop = max(0.0, best_score - worst_neighbor)
    if drop > 1.5 * iqr_scale:
        return False

    param_summary = ", ".join([f"{p}({imp:.1%})" for p, imp in top_params])
    print(f"✅ Plateau passed: Top [{param_summary}] and neighbor maximum drop {drop:.3f}")
    return True
