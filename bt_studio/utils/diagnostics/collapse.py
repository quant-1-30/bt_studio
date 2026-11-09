#! /usr/bin/env python3

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from bt_studio.utils.io import atomic_save_json

try:
    import polars as pl
    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False


def _to_pandas(results_df: Any) -> pd.DataFrame:
    if _HAS_POLARS and isinstance(results_df, pl.DataFrame):
        return results_df.to_pandas()
    return results_df


def _get_param_columns(df: pd.DataFrame) -> List[str]:
    param_cols = [c for c in df.columns if c.startswith("config/")]
    if not param_cols:
        exclude = {
            "trial_id", "metrics_score", "u_pval", "trigger_count",
            "valid_sample_ratio", "autocorr", "time_this_iter_s", "time_total_s",
        }
        param_cols = [c for c in df.columns if c not in exclude]
    return param_cols


def _clean_param_name(name: str) -> str:
    return name.replace("config/", "")


def is_discrete_param(
    name: str,
    series: pd.Series,
    search_bounds: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, bool]:
    """Return (is_discrete, is_categorical_string) for a hyperparameter series."""
    clean = series.dropna()
    if len(clean) == 0:
        return False, False
    if isinstance(clean.iloc[0], (str, bool)):
        return True, True

    bounds = (search_bounds or {}).get(name) or (search_bounds or {}).get(f"config/{name}")
    vals = clean.values
    is_int_like = bool(np.allclose(vals, np.round(vals), atol=1e-5))

    if isinstance(bounds, (list, tuple)):
        if any(isinstance(x, (str, bool)) for x in bounds):
            return True, True
        if len(bounds) > 2:
            return True, False
        if len(bounds) == 2 and all(isinstance(x, (int, np.integer)) for x in bounds):
            return is_int_like, False
        if len(bounds) == 2:
            return False, False

    return is_int_like, False


def _landscape_variance_ratio(
    param_values: np.ndarray,
    target_values: np.ndarray,
    is_discrete: bool = False,
    n_bins: int = 8,
) -> float:
    if len(param_values) < 5:
        return float("nan")

    total_var = float(np.nanvar(target_values))
    if total_var < 1e-12:
        return 0.0

    n_unique = len(np.unique(param_values))
    actual_bins = min(n_bins, n_unique)
    if actual_bins < 2:
        return 0.0

    if is_discrete:
        groups = np.round(param_values, decimals=6)
    else:
        edges = np.linspace(param_values.min(), param_values.max(), actual_bins + 1)
        if len(np.unique(edges)) < len(edges):
            return 0.0
        groups = np.digitize(param_values, edges[1:-1])

    grand_mean = float(np.nanmean(target_values))
    ss_between = sum(
        int((mask := (groups == g)).sum()) * ((float(np.nanmean(target_values[mask])) - grand_mean) ** 2) # variance
        for g in np.unique(groups)
        if np.any(groups == g)
    )
    return float(np.clip((ss_between / len(target_values)) / total_var, 0.0, 1.0))


def _calc_spread_ratio(
    col_series: pd.Series,
    values: np.ndarray,
    bounds: Any,
    is_categorical: bool,
) -> float:
    if not isinstance(bounds, (list, tuple)):
        return float("nan")
    if is_categorical or len(bounds) > 2:
        total_categories = len(set(bounds))
        ratio = len(set(col_series)) / total_categories if total_categories > 0 else 0.0
    elif len(bounds) == 2:
        search_range = float(abs(bounds[1] - bounds[0]))
        ratio = (float(values.max() - values.min()) / search_range) if search_range > 0 else 0.0
    else:
        return float("nan")
    return float(np.clip(ratio, 0.0, 1.0))


def _calc_param_entropy(values: np.ndarray, is_discrete: bool, n_bins: int = 8) -> float:
    """Normalized Shannon entropy in [0, 1] over discrete categories or equal-width bins."""
    if len(values) <= 1 or float(values.max() - values.min()) <= 1e-12:
        return 0.0
    if is_discrete:
        _, counts = np.unique(values, return_counts=True)
    else:
        counts, _ = np.histogram(values, bins=min(n_bins, max(2, len(values) // 4)))
        counts = counts[counts > 0]
    if len(counts) <= 1:
        return 0.0
    probs = counts / counts.sum()
    return float(np.clip(-np.sum(probs * np.log(probs)) / np.log(len(counts)), 0.0, 1.0))


def _calc_local_turbulence(
    col: str,
    values: np.ndarray,
    valid: pd.DataFrame,
    param_cols: List[str],
    search_bounds: Dict[str, Any],
    target_vals: np.ndarray,
    best_idx: int,
    best_score: float,
    iqr_scale: float,
    higher_is_better: bool,
) -> Tuple[float, float]:
    best_val = values[best_idx]
    delta = max(float(np.std(values)) * 0.5, 1e-5)
    neighbors_mask = (values >= best_val - delta) & (values <= best_val + delta)

    for other_col in param_cols:
        if other_col == col or other_col not in valid.columns:
            continue
        o_series = valid[other_col].dropna()
        if len(o_series) == 0:
            continue
        o_disc, o_cat = is_discrete_param(_clean_param_name(other_col), o_series, search_bounds)
        if not o_disc:
            continue
        if o_cat:
            o_vals = valid[other_col].astype(str).values
            neighbors_mask &= (o_vals == o_vals[best_idx])
        else:
            o_vals = valid[other_col].values.astype(float)
            neighbors_mask &= np.isclose(o_vals, o_vals[best_idx], rtol=0.0, atol=1e-8)

    neighbors_mask[best_idx] = False
    neighbor_scores = target_vals[neighbors_mask]
    if len(neighbor_scores) < 2:
        return 0.0, 0.0

    drop = (
        (best_score - float(neighbor_scores.min()))
        if higher_is_better
        else (float(neighbor_scores.max()) - best_score)
    )
    return drop / iqr_scale, float(neighbor_scores.std()) / iqr_scale


def detect_space_collapse(
    results_df: Any,
    target: str = "metrics_score",
    search_bounds: Optional[Dict[str, Any]] = None,
    higher_is_better: bool = True,
) -> Dict[str, Any]:
    df = _to_pandas(results_df)
    param_cols = _get_param_columns(df)

    raw_cols_to_check = [c for c in param_cols if c in df.columns]
    valid = df.dropna(subset=raw_cols_to_check + [target]).copy()
    valid = valid[np.isfinite(valid[target])].copy()

    if len(valid) < 3:
        raise ValueError(f"Too few valid trials ({len(valid)}) to diagnose space collapse.")

    search_bounds = search_bounds or {}
    diagnostics: Dict[str, Any] = {}

    target_vals = valid[target].values
    best_idx = int(np.nanargmax(target_vals) if higher_is_better else np.nanargmin(target_vals))
    best_score = float(target_vals[best_idx])

    q75, q25 = np.percentile(target_vals, [75, 25])
    iqr_scale = max(float(q75 - q25), 1e-4)

    for col in param_cols:
        name = _clean_param_name(col)
        if col not in valid.columns:
            continue

        col_series = valid[col].dropna()
        if len(col_series) == 0:
            continue

        is_discrete, is_categorical = is_discrete_param(name, col_series, search_bounds)
        bounds = search_bounds.get(name) or search_bounds.get(col)
        values = (
            pd.factorize(col_series)[0].astype(float)
            if is_categorical
            else col_series.values.astype(float)
        )

        spread_ratio = _calc_spread_ratio(col_series, values, bounds, is_categorical)
        kde_entropy = _calc_param_entropy(values, is_discrete)
        landscape_var_ratio = _landscape_variance_ratio(values, target_vals, is_discrete=is_discrete)

        worst_drop_ratio, local_turbulence = (
            (0.0, 0.0)
            if is_categorical
            else _calc_local_turbulence(
                col, values, valid, param_cols, search_bounds,
                target_vals, best_idx, best_score, iqr_scale, higher_is_better,
            )
        )

        is_collapsed = False
        is_isolated_spike = False
        is_flat = False
        reasons = []

        if not np.isnan(spread_ratio) and spread_ratio < 0.40:
            is_collapsed = True
            reasons.append(f"spread_ratio={spread_ratio:.2f} < 0.40")
        if not np.isnan(kde_entropy) and kde_entropy < 0.15:
            is_collapsed = True
            reasons.append(f"kde_entropy={kde_entropy:.3f} < 0.15")
        if not np.isnan(landscape_var_ratio) and landscape_var_ratio < 0.05:
            is_flat = True
            reasons.append(f"landscape_var_ratio={landscape_var_ratio:.3f} < 0.05")
        if worst_drop_ratio > 1.5 or local_turbulence > 1.0:
            is_isolated_spike = True
            reasons.append(f"worst_drop={worst_drop_ratio:.2f}xIQR, turbulence={local_turbulence:.2f}xIQR")

        diagnostics[name] = {
            "n_samples": len(values),
            "is_discrete": is_discrete,
            "is_categorical": is_categorical,
            "spread_ratio": float(spread_ratio),
            "kde_entropy": float(kde_entropy),
            "landscape_variance_ratio": float(landscape_var_ratio),
            "worst_drop_ratio": float(worst_drop_ratio),
            "local_turbulence": float(local_turbulence),
            "is_collapsed": is_collapsed,
            "is_flat": is_flat,
            "is_isolated_spike": is_isolated_spike,
            "reasons": reasons,
        }

    param_dicts = [d for d in diagnostics.values() if isinstance(d, dict)]
    if any(d["is_isolated_spike"] for d in param_dicts):
        verdict = "isolated_spike"
    elif param_dicts and all(d["is_flat"] for d in param_dicts):
        verdict = "flat_landscape"
    elif any(d["is_collapsed"] for d in param_dicts):
        verdict = "collapsed"
    else:
        verdict = "healthy_plateau"

    diagnostics["overall_verdict"] = verdict
    diagnostics["_param_cols"] = [f"config/{p}" for p in diagnostics if not p.startswith("_") and p != "overall_verdict"]
    diagnostics["_target"] = target
    diagnostics["_n_trials"] = len(valid)
    return diagnostics


_VERDICT_TAG = {
    "healthy_plateau": "[OK]",
    "collapsed": "[!]",
    "isolated_spike": "[^]",
    "flat_landscape": "[--]",
}


def print_collapse_report(diagnostics: dict) -> str:
    verdict = diagnostics.get("overall_verdict", "unknown")
    tag = _VERDICT_TAG.get(verdict, "?")
    lines = [
        "=" * 68,
        f"{tag} Parameter-Space Verdict: {verdict.upper()}",
        f"  n_trials = {diagnostics.get('_n_trials', '?')} | target = {diagnostics.get('_target', '?')}",
        "-" * 68,
    ]
    for k, d in diagnostics.items():
        if k.startswith("_") or not isinstance(d, dict):
            continue
        flag = (
            "COLLAPSED" if d.get("is_collapsed")
            else "SPIKE" if d.get("is_isolated_spike")
            else "FLAT" if d.get("is_flat") else "OK"
        )
        lines.append(
            f"  {k:<16} {flag:<10} spread={d.get('spread_ratio', float('nan')):.2f} "
            f"entropy={d.get('kde_entropy', float('nan')):.3f} "
            f"lv_ratio={d.get('landscape_variance_ratio', float('nan')):.3f} "
            f"worst_drop={d.get('worst_drop_ratio', 0.0):.2f}xIQR"
        )
        for r in d.get("reasons", []):
            lines.append(f"      - {r}")
    lines.append("=" * 68)
    report = "\n".join(lines)
    print(report)
    return report


def build_collapse_report(
    run_name: str,
    feature_col: str,
    model_id: Any,
    diagnostics: dict,
    search_bounds: Optional[dict] = None,
) -> dict:
    param_diag = {
        k: v for k, v in diagnostics.items()
        if not k.startswith("_") and isinstance(v, dict)
    }
    return {
        "run_name": run_name,
        "feature_col": feature_col,
        "model_id": model_id,
        "verdict": diagnostics.get("overall_verdict", "unknown"),
        "n_trials": diagnostics.get("_n_trials", 0),
        "search_bounds": search_bounds or {},
        "diagnostics": param_diag,
        "created_at": datetime.now().isoformat(),
    }

