#! /usr/bin/env python3

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

try:
    import polars as pl
    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False


# ============================================================================
# Base Utilities
# ============================================================================

def _to_pandas(results_df: Any) -> pd.DataFrame:
    if _HAS_POLARS and isinstance(results_df, pl.DataFrame):
        return results_df.to_pandas()
    return results_df


def _get_param_columns(df: pd.DataFrame) -> List[str]:
    param_cols = [c for c in df.columns if c.startswith("config/")]
    if not param_cols:
        exclude = {
            "trial_id", "metrics_score", "u_pval", "trigger_count",
            "valid_sample_ratio", "autocorr", "time_this_iter_s", "time_total_s"
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
    """
    (is_discrete , is_categorical_string)
    
    Priority:
    1. str / bool --> discrete
    2. search_bounds Ground Truth
    3. infer ( float maybe continual)
    """
    first_valid = series.dropna().iloc[0] if len(series.dropna()) > 0 else None
    if isinstance(first_valid, (str, bool)):
        return True, True

    bounds = (search_bounds or {}).get(name) or (search_bounds or {}).get(f"config/{name}")

    if bounds is not None and isinstance(bounds, (list, tuple)):
        # 1. str / bool
        if any(isinstance(x, (str, bool)) for x in bounds):
            return True, True
        # 2. enum eg [0.001, 0.01, 0.1] 或 [16, 32, 64, 128]
        if len(bounds) > 2:
            return True, False
        # 3. integer config and val  eg [1, 100]
        if len(bounds) == 2 and all(isinstance(x, (int, np.integer)) for x in bounds):
            vals = series.dropna().values
            if len(vals) > 0 and np.allclose(vals, np.round(vals), atol=1e-5):
                return True, False
        # 4. float eg [0.0, 1.0]
        if len(bounds) == 2 and any(isinstance(x, (float, np.floating)) for x in bounds):
            return False, False

    # no bounds
    vals = series.dropna().values
    if len(vals) == 0:
        return False, False

    # pure integer
    if np.allclose(vals, np.round(vals), atol=1e-5):
        return True, False

    # intend float maybe not discrete otherwise collapse
    return False, False


def _landscape_variance_ratio(
    param_values: np.ndarray,
    target_values: np.ndarray,
    is_discrete: bool = False,
    n_bins: int = 8,
) -> float:
    if len(param_values) < 5:
        return float("nan")

    total_var = np.nanvar(target_values)
    if total_var < 1e-12:
        return 0.0

    grand_mean = np.nanmean(target_values)
    n_unique = len(np.unique(param_values))
    actual_bins = min(n_bins, n_unique)
    if actual_bins < 2:
        return 0.0

    try:
        ss_between = 0.0
        if is_discrete:
            rounded_vals = np.round(param_values, decimals=7)
            for uv in np.unique(rounded_vals):
                mask = (rounded_vals == uv)
                n_k = mask.sum()
                if n_k > 0:
                    mean_k = np.nanmean(target_values[mask])
                    ss_between += n_k * ((mean_k - grand_mean) ** 2)
        else:
            bin_edges = np.linspace(param_values.min(), param_values.max(), actual_bins + 1)
            # avoid duplicate bounds``
            if len(np.unique(bin_edges)) < len(bin_edges):
                return 0.0
            bin_indices = np.digitize(param_values, bin_edges[1:-1])
            for bi in range(actual_bins):
                mask = (bin_indices == bi)
                n_k = mask.sum()
                if n_k > 0:
                    mean_k = np.nanmean(target_values[mask])
                    ss_between += n_k * ((mean_k - grand_mean) ** 2)

        var_between = ss_between / len(target_values)
        return float(np.clip(var_between / total_var, 0.0, 1.0))
    except Exception:
        return float("nan")


# ============================================================================
# Core Diagnosis
# ============================================================================

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
    best_score = target_vals[best_idx]

    q75, q25 = np.percentile(target_vals, [75, 25])
    iqr_scale = max(float(q75 - q25), 1e-4)

    for col in param_cols:
        name = _clean_param_name(col)
        if col not in valid.columns:
            continue

        col_series = valid[col].dropna()
        if len(col_series) == 0:
            continue

        # 1. discrete or categorical
        is_discrete, is_categorical = is_discrete_param(name, col_series, search_bounds)
        bounds = search_bounds.get(name) or search_bounds.get(col)

        if is_categorical:
            factorized_codes, uniques = pd.factorize(col_series)
            values = factorized_codes.astype(float)
        else:
            values = col_series.values.astype(float)

        # 2. Spread Ratio
        spread_ratio = float("nan")
        if bounds is not None and isinstance(bounds, (list, tuple)):
            if is_categorical or len(bounds) > 2:
                
                unique_explored = len(set(col_series))
                total_categories = len(set(bounds))
                spread_ratio = unique_explored / total_categories if total_categories > 0 else 0.0
            elif len(bounds) == 2:
                
                actual_range = float(values.max() - values.min())
                search_range = float(abs(bounds[1] - bounds[0]))
                spread_ratio = (actual_range / search_range) if search_range > 0 else 0.0
            spread_ratio = float(np.clip(spread_ratio, 0.0, 1.0))

        # 3. Entropy fixbug
        actual_range = float(values.max() - values.min())
        std_val = float(np.std(values))

        if not is_discrete and std_val > 1e-10 and len(values) > 5:
            try:
                # extend  3*bw 
                kde = gaussian_kde(values, bw_method="scott")
                bw = np.sqrt(kde.covariance[0, 0])
                eval_pts = np.linspace(values.min() - 2 * bw, values.max() + 2 * bw, 250)
                p = np.clip(kde(eval_pts), 1e-12, None)
                dx = eval_pts[1] - eval_pts[0]
                p = p / (np.sum(p) * dx) 
                diff_entropy = -np.sum(p * np.log(p)) * dx
                h_max = np.log(actual_range) if actual_range > 1e-12 else 0.0
                kde_entropy = float(np.clip(np.exp(diff_entropy - h_max), 0.0, 1.0))
            except Exception:
                kde_entropy = float("nan")
        else:
            unique, counts = np.unique(values, return_counts=True)
            probs = counts / counts.sum()
            shannon_h = -np.sum(probs * np.log(probs))
            max_h = np.log(len(unique)) if len(unique) > 1 else 1.0
            kde_entropy = float(shannon_h / max_h) if max_h > 0 else 0.0

        # 4. ANOVA
        landscape_var_ratio = _landscape_variance_ratio(values, target_vals, is_discrete=is_discrete)

        # 5. Worst Neighbor & Turbulence) - intended for continuals
        worst_drop_ratio = 0.0
        local_turbulence = 0.0

        if not is_categorical:
            best_val = values[best_idx]
            delta = actual_range * 0.08 if actual_range > 0 else 1e-5
            neighbors_mask = (values >= best_val - delta) & (values <= best_val + delta)
            neighbors_mask[best_idx] = False

            neighbor_scores = target_vals[neighbors_mask]
            if len(neighbor_scores) >= 2:
                if higher_is_better:
                    neighbor_worst = float(neighbor_scores.min())
                    worst_drop_ratio = (best_score - neighbor_worst) / iqr_scale
                else:
                    neighbor_worst = float(neighbor_scores.max())
                    worst_drop_ratio = (neighbor_worst - best_score) / iqr_scale
                local_turbulence = float(neighbor_scores.std()) / iqr_scale

        # 6. rule
        is_collapsed = False
        is_isolated_spike = False
        is_flat = False
        reasons = []

        if not np.isnan(spread_ratio) and spread_ratio < 0.40:
            is_collapsed = True
            reasons.append(f"spread_ratio={spread_ratio:.2f} < 0.40 (探索范围坍塌)")

        if not np.isnan(kde_entropy) and kde_entropy < 0.15:
            is_collapsed = True
            reasons.append(f"kde_entropy={kde_entropy:.3f} < 0.15 (样本极度集中)")

        if not np.isnan(landscape_var_ratio) and landscape_var_ratio < 0.05:
            is_flat = True
            reasons.append(f"landscape_var_ratio={landscape_var_ratio:.3f} < 0.05 (参数调节无效果)")

        if worst_drop_ratio > 1.5 or local_turbulence > 1.0:
            is_isolated_spike = True
            reasons.append(f"worst_drop={worst_drop_ratio:.2f}xIQR, turbulence={local_turbulence:.2f}xIQR (孤立尖刺峰)")

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

    # conclusion
    any_spike = any(d["is_isolated_spike"] for d in diagnostics.values() if isinstance(d, dict))
    any_collapsed = any(d["is_collapsed"] for d in diagnostics.values() if isinstance(d, dict))
    all_flat = len(diagnostics) > 0 and all(d["is_flat"] for d in diagnostics.values() if isinstance(d, dict))

    if any_spike:
        verdict = "isolated_spike"
    elif all_flat:
        verdict = "flat_landscape"
    elif any_collapsed:
        verdict = "collapsed"
    else:
        verdict = "healthy_plateau"

    diagnostics["overall_verdict"] = verdict
    diagnostics["_param_cols"] = [f"config/{p}" for p in diagnostics.keys() if not p.startswith("_") and p != "overall_verdict"]
    diagnostics["_target"] = target
    diagnostics["_n_trials"] = len(valid)

    return diagnostics
