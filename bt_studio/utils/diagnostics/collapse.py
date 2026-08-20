#! /usr/bin/env python3
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from bt_studio.utils.io import atomic_save_json

try:
    import polars as pl
    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False


# ============================================================================
# Base Tool
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


def _is_discrete_data_driven(values: np.ndarray, atol: float = 1e-5) -> bool:
    """search bound auto infer"""
    arr = np.asarray(values)
    n_samples = len(arr)
    if n_samples == 0:
        return False

    # 1. strict int
    if np.allclose(arr, np.round(arr), atol=atol):
        return True

    # 2. float
    n_unique = len(np.unique(arr))
    uniqueness_ratio = n_unique / n_samples
    if n_samples >= 4 and uniqueness_ratio <= 0.60:
        return True

    # 3. volume limit
    max_classes = max(4, int(np.sqrt(n_samples)))
    if n_unique <= max_classes and uniqueness_ratio < 0.85:
        return True

    return False


def is_discrete_param_by_bounds(
    name: str,
    values: np.ndarray,
    search_bounds: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    1. search_bounds (Ground Truth)
    2. auto infer
    """
    bounds = (search_bounds or {}).get(name) or (search_bounds or {}).get(f"config/{name}")
    
    if bounds is not None:
        if isinstance(bounds, (list, tuple)):
            # categorial
            if any(isinstance(x, (str, bool)) for x in bounds):
                return True
            # enum
            if len(bounds) > 2:
                return True
            # eg randint [1, 10]
            if len(bounds) == 2 and all(isinstance(x, (int, np.integer)) for x in bounds):
                if len(values) > 0 and np.allclose(values, np.round(values), atol=1e-5):
                    return True

    return _is_discrete_data_driven(values)


def _landscape_variance_ratio(
    param_values: np.ndarray,
    target_values: np.ndarray,
    is_discrete: bool = False,
    n_bins: int = 8,
) -> float:
    # ANOVA eta^2
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
            # 离散变量
            rounded_vals = np.round(param_values, decimals=7)
            unique_vals = np.unique(rounded_vals)

            for uv in unique_vals:
                # 2.  rounded / isclose
                mask = (rounded_vals == uv)
                n_k = mask.sum()
                if n_k > 0:
                    mean_k = np.nanmean(target_values[mask])
                    ss_between += n_k * ((mean_k - grand_mean) ** 2)
        else:
            # 连续变量
            bin_edges = np.linspace(param_values.min(), param_values.max(), actual_bins + 1)
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
# Core Diagnose
# ============================================================================

def detect_space_collapse(
    results_df: Any,
    target: str = "metrics_score",
    search_bounds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    df = _to_pandas(results_df)
    param_cols = _get_param_columns(df)

    raw_cols_to_check = [c if c in df.columns else f"config/{c}" for c in param_cols]
    valid = df.dropna(subset=[c for c in raw_cols_to_check if c in df.columns] + [target]).copy()
    valid = valid[np.isfinite(valid[target])].copy()

    if len(valid) < 3:
        raise ValueError(f"trail only ({len(valid)})")

    search_bounds = search_bounds or {}
    diagnostics: Dict[str, Any] = {}

    target_vals = valid[target].values
    best_idx = int(np.nanargmax(target_vals))
    best_score = target_vals[best_idx]

    q75, q25 = np.percentile(target_vals, [75, 25])
    iqr_scale = max(float(q75 - q25), 1e-4)

    for col in param_cols:
        name = _clean_param_name(col)
        raw_col_name = col if col in valid.columns else f"config/{col}"
        if raw_col_name not in valid.columns:
            continue

        col_series = valid[raw_col_name].dropna()
        if len(col_series) == 0:
            continue

        first_elem = col_series.iloc[0]
        is_categorical = isinstance(first_elem, str)
        bounds = search_bounds.get(name, search_bounds.get(col))

        # 1. is_discrete
        if is_categorical:
            is_discrete = True
            values = pd.factorize(col_series)[0].astype(float)
        else:
            values = col_series.values.astype(float)
            is_discrete = is_discrete_param_by_bounds(name, values, search_bounds)

        # 2. spread_ratio
        if bounds is not None and len(bounds) >= 2:
            if is_categorical or (isinstance(bounds, (list, tuple)) and isinstance(bounds[0], str)):
                unique_explored = len(set(col_series))
                total_categories = len(set(bounds))
                spread_ratio = unique_explored / total_categories if total_categories > 0 else 0.0
            else:
                actual_range = float(values.max() - values.min())
                search_range = float(bounds[-1] - bounds[0])
                spread_ratio = (actual_range / search_range) if search_range > 0 else 0.0
            spread_ratio = float(np.clip(spread_ratio, 0.0, 1.0))
        else:
            spread_ratio = float("nan")

        # 3. KDE Entropy vs Shannon Entropy
        actual_range = float(values.max() - values.min())
        std_val = float(np.std(values))
        if not is_discrete and std_val > 1e-10 and len(values) > 5:
            try:
                kde = gaussian_kde(values, bw_method="scott")
                eval_pts = np.linspace(values.min(), values.max(), 200)
                p = np.clip(kde(eval_pts), 1e-12, None)
                dx = eval_pts[1] - eval_pts[0]
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

        # 4. anova
        landscape_var_ratio = _landscape_variance_ratio(values, target_vals, is_discrete=is_discrete)

        # 5. Worst Neighbor & Turbulence
        best_val = valid[raw_col_name].values[best_idx]
        if is_categorical:
            best_val = values[best_idx]

        delta = actual_range * 0.08 if actual_range > 0 else 1e-5
        neighbors_mask = (values >= best_val - delta) & (values <= best_val + delta)
        neighbors_mask[best_idx] = False

        neighbor_scores = target_vals[neighbors_mask]
        if len(neighbor_scores) >= 2:
            neighbor_worst = float(neighbor_scores.min())
            neighbor_std = float(neighbor_scores.std())
            worst_drop_ratio = (best_score - neighbor_worst) / iqr_scale
            local_turbulence = neighbor_std / iqr_scale
        else:
            worst_drop_ratio = 0.0
            local_turbulence = 0.0

        # 6. rule
        is_collapsed = False
        is_isolated_spike = False
        is_flat = False
        reasons = []

        if not is_discrete and not np.isnan(spread_ratio) and spread_ratio < 0.40:
            is_collapsed = True
            reasons.append(f"spread_ratio={spread_ratio:.2f} < 0.40 (探索范围坍塌)")

        if not np.isnan(kde_entropy) and kde_entropy < 0.15 and not is_discrete:
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


# ============================================================================
# report helper
# ============================================================================

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


def save_collapse_report(report: dict, output_path: str) -> None:
    atomic_save_json(report, output_path)
