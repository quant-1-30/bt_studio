#! /usr/bin/env python3
from __future__ import annotations

import os
import traceback
import polars as pl
from typing import Any, Dict, List, Tuple

from bt_studio.constant import TUNE_COLLAPSE_DIR
from .collapse import detect_space_collapse
from .recorder import (
    build_collapse_report,
    save_collapse_report,
    print_collapse_report
)

__all__ = ["build_search_bounds", "run_collapse_check"]


def build_search_bounds(search_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    HPO search_config ---> collapse bound dict
    - string: return 
    - int list / tuple : sort [min. max]
    - filter num_trails
    """
    bounds: Dict[str, Any] = {}
    if not isinstance(search_config, dict):
        return bounds

    for k, v in search_config.items():
        if k in ("num_trials", "max_concurrent_trials", "metric", "mode"):
            continue
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            continue

        raw_list = list(v)
        
        # 1. Categorical
        if any(isinstance(x, str) for x in raw_list):
            bounds[k] = list(dict.fromkeys(raw_list)) # unqiue 
        # 2. Continuous / Discrete Int
        else:
            try:
                sorted_vals = sorted(raw_list)
                all_int = all(isinstance(x, (int, bool)) for x in sorted_vals)
                # [min, max]
                bounds[k] = sorted_vals if all_int else [sorted_vals[0], sorted_vals[-1]]
            except Exception:
                continue

    return bounds


def run_collapse_check(
    df_results: pl.DataFrame,
    exp_config: Dict[str, Any],
    model_id: Any,
    strict_healthy: bool = False,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Parameters
    ----------
    df_results : pl.DataFrame
        Ray Tune DataFrame
    exp_config : Dict[str, Any]
        common_params and search_bounds
    model_id : Any
        ID train_month[-1]
    strict_healthy : bool, default False
        True and 'healthy_plateau' pass
        False and 'flat_landscape' and 'isolated_spike' 'collapsed'

    Returns
    -------
    Tuple[bool, str, Dict[str, Any]]
        - is_valid: bool
        - verdict: 'healthy_plateau' | 'collapsed' | 'isolated_spike' | 'flat_landscape'
        - diagnostics: dict
    """
    if df_results is None or df_results.height < 5:
        print("[Collapse] trial < 5 and skip diagnose")
        return True, "insufficient_samples", {}

    try:
        common_cfg = exp_config.get("common_params", {})
        search_bounds = exp_config.get("search_bounds", {})
        feature_col = exp_config.get("feature_col", "unknown_feat")
        run_name = exp_config.get("run_name", f"tune_{model_id}")
        
        bounds = build_search_bounds(search_bounds)

        diagnostics = detect_space_collapse(
            results_df=df_results,
            target="metrics_score",
            search_bounds=bounds,
        )
        verdict = diagnostics.get("overall_verdict", "unknown")

        os.makedirs(TUNE_COLLAPSE_DIR, exist_ok=True)
        report = build_collapse_report(
            run_name=run_name,
            feature_col=feature_col,
            model_id=model_id,
            diagnostics=diagnostics,
            search_bounds=search_bounds,
        )
        report_path = os.path.join(
            TUNE_COLLAPSE_DIR, f"collapse_{model_id}_{feature_col}.json"
        )
        save_collapse_report(report, report_path)

        print(f"📊 [Collapse Check] Model={model_id} | Feature={feature_col} | Verdict={verdict.upper()}")
        print_collapse_report(diagnostics)

        if strict_healthy:
            is_valid = (verdict == "healthy_plateau")
        else:
            is_valid = verdict not in ("flat_landscape", "isolated_spike")

        return is_valid, verdict, diagnostics

    except Exception as e:
        print(f"⚠️ [Collapse Check] not vital: {e}\n{traceback.format_exc()}")
        return True, "error_fallback", {}
