#! /usr/bin/env python3
from __future__ import annotations

import os
import traceback
import polars as pl
from typing import Any, Dict, List, Optional, Tuple

from bt_studio.constant import TUNE_COLLAPSE_DIR
from bt_studio.default_config import (
    resolve_feature_col,
    search_spec_to_collapse_bounds,
)
from .collapse import (
    detect_space_collapse,
    build_collapse_report,
    print_collapse_report,
)

from bt_studio.utils.io import atomic_save_json


__all__ = ["build_search_bounds", "run_collapse_check"]


build_search_bounds = search_spec_to_collapse_bounds


def run_collapse_check(
    df_results: pl.DataFrame,
    exp_config: Dict[str, Any],
    model_id: Any,
    strict_healthy: bool = False,
    collapse_dir: Optional[str] = None,
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
        common_cfg = exp_config.get("common_params") or exp_config.get("common_config") or {}
        search_bounds = exp_config.get("search_spec") or {}
        try:
            feature_col = resolve_feature_col(
                common_cfg, exp_config.get("ast_recipe"), exp_config.get("feature_col")
            )
        except ValueError:
            feature_col = common_cfg.get("_task_id") or "unknown_feat"

        run_name = (
            exp_config.get("run_name")
            or common_cfg.get("run_name")
            or common_cfg.get("_task_id")
            or f"tune_{model_id}"
        )
        
        bounds = build_search_bounds(search_bounds)

        diagnostics = detect_space_collapse(
            results_df=df_results,
            target="metrics_score",
            search_bounds=bounds,
        )
        verdict = diagnostics.get("overall_verdict", "unknown")

        report_dir = collapse_dir or TUNE_COLLAPSE_DIR
        os.makedirs(report_dir, exist_ok=True)
        report = build_collapse_report(
            run_name=run_name,
            feature_col=feature_col,
            model_id=model_id,
            diagnostics=diagnostics,
            search_bounds=search_bounds,
        )
        report_path = os.path.join(
            report_dir, f"collapse_{model_id}_{feature_col}.json"
        )
        atomic_save_json(report, report_path)

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
