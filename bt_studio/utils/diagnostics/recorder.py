#!/usr/bin/env python3

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from bt_studio.utils.io import atomic_save_json

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
