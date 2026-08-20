#!/usr/bin/env python3

import os
import sys
import json
import tempfile

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from bt_studio.utils.diagnostics import (
    detect_space_collapse,
    print_collapse_report,
    build_collapse_report,
    save_collapse_report,
)
from bt_studio.utils.diagnostics.hpo_check import build_search_bounds

RNG = np.random.default_rng(42)


def _make_df(cols: dict) -> pd.DataFrame:
    data = {f"config/{k}": v for k, v in cols.items() if k != "metrics_score"}
    data["metrics_score"] = cols["metrics_score"]
    return pd.DataFrame(data)


def _healthy_df():
    """Converged-search landscape: explore broad, exploit the plateau."""
    n_explore, n_exploit = 30, 50
    ds = np.concatenate([RNG.choice([3, 5], n_explore), np.full(n_exploit, 4)])
    mm = np.concatenate([RNG.choice([30, 60], n_explore), np.full(n_exploit, 45)])
    tr = np.concatenate([RNG.uniform(0.55, 0.85, n_explore),
                         RNG.normal(0.70, 0.02, n_exploit)])
    score = (
        -20.0 * (tr - 0.70) ** 2
        - 0.05 * (ds - 4) ** 2
        - 0.0002 * (mm - 45) ** 2
        + RNG.normal(0, 0.02, n_explore + n_exploit)
    )
    bounds = {"downsample": [3, 4, 5], "motif_minutes": [30, 45, 60],
              "threshold_r": [0.55, 0.85]}
    return _make_df({"downsample": ds, "motif_minutes": mm,
                     "threshold_r": tr, "metrics_score": score}), bounds


def _collapsed_df():
    """Narrow-band continuous sampling (spread 0.17 < 0.4) + group step.

    The ds==5 group sits far above the rest, inflating the global IQR so
    noise slices stay under the spike threshold; lookback never reaches
    beyond [20, 35] inside its [10, 100] search range.
    """
    rng = np.random.default_rng(7)
    n = 60
    ds = rng.choice([3, 4, 5], n)
    mm = rng.choice([30, 45, 60], n)
    lb = rng.uniform(20.0, 35.0, n)
    score = 5.0 * (ds == 5) + rng.normal(0, 0.08, n)
    bounds = {"downsample": [3, 4, 5], "motif_minutes": [30, 45, 60],
              "lookback": [10, 100]}
    return _make_df({"downsample": ds, "motif_minutes": mm,
                     "lookback": lb, "metrics_score": score}), bounds


def _spike_df(n=40):
    """Flat noisy landscape + one extreme isolated spike."""
    ds = RNG.choice([3, 4, 5], n)
    mm = RNG.choice([30, 45, 60], n)
    tr = RNG.uniform(0.55, 0.85, n)
    score = RNG.normal(0, 0.1, n)
    tr[0] = 0.70
    score[0] = 10.0                            # ~100x the noise floor
    bounds = {"downsample": [3, 4, 5], "motif_minutes": [30, 45, 60],
              "threshold_r": [0.55, 0.85]}
    return _make_df({"downsample": ds, "motif_minutes": mm,
                     "threshold_r": tr, "metrics_score": score}), bounds


def _flat_df(n=40):
    """Params have zero effect on the target."""
    ds = RNG.choice([3, 4, 5], n)
    mm = RNG.choice([30, 45, 60], n)
    tr = RNG.uniform(0.55, 0.85, n)
    score = RNG.normal(0, 1e-9, n)             # constant target
    bounds = {"downsample": [3, 4, 5], "motif_minutes": [30, 45, 60],
              "threshold_r": [0.55, 0.85]}
    return _make_df({"downsample": ds, "motif_minutes": mm,
                     "threshold_r": tr, "metrics_score": score}), bounds


def main() -> int:
    # --- verdicts ---------------------------------------------------------
    df, bounds = _healthy_df()
    v = detect_space_collapse(df, "metrics_score", bounds)["overall_verdict"]
    healthy_df, healthy_bounds = df, bounds
    assert v == "healthy_plateau", f"expected healthy_plateau, got {v}"
    print(f"[ok] healthy → {v}")

    df, bounds = _collapsed_df()
    v = detect_space_collapse(df, "metrics_score", bounds)["overall_verdict"]
    assert v == "collapsed", f"expected collapsed, got {v}"
    print(f"[ok] collapsed → {v}")

    df, bounds = _spike_df()
    v = detect_space_collapse(df, "metrics_score", bounds)["overall_verdict"]
    assert v == "isolated_spike", f"expected isolated_spike, got {v}"
    print(f"[ok] spike → {v}")

    df, bounds = _flat_df()
    v = detect_space_collapse(df, "metrics_score", bounds)["overall_verdict"]
    assert v == "flat_landscape", f"expected flat_landscape, got {v}"
    print(f"[ok] flat → {v}")

    # --- build_search_bounds ------------------------------------------------
    sb = build_search_bounds({
        "downsample": [3, 4, 5], "motif_minutes": [30, 45, 60, 90],
        "threshold_r": [0.65, 0.85], "num_trials": 50,
    })
    assert sb["downsample"] == [3, 4, 5], sb
    assert sb["motif_minutes"] == [30, 45, 60, 90], sb
    assert sb["threshold_r"] == [0.65, 0.85], sb
    assert "num_trials" not in sb
    print("[ok] build_search_bounds: discrete passthrough + continuous min/max + budget skipped")

    # --- report roundtrip ----------------------------------------------------
    df, bounds = healthy_df, healthy_bounds
    diag = detect_space_collapse(df, "metrics_score", bounds)
    report = build_collapse_report("fsm_hpo_test_202412", "test_feat", 202412,
                                   diag, bounds)
    assert report["verdict"] == "healthy_plateau"
    assert report["feature_col"] == "test_feat"
    assert report["model_id"] == 202412
    assert "downsample" in report["diagnostics"]

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "collapse_202412_test_feat.json")
        save_collapse_report(report, path)
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["verdict"] == report["verdict"]
        assert loaded["diagnostics"] == report["diagnostics"]
    print("[ok] collapse report JSON roundtrip")

    text = print_collapse_report(diag)
    assert "healthy_plateau".upper() in text.upper()
    print("[ok] print_collapse_report renders verdict")

    print("\n[test_collapse] ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())