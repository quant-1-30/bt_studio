#!/usr/bin/env python3
"""HPO 参数空间塌陷检测 (utils.diagnostics.collapse) 单测。

由原脚本式 main() 测试改写为 pytest 用例 (原版 pytest 收集数为 0)。
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd

from bt_studio.utils.diagnostics import (
    detect_space_collapse,
    print_collapse_report,
    build_collapse_report,
)
from bt_studio.utils.io import atomic_save_json

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
    """Narrow-band continuous sampling (spread 0.17 < 0.4) + group step."""
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


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------

def test_verdict_healthy_plateau():
    df, bounds = _healthy_df()
    v = detect_space_collapse(df, "metrics_score", bounds)["overall_verdict"]
    assert v == "healthy_plateau", f"expected healthy_plateau, got {v}"


def test_verdict_collapsed():
    df, bounds = _collapsed_df()
    v = detect_space_collapse(df, "metrics_score", bounds)["overall_verdict"]
    assert v == "collapsed", f"expected collapsed, got {v}"


def test_verdict_isolated_spike():
    df, bounds = _spike_df()
    v = detect_space_collapse(df, "metrics_score", bounds)["overall_verdict"]
    assert v == "isolated_spike", f"expected isolated_spike, got {v}"


def test_verdict_flat_landscape():
    df, bounds = _flat_df()
    v = detect_space_collapse(df, "metrics_score", bounds)["overall_verdict"]
    assert v == "flat_landscape", f"expected flat_landscape, got {v}"


# ---------------------------------------------------------------------------
# build_search_bounds
# ---------------------------------------------------------------------------

def test_build_search_bounds():
    sb = build_search_bounds({
        "downsample": [3, 4, 5], "motif_minutes": [30, 45, 60, 90],
        "threshold_r": {"type": "uniform", "low": 0.65, "high": 0.85}, "num_trials": 50,
    })
    assert sb["downsample"] == [3, 4, 5], sb
    assert sb["motif_minutes"] == [30, 45, 60, 90], sb
    assert sb["threshold_r"] == [0.65, 0.85], sb
    assert "num_trials" not in sb


def test_build_search_bounds_accepts_normalized_spec():
    """build_exp_config 的 normalized search_spec 也必须能转成 collapse bounds。"""
    sb = build_search_bounds({
        "downsample": {"type": "choice", "values": [3, 4, 5]},
        "threshold_r": {"type": "uniform", "low": 0.65, "high": 0.85},
        "num_trials": {"type": "choice", "values": [2]},
    })
    assert sb["downsample"] == [3, 4, 5], sb
    assert sb["threshold_r"] == [0.65, 0.85], sb
    assert "num_trials" not in sb


# ---------------------------------------------------------------------------
# report roundtrip
# ---------------------------------------------------------------------------

def test_collapse_report_json_roundtrip():
    df, bounds = _healthy_df()
    diag = detect_space_collapse(df, "metrics_score", bounds)
    report = build_collapse_report("fsm_hpo_test_202412", "test_feat", 202412,
                                   diag, bounds)
    assert report["verdict"] == "healthy_plateau"
    assert report["feature_col"] == "test_feat"
    assert report["model_id"] == 202412
    assert "downsample" in report["diagnostics"]

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "collapse_202412_test_feat.json")
        atomic_save_json(report, path)
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["verdict"] == report["verdict"]
        assert loaded["diagnostics"] == report["diagnostics"]


def test_print_collapse_report_renders_verdict():
    df, bounds = _healthy_df()
    diag = detect_space_collapse(df, "metrics_score", bounds)
    text = print_collapse_report(diag)
    assert "healthy_plateau".upper() in text.upper()
