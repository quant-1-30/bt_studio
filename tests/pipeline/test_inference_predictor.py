#!/usr/bin/env python3
"""pipeline.inference (FSMPredictor) 回归测试。

覆盖 2026-09 代码审计定位的逻辑 bug:
    L   FSMPredictor 依赖运行时 T1_rets, 配置漂移即 KeyError
"""

import numpy as np

from bt_studio.pipeline.inference import FSMPredictor


def test_fsm_predictor_pins_ckpt_t1_rets():
    ckpt = {
        "config": {"downsample": 3, "motif_minutes": 30, "m": 10,
                   "threshold_r": 0.7, "threshold_d": 2.4},
        "motif": np.zeros((1, 10)),
        "fsm_matrix": {
            "P(exit_5m|Macro)": [[0.2] * 5] * 3,
            "state_return_vectors": {"exit_5m": [0.1] * 5},
        },
        "T1_rets": {"exit_5m": 5},
        "decay_minutes": 15,
    }
    common = {
        "T1_rets": {"exit_30m": 30},  # 运行时配置与训练不一致
        "decay_minutes": 30,
        "dtw_window_frac": 0.1,
        "ranking_window": 5,
    }
    pred = FSMPredictor(ckpt, common)  # 旧实现按运行时 key 取矩阵 -> KeyError
    assert pred.target_names == ["exit_5m"]


def test_fsm_predictor_falls_back_to_runtime_cfg():
    ckpt = {
        "config": {"downsample": 3, "motif_minutes": 30, "m": 10,
                   "threshold_r": 0.7, "threshold_d": 2.4},
        "motif": np.zeros((1, 10)),
        "fsm_matrix": {
            "P(exit_5m|Macro)": [[0.2] * 5] * 3,
            "state_return_vectors": {"exit_5m": [0.1] * 5},
        },
        # 旧版 ckpt 无 T1_rets -> 回落运行时配置
    }
    common = {"T1_rets": {"exit_5m": 5}, "decay_minutes": 15,
              "dtw_window_frac": 0.1, "ranking_window": 5}
    pred = FSMPredictor(ckpt, common)
    assert pred.target_names == ["exit_5m"]


def test_fsm_predictor_fails_fast_on_missing_state_return_vectors():
    """N2: FSMPredictor must fail-fast with ValueError if state_return_vectors is missing or incomplete."""
    import pytest

    bad_ckpt = {
        "config": {"downsample": 3, "motif_minutes": 30, "m": 10, "threshold_r": 0.7, "threshold_d": 2.4},
        "motif": np.zeros((1, 10)),
        "fsm_matrix": {
            "P(exit_5m|Macro)": [[0.2] * 5] * 3,
            # missing state_return_vectors
        },
        "T1_rets": {"exit_5m": 5},
    }
    with pytest.raises(ValueError, match="state_return_vectors"):
        FSMPredictor(bad_ckpt, {"T1_rets": {"exit_5m": 5}, "decay_minutes": 15})


def test_dtw_subseq_objective_match_and_fsm_sizer():
    """Verify objective position-invariant subsequence DTW matching and FSMSizer risk-parity allocation."""
    from bt_studio.pipeline.patterns.astc import calc_min_subseq_dtw
    from bt_studio.plugins.backtest.sizers import FSMSizer

    # 1. Objective subsequence DTW: identical waveform at early vs late window yields identical distance
    m = 10
    motif = np.sin(np.linspace(0, 2 * np.pi, m))
    z_motif = (motif - motif.mean()) / motif.std()
    noise = np.random.RandomState(1).normal(0, 0.05, m)

    curve_early = np.random.RandomState(0).normal(0, 0.1, 60)
    curve_early[2 : 2 + m] = motif + noise

    curve_late = np.random.RandomState(0).normal(0, 0.1, 60)
    curve_late[48 : 48 + m] = motif + noise

    d_early = calc_min_subseq_dtw(curve_early, z_motif, 3, 5.0)
    d_late = calc_min_subseq_dtw(curve_late, z_motif, 3, 5.0)
    assert abs(d_late - d_early) < 1e-12

    # 2. FSMSizer: 波动尺度 sid_scale=0.04 的高波标的与 sid_scale=0.02 的低波标的同分时, 低波标的获得 2 倍风险平价权重
    sizer = FSMSizer(stake=0.90, target_expected_ret=0.003, max_single_weight=0.80, vol_parity=True)
    topk_info = {
        b"600000": {"fsm_score": 1.0, "expected_return": 0.003, "sid_scale": 0.02},
        b"688001": {"fsm_score": 1.0, "expected_return": 0.003, "sid_scale": 0.04},
    }
    alloc = sizer._getsizing(topk_info, snapshot=None, isbuy=True)
    assert alloc[b"600000"] > alloc[b"688001"]
    assert abs(alloc[b"600000"] / alloc[b"688001"] - 2.0) < 1e-4

    # 宏观期望 <= 0 时自动空仓防守
    neg_info = {b"600000": {"fsm_score": 1.0, "expected_return": -0.001}}
    assert sizer._getsizing(neg_info, snapshot=None, isbuy=True)[b"600000"] == 0.0
