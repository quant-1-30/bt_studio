#!/usr/bin/env python3

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
        "T1_rets": {"exit_30m": 30}, 
        "decay_minutes": 30,
        "dtw_window_frac": 0.1,
        "ranking_window": 5,
    }
    pred = FSMPredictor(ckpt, common)  
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
    from bt_studio.trade import FSMSizer

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

    # 2. FSMSizer: pure fsm_score conviction weighting + tranche exit weights (zero sid_scale coupling)
    sizer = FSMSizer(stake=0.90, target_expected_ret=0.003, max_single_weight=0.80, exit_weights=[0.5, 0.3, 0.2])
    topk_info = {
        b"600000": {"fsm_score": 2.0, "expected_return": 0.003},
        b"688001": {"fsm_score": 1.0, "expected_return": 0.003},
    }
    alloc = sizer._getsizing(topk_info, snapshot=None, isbuy=True)
    assert alloc[b"600000"] > alloc[b"688001"]
    assert abs(alloc[b"600000"] / alloc[b"688001"] - 2.0) < 1e-4

    # Tranche exit ratios: stage 0 sells 0.5/1.0=0.5, stage 1 sells 0.3/0.5=0.6 of remaining, stage 2 sells 1.0 of remaining
    from types import SimpleNamespace as NS
    dummy_snap = NS(positions=[NS(sid=b"600000", size=1000)])
    sizer.set_exit_stage(0)
    assert abs(sizer._getsizing({}, dummy_snap, isbuy=False)[b"600000"] - 0.5) < 1e-6
    sizer.set_exit_stage(1)
    assert abs(sizer._getsizing({}, dummy_snap, isbuy=False)[b"600000"] - 0.6) < 1e-6
    sizer.set_exit_stage(2)
    assert abs(sizer._getsizing({}, dummy_snap, isbuy=False)[b"600000"] - 1.0) < 1e-6

    neg_info = {b"600000": {"fsm_score": 1.0, "expected_return": -0.001}}
    assert sizer._getsizing(neg_info, snapshot=None, isbuy=True)[b"600000"] == 0.0


def test_fsm_sizer_max_single_weight_is_a_sparse_topk_concentration_cap():
    from bt_studio.trade import FSMSizer

    def alloc(n: int) -> dict:
        info = {f"{i:06d}".encode(): {"fsm_score": 1.0, "expected_return": 0.003} for i in range(n)}
        return FSMSizer(stake=0.90, max_single_weight=0.5)._getsizing(info, snapshot=None, isbuy=True)

    # topk=1
    a1 = alloc(1)
    assert a1[b"000000"] == 0.5
    # topk=2
    a2 = alloc(2)
    assert all(abs(w - 0.45) < 1e-9 for w in a2.values())
    # topk=5
    a5 = alloc(5)
    assert all(abs(w - 0.18) < 1e-9 for w in a5.values())
    assert abs(sum(a5.values()) - 0.90) < 1e-9
