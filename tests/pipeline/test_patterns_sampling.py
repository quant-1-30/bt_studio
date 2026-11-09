#!/usr/bin/env python3
"""pipeline.patterns / 采样回归测试。

覆盖 2026-09 代码审计定位的逻辑 bug:
    M12 均衡采样 50/50 未实现 (纯均匀抽样, 名实不符)
    L   autocorr 的 shift(1) 时空折叠 → 改为日级触发率自相关
以及 compute_macro_states 独立提取后的形状回归。
"""

import datetime

import numpy as np
import polars as pl
import pytest

from bt_studio.pipeline.preprocess.macro import compute_macro_states
from bt_studio.pipeline.patterns.pattern import _select_positive_motif_curves
from bt_studio.pipeline.metrics.gates import daily_trigger_autocorr


# ---------------------------------------------------------------------------
# Supervised Positive Motif Curve Selection (replaces legacy get_balanced_samples)
# ---------------------------------------------------------------------------

def test_select_positive_motif_curves_filters_outperformers():
    rng = np.random.RandomState(0)
    n_total, L = 60, 50
    curves = rng.normal(0.0, 0.1, (n_total, L))
    # First 15 rows are positive outperformers (rank >= 0.70, raw > 0)
    raw_traj = np.array([0.05] * 15 + [-0.02] * 45)
    rank_traj = np.array([0.85] * 15 + [0.30] * 45)
    base_df = pl.DataFrame({"raw_trajectory": raw_traj, "rank_trajectory": rank_traj})

    pos_curves = _select_positive_motif_curves(curves, base_df, m=5, common_config={"topk": 1})
    assert pos_curves.shape == (15, L)
    np.testing.assert_array_equal(pos_curves, curves[:15])


# ---------------------------------------------------------------------------
# L: autocorr 改为日级触发率自相关 (消除停牌"时空折叠")
# ---------------------------------------------------------------------------

def testdaily_trigger_autocorr():
    days = [datetime.date(2024, 1, i) for i in range(2, 7)]
    rows = []
    for d in days:
        for sid in ("600000", "600001"):
            rows.append({"day": d, "sid": sid, "is_trigger": 1})
        for sid in ["600099"]:
            rows.append({"day": d, "sid": sid, "is_trigger": 0})
    # 日级触发率 = 2/3 恒定 -> 自相关为 nan/0 而非行级拼接噪声
    df = pl.DataFrame(rows)
    val = daily_trigger_autocorr(df)
    assert np.isnan(val) or val == 0.0

    # 交替日: 全市场同日全触发/全不触发 -> 日级 rate 序列 [1,0,1,0,1], lag-1 corr = -1
    rows = []
    for i, d in enumerate(days):
        trig = 1 if i % 2 == 0 else 0
        for sid in ("600000", "600001"):
            rows.append({"day": d, "sid": sid, "is_trigger": trig})
    val2 = daily_trigger_autocorr(pl.DataFrame(rows))
    assert val2 == pytest.approx(-1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# compute_macro_states 独立提取的形状回归
# ---------------------------------------------------------------------------

def test_compute_macro_states_matches_legacy_shape():
    """独立提取的 compute_macro_states 应产出 (day, macro_state)。"""
    days = pl.Series([
        datetime.date(2024, 1, 2) + datetime.timedelta(days=i) for i in range(20)
    ])
    panel = pl.DataFrame({
        "day": days,
        "sid": ["600000"] * days.len(),
        "lag_0": [[0.1, 0.2, 0.3]] * days.len(),
    })
    out = compute_macro_states(panel, {"ranking_window": 5})
    assert set(out.columns) == {"day", "macro_state"}
    assert out.height == panel.height
    assert out["macro_state"].is_null().sum() == 0  # fill_null(1) 兜底


def test_get_candidate_motifs_excludes_paired_neighbor_twin():
    """Matrix Profile 排斥区必须同时屏蔽 anchor 与其配对近邻, 避免 Top-K 被双胞胎占满。"""
    from bt_studio.pipeline.patterns.astc import get_candidate_motifs

    rng = np.random.RandomState(42)
    m = 10
    # 构造两组完全独立的重复模式: 模式 A (正弦) 与 模式 B (锯齿)
    pat_a = np.sin(np.linspace(0, 2 * np.pi, m))
    pat_b = np.linspace(-1.0, 1.0, m) ** 2
    series = rng.normal(0, 2.0, 200)
    series[10:10 + m] = pat_a
    series[60:60 + m] = pat_a + rng.normal(0, 1e-4, m)   # A 的配对双胞胎
    series[120:120 + m] = pat_b
    series[170:170 + m] = pat_b + rng.normal(0, 1e-4, m)  # B 的配对双胞胎

    cands = get_candidate_motifs(series, {"m": m, "threshold_d": 5.0}, {"topk": 2})
    assert len(cands) == 2
    # 若未屏蔽配对近邻, cands[0] 与 cands[1] 都会是 pat_a 的双胞胎 (相关系数 ~1.0);
    # 屏蔽后, 第 2 个候选必然选到独立模式 pat_b!
    corr_01 = float(np.corrcoef(cands[0], cands[1])[0, 1])
    assert corr_01 < 0.8, f"Top-2 motifs 不应是同一对双胞胎 (corr={corr_01:.3f})"


def test_numba_dtw_and_lb_keogh_exact_equivalence_with_dtaidistance():
    """Verify _banded_dtw_sq, _compute_keogh_envelope, and calc_min_subseq_dtw match dtaidistance across boundaries."""
    import math
    from dtaidistance import dtw
    from bt_studio.pipeline.patterns.astc import (
        _banded_dtw_sq,
        _compute_keogh_envelope,
        calc_min_subseq_dtw,
    )

    rng = np.random.default_rng(2026)
    for w in (0, 1, 2, 3, 5, 10):
        for r, c in ((10, 10), (8, 12), (14, 9)):
            s1 = rng.standard_normal(r)
            s2 = rng.standard_normal(c)
            d_ref = dtw.distance(s1, s2, window=None if w <= 0 else w)
            d_nb = math.sqrt(_banded_dtw_sq(s1, s2, w, np.inf))
            assert d_nb == pytest.approx(d_ref, rel=1e-12, abs=1e-12)

        # LB_Keogh envelope equivalence on equal-length series
        s1_eq = rng.standard_normal(12)
        s2_eq = rng.standard_normal(12)
        u, l = _compute_keogh_envelope(s2_eq, w)
        lb_nb = math.sqrt(float(np.sum(np.where(s1_eq > u, (s1_eq - u) ** 2, np.where(s1_eq < l, (l - s1_eq) ** 2, 0.0)))))
        lb_ref = dtw.lb_keogh(s1_eq, s2_eq, window=None if w <= 0 else w)
        assert lb_nb == pytest.approx(lb_ref, rel=1e-12, abs=1e-12)

    # Boundary: negative threshold_d must return inf immediately
    curve = rng.standard_normal(40)
    motif = rng.standard_normal(8)
    z_motif = (motif - motif.mean()) / motif.std()
    assert np.isinf(calc_min_subseq_dtw(curve, z_motif, dtw_w=3, threshold_d=-0.1))

