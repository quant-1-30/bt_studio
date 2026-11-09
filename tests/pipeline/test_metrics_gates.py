#!/usr/bin/env python3
"""pipeline.metrics / 统计门控专项单测 (P0 统计修正):

1. score.py: Jeffreys 平滑 win_rate 有界 + 零样本硬罚 (-500);
2. score.py: -ln(u_pval) 软饱和 (严格单调、增量有界、极小 p 趋于饱和);
3. _stats.py: _daily_paired_pval 按日配对符号秩检验 (有效样本量 = 天数);
4. _stats.py: _raw_excess_gate 原始收益经济阈值硬门控 (标量 / 按目标分档)。

注: 2026-09 重构后 calculate_hpo_score 契约为 7 参数
(u_pval, trigger_count, cond_rets, uncond_rets, cond_momentum,
 tune_config, common_config) —— z_gap 可成交子集过滤不在该层实现,
本文件按当前实际接口与行为断言。
"""

import numpy as np
import polars as pl

from bt_studio.pipeline.metrics import calculate_hpo_score
from bt_studio.pipeline.metrics.gates import daily_paired_pval, raw_excess_gate


def _common_cfg(**over):
    cfg = {
        "alternative": "greater",
        "eps": 1e-8,
        "T1_rets": {"exit_5m": 5, "exit_30m": 30},
        "dtw_window_frac": 0.1,
        # max_neg_ln_p / min_excess_ret 缺省走代码默认, 显式给出便于断言
        "max_neg_ln_p": 10.0,
    }
    cfg.update(over)
    return cfg


TUNE_CFG = {"motif_minutes": 30}


# ---------------------------------------------------------------------------
# 1. 打分函数基础契约
# ---------------------------------------------------------------------------

def _score(u_pval, n, rets, uncond, momentum, cfg=None, tune_cfg=None):
    return calculate_hpo_score(
        u_pval, n, rets, uncond, momentum,
        tune_cfg or TUNE_CFG, cfg or _common_cfg())


def test_score_rewards_positive_excess_over_negative():
    """'greater' 备择下, 正超额 (中位数差为正) 的分数应高于负超额。"""
    pos = _score(1e-4, 20, np.full(20, 0.05), np.zeros(30), np.full(20, 0.03))
    neg = _score(1e-4, 20, np.full(20, -0.05), np.zeros(30), np.full(20, 0.03))
    assert pos > neg


def test_score_jeffreys_smoothing_bounds_win_rate():
    """全胜的 win_rate 不取 1 极端值 (ln 空间有界), 分数有限且量级合理。"""
    s_all_win = _score(
        0.5, 12, np.full(12, 0.05), np.zeros(30), np.full(12, 0.03))
    assert np.isfinite(s_all_win)
    assert -20.0 < s_all_win < 20.0


def test_score_zero_triggers_returns_floor():
    """trigger_count=0 或空收益 -> 固定 -500 重罚。"""
    assert _score(0.5, 0, np.array([0.05]), np.zeros(30), np.zeros(1)) == -500.0
    assert _score(0.5, 12, np.array([]), np.zeros(30), np.zeros(12)) == -500.0


def test_score_sample_size_attenuates_complexity_penalty():
    """复杂度罚项 ~ k/sqrt(n): 同配置下样本更多 -> 分数更高。"""
    small = _score(1e-4, 8, np.full(8, 0.05), np.zeros(30), np.full(8, 0.03))
    large = _score(1e-4, 200, np.full(200, 0.05), np.zeros(30), np.full(200, 0.03))
    assert large > small


# ---------------------------------------------------------------------------
# 2. p 值截断
# ---------------------------------------------------------------------------

def test_score_pval_term_soft_saturated():
    """p 值项软饱和: 严格单调 (不并列)、增量有界、极小 p 趋于饱和。"""
    rets = np.full(30, 0.05)
    momentum = np.full(30, 0.03)
    uncond = np.zeros(50)

    def _s(p):
        return _score(p, 30, rets, uncond, momentum)

    # 严格单调: 更小的 p 严格加分 (硬截断会让 1e-12 与 1e-10 并列,
    # 并列导致塌陷检测 IQR 压缩 -> 门控误报)
    assert _s(1e-12) > _s(1e-10) > _s(1e-3)

    # 有界: p 从 1e-3 压到 1e-12, 总增量不超过 max_neg_ln_p
    assert (_s(1e-12) - _s(1e-3)) < 10.0

    # 饱和: 极小 p 区间增量趋零
    assert (_s(1e-300) - _s(1e-40)) < 0.1


# ---------------------------------------------------------------------------
# 3. 按日配对检验
# ---------------------------------------------------------------------------

def _mk_daily_frames(diffs, n_cond=3, n_uncond=3):
    """按 diffs 逐日构造 triggers / complementary: 每日 cond 均值秩 =
    0.5 + diff, uncond 均值秩 = 0.5 (rank_trajectory 直接给均值)。"""
    trig_rows, uncond_rows = [], []
    for i, d in enumerate(diffs):
        cond_base, uncond_base = 0.5 + d, 0.5
        for k in range(n_cond):
            trig_rows.append({"day": i, "sid": f"s{k}", "rank_trajectory": cond_base})
        for k in range(n_uncond):
            uncond_rows.append({"day": i, "sid": f"u{k}", "rank_trajectory": uncond_base})
    return pl.DataFrame(trig_rows), pl.DataFrame(uncond_rows)


def test_day_paired_pval_strong_signal_small_p():
    trig, uncond = _mk_daily_frames([0.2] * 20)
    p = daily_paired_pval(trig, uncond, "greater")
    assert p < 1e-3


def test_day_paired_pval_noise_large_p():
    rng = np.random.RandomState(7)
    diffs = rng.choice([-0.2, 0.2], size=20)   # 对称噪声
    trig, uncond = _mk_daily_frames(diffs)
    p = daily_paired_pval(trig, uncond, "greater")
    assert p > 0.1


def test_day_paired_pval_few_days_conservative():
    """n_days < 3 -> 保守返回 1.0; 3~5 天小样本走精确分布 (无悬崖)。"""
    trig, uncond = _mk_daily_frames([0.2] * 1)
    assert daily_paired_pval(trig, uncond, "greater") == 1.0

    # 3 天全正 -> 精确 p = 2^-3 (弱证据的诚实定价, 而非人为 guard 跳变到 1.0)
    trig3, uncond3 = _mk_daily_frames([0.2] * 3)
    assert abs(daily_paired_pval(trig3, uncond3, "greater") - 0.125) < 1e-12


def test_day_paired_pval_ignores_row_level_inflation():
    """同日加密度 (行数膨胀) 不改变 p —— 有效样本量是天数不是行数。"""
    trig_small, uncond_small = _mk_daily_frames([0.2] * 20, n_cond=3, n_uncond=3)
    trig_dense, uncond_dense = _mk_daily_frames([0.2] * 20, n_cond=300, n_uncond=300)
    p_small = daily_paired_pval(trig_small, uncond_small, "greater")
    p_dense = daily_paired_pval(trig_dense, uncond_dense, "greater")
    assert abs(p_small - p_dense) < 1e-12


# ---------------------------------------------------------------------------
# 4. 原始收益经济阈值硬门控
# ---------------------------------------------------------------------------

def _mk_gate_frames(cond_raws, uncond_raws, z_gap=0.0):
    trig = pl.DataFrame({
        "day": list(range(len(cond_raws))),
        "z_gap": [z_gap] * len(cond_raws),
        "raw_trajectory": cond_raws,
        "raw_exit_5m": cond_raws,
        "raw_exit_30m": [r * 2.0 for r in cond_raws],
    })
    uncond = pl.DataFrame({
        "day": list(range(len(uncond_raws))),
        "raw_trajectory": uncond_raws,
        "raw_exit_5m": uncond_raws,
        "raw_exit_30m": [r * 2.0 for r in uncond_raws],
    })
    return trig, uncond


def test_raw_excess_gate_scalar_pass_and_fail():
    trig, uncond = _mk_gate_frames([0.03] * 20, [0.0] * 50)
    ok, reason = raw_excess_gate(trig, uncond, _common_cfg(min_excess_ret=0.02))
    assert ok

    fail, reason = raw_excess_gate(trig, uncond, _common_cfg(min_excess_ret=0.05))
    assert not fail
    assert "min_excess_ret" in reason and "5.0000%" in reason


def test_raw_excess_gate_per_target_dict():
    """分档 dict: exit_5m 超额 3% 达标、exit_30m 超额 6% 不达标 -> 失败。"""
    trig, uncond = _mk_gate_frames([0.03] * 20, [0.0] * 50)
    fail, reason = raw_excess_gate(
        trig, uncond, _common_cfg(min_excess_ret={"exit_5m": 0.02, "exit_30m": 0.10}))
    assert not fail
    assert "exit_30m" in reason


def test_raw_excess_gate_disabled_by_default():
    trig, uncond = _mk_gate_frames([0.0] * 10, [0.0] * 10)
    ok, reason = raw_excess_gate(trig, uncond, _common_cfg())
    assert ok and reason == ""
    ok, _ = raw_excess_gate(trig, uncond, _common_cfg(min_excess_ret=0.0))
    assert ok

