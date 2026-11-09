#!/usr/bin/env python3
"""pipeline.preprocess 回归测试: 面板构建与股票池过滤。

覆盖 2026-09 代码审计定位的逻辑 bug:
    L   14:50 入场价缺 bar 时退化当日收盘价的路径未实现 (内过滤丢行)
    L   active_ratio 分母不含 null-volume 行 → 停牌门控失效
"""

import datetime

import numpy as np
import polars as pl
import pytest

from bt_studio.pipeline.preprocess import universe_sample
from bt_studio.pipeline.preprocess.panel import build_static_panel


# ---------------------------------------------------------------------------
# L: 入场价缺 bar 时退化当日收盘价
# ---------------------------------------------------------------------------

ENTRY_CC = {
    "eps": 1e-8,
    "exclude_bars": 10,
    "T1_rets": {"exit_5m": 5},
}


def test_entry_price_falls_back_to_daily_close():
    days = [datetime.date(2024, 1, 2), datetime.date(2024, 1, 3),
            datetime.date(2024, 1, 4)]
    hf_parts, daily_rows = [], []
    for sid, base in (("600000", 100.0), ("600001", 200.0)):
        for di, day in enumerate(days):
            close_arr = np.full(240, base + di)  # 平价, 便于手算期望
            hf_parts.append(pl.DataFrame({
                "day": [day] * 240, "sid": [sid] * 240,
                "bar_idx": np.arange(240, dtype=np.int32),
                "open": close_arr, "high": close_arr * 1.001,
                "low": close_arr * 0.999, "close": close_arr,
                "volume": np.full(240, 10.0), "amount": close_arr * 10.0,
            }))
            daily_rows.append({"day": day, "sid": sid,
                               "close": base + di + 0.5,  # 日线收盘与 bar 锚不同
                               "high": base + 1, "low": base - 1,
                               "volume": 100.0, "amount": 1000.0})
    hf = pl.concat(hf_parts)
    daily = pl.DataFrame(daily_rows)

    # sid=600000 的 day0 缺 14:50 bar (bar_idx=230)
    hole = hf.filter(~((pl.col("sid") == "600000")
                       & (pl.col("day") == days[0])
                       & (pl.col("bar_idx") == 230)))
    static = build_static_panel(hole.lazy(), daily.lazy(), ENTRY_CC,
                                is_train=True).collect()

    row = static.filter((pl.col("sid") == "600000") & (pl.col("day") == days[0]))
    assert row.height == 1, "缺 14:50 bar 的样本必须退化用日线收盘, 而不是整行消失"
    # entry = 日线收盘 100.5; T+1 (day1) exit_5m close = bar4 close = 101
    expected_raw = 101.0 / 100.5 - 1.0
    assert row["raw_exit_5m"][0] == pytest.approx(expected_raw, abs=1e-9)


# ---------------------------------------------------------------------------
# L: active_ratio 分母计入停牌 (volume=null) 行
# ---------------------------------------------------------------------------

def test_active_ratio_counts_suspended_null_volume_rows():
    days = [20240101 + i for i in range(10)] + [20240201 + i for i in range(5)]
    rows = []
    for i, d in enumerate(days):
        # 2 月为交易日 (1 月样本 PIT 生效的月份)
        vol_good = 100.0
        vol_bad = None if i < 3 else 100.0  # 1 月停牌 3/10 天 -> ratio 0.7
        rows.append({"sid": "600000", "day": d, "close": 10.0 + i,
                     "high": 10.5, "low": 9.5, "volume": vol_good, "amount": 1000.0})
        rows.append({"sid": "600001", "day": d, "close": 20.0 + i,
                     "high": 20.5, "low": 19.5, "volume": vol_bad, "amount": 1000.0})
    daily = pl.LazyFrame(rows)
    uni = pl.LazyFrame({"sid": ["600000", "600001"],
                        "first_trading": [20200101, 20200101]})

    out = universe_sample(uni, daily, {"days_since_ipo": 0, "eps": 1e-8})
    sids = set(out.collect()["sid"].to_list())
    assert "600000" in sids
    assert "600001" not in sids, "停牌 30% 的标的必须被 active_ratio>=0.90 剔除"


def test_align_skeleton_produces_full_240_bars():
    """align_skeleton aligns raw ticks onto a 240-bar skeleton without strategy-specific exclude_bars parameters."""
    from bt_studio.pipeline.preprocess.prepare import align_skeleton

    raw_ticks = pl.DataFrame({
        "sid": [b"600000", b"600000"],
        "tick": [1704159060, 1704159120],
        "open": [10.0, 10.1],
        "high": [10.2, 10.3],
        "low": [9.9, 10.0],
        "close": [10.1, 10.2],
        "volume": [100.0, 100.0],
        "amount": [1010.0, 1020.0],
    })
    aligned = align_skeleton(raw_ticks.lazy()).collect()
    assert aligned.height == 240
    assert set(aligned["sid"].to_list()) == {b"600000"}

    # R1: Zero pre-entry volume day in build_static_panel produces null raw_* label (does not dilute uncond baseline)
    days = [datetime.date(2024, 1, 2), datetime.date(2024, 1, 3)]
    hf_zero_vol = pl.concat([
        pl.DataFrame({
            "day": [days[0]] * 240, "sid": ["600000"] * 240,
            "bar_idx": np.arange(240, dtype=np.int32),
            "open": np.full(240, 100.0), "high": np.full(240, 100.0),
            "low": np.full(240, 100.0), "close": np.full(240, 100.0),
            "volume": np.zeros(240), "amount": np.zeros(240),
        }),
        pl.DataFrame({
            "day": [days[1]] * 240, "sid": ["600000"] * 240,
            "bar_idx": np.arange(240, dtype=np.int32),
            "open": np.full(240, 101.0), "high": np.full(240, 102.0),
            "low": np.full(240, 100.0), "close": np.full(240, 101.0),
            "volume": np.full(240, 10.0), "amount": np.full(240, 1010.0),
        }),
    ])
    daily_zv = pl.DataFrame({"day": days, "sid": ["600000", "600000"], "close": [100.0, 101.0]})
    static_zv = build_static_panel(hf_zero_vol.lazy(), daily_zv.lazy(), ENTRY_CC, is_train=True).collect()
    assert static_zv["raw_exit_5m"][0] is None


def test_causal_boundary_no_oos_label_leak_and_no_fake_null_rank():
    """Verify: (1) is_train=True drops last feat day even if daily_lf has OOS dates;
    (2) is_train=False keeps last feat day for live inference;
    (3) prepare_fsm_base_df keeps rank_* as None when raw_* is null;
    (4) compute_macro_states truncates exclude_bars dynamically without downsample in common_config."""
    from bt_studio.pipeline.preprocess.macro import compute_macro_states
    from bt_studio.pipeline.patterns.fsm import prepare_fsm_base_df

    days_feat = [datetime.date(2024, 1, 2), datetime.date(2024, 1, 3)]
    days_daily = days_feat + [datetime.date(2024, 1, 4)]  # daily_lf includes OOS day
    hf_parts = [
        pl.DataFrame({
            "day": [d] * 240, "sid": ["600000"] * 240,
            "bar_idx": np.arange(240, dtype=np.int32),
            "open": np.full(240, 100.0), "high": np.full(240, 101.0),
            "low": np.full(240, 99.0), "close": np.full(240, 100.0),
            "volume": np.full(240, 10.0), "amount": np.full(240, 1000.0),
        })
        for d in days_feat
    ]
    daily = pl.DataFrame({
        "day": days_daily, "sid": ["600000"] * 3,
        "close": [100.0, 101.0, 102.0], "high": [101.0, 102.0, 103.0], "low": [99.0, 100.0, 101.0],
    })

    # (1) is_train=True must NOT include days_feat[-1] (whose T+1 is in OOS)
    train_static = build_static_panel(pl.concat(hf_parts).lazy(), daily.lazy(), ENTRY_CC, is_train=True).collect()
    assert train_static["day"].to_list() == [days_feat[0]]

    # (2) is_train=False preserves days_feat[-1] for live OOS inference
    oos_static = build_static_panel(pl.concat(hf_parts).lazy(), daily.lazy(), ENTRY_CC, is_train=False).collect()
    assert set(oos_static["day"].to_list()) == set(days_feat)

    # (3) Null raw_* must yield null rank_* and null state_* (not 0.5 / state 2)
    panel_with_null = pl.DataFrame({
        "day": [days_feat[0], days_feat[1]],
        "sid": [b"600000", b"600000"],
        "lag_0": [[1.0] * 48, [1.0] * 48],
        "raw_exit_5m": [0.01, None],
        "intra_exit_5m": [0.005, None],
    })
    base_df = prepare_fsm_base_df(panel_with_null, {**ENTRY_CC, "ranking_window": 2, "decay_minutes": 15})
    null_row = base_df.filter(pl.col("day") == days_feat[1])
    assert null_row["rank_exit_5m"][0] is None
    assert null_row["state_exit_5m"][0] is None

    # (4) compute_macro_states shifts daily_ofi_median by 1 day: perturbing day T's lag_0 never affects day <= T macro_state
    macro_panel = pl.DataFrame({
        "day": [datetime.date(2024, 1, 2), datetime.date(2024, 1, 3), datetime.date(2024, 1, 4)],
        "sid": [b"600000"] * 3,
        "lag_0": [[1.0] * 48, [1.0] * 48, [99999.0] * 48],
    })
    macro_clean = pl.DataFrame({
        "day": [datetime.date(2024, 1, 2), datetime.date(2024, 1, 3), datetime.date(2024, 1, 4)],
        "sid": [b"600000"] * 3,
        "lag_0": [[1.0] * 48, [1.0] * 48, [1.0] * 48],
    })
    cc = {"ranking_window": 2}
    assert compute_macro_states(macro_panel, cc).equals(compute_macro_states(macro_clean, cc))


def test_build_static_panel_causal_ex_dividend_adjustment():
    """Verify bt_sdk.core.factor.apply_factor (Hfq) in prepare_tick adjusts raw_* for T+1 splits while intra_* is invariant."""
    import bt_sdk.core.factor as f
    import adj_factor
    from bt_protocol.constant import FactorTopic
    from bt_sdk.core.factor import apply_factor

    days = [datetime.date(2024, 6, 3), datetime.date(2024, 6, 4)]
    # Day 0 (20240603): stock trades at 20.0. Overnight into Day 1 (20240604): 10-for-10 split (adj_factor = 2.0).
    # Day 1: opens at 10.0 (0% overnight return) and rises to 10.2 at bar 4 (+2.0% true holding & intraday return).
    hf_d0 = pl.DataFrame({
        "day": [20240603] * 240, "sid": [b"600000"] * 240,
        "bar_idx": np.arange(240, dtype=np.int32),
        "open": np.full(240, 20.0), "high": np.full(240, 20.0),
        "low": np.full(240, 20.0), "close": np.full(240, 20.0),
        "volume": np.full(240, 10.0), "amount": np.full(240, 200.0),
    })
    hf_d1 = pl.DataFrame({
        "day": [20240604] * 240, "sid": [b"600000"] * 240,
        "bar_idx": np.arange(240, dtype=np.int32),
        "open": np.full(240, 10.0), "high": np.full(240, 10.2),
        "low": np.full(240, 10.0), "close": np.full(240, 10.2),
        "volume": np.full(240, 20.0), "amount": np.full(240, 204.0),
    })
    res_hfq = adj_factor.calc_adjust_factors_columns(
        np.array([20240603, 20240604], dtype=np.int32),
        np.array([20.0, 10.2], dtype=np.float64),
        np.array([20240604], dtype=np.int32),
        np.array([10.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        f._EMPTY_I32, f._EMPTY_F64, f._EMPTY_F64,
        adj_factor.AdjustType.Backward,
    )
    hf_adj = apply_factor({b"600000": pl.concat([hf_d0, hf_d1])}, {b"600000": res_hfq}, FactorTopic.Hfq)[b"600000"]
    daily = pl.DataFrame({"day": days, "sid": ["600000", "600000"], "close": [20.0, 10.2]})

    static_ex = build_static_panel(hf_adj.lazy(), daily.lazy(), ENTRY_CC, is_train=True).collect()
    assert static_ex["raw_exit_5m"][0] == pytest.approx(0.02, abs=1e-9)
    assert static_ex["intra_exit_5m"][0] == pytest.approx(0.02, abs=1e-9)


