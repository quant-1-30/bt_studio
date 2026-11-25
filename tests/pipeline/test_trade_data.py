#! /usr/bin/env python3

import datetime
from array import array
from types import SimpleNamespace as NS

import pytest

from bt_studio.trade import daily_hfq_scales, snapshot_to_bars


def test_snapshot_to_bars_shape_and_columns():
    snap = {
        "sid": [b"600000", b"600000"],
        "bar_idx": array("i", [0, 1]),
        "open": array("d", [10.0, 10.1]),
        "high": array("d", [10.2, 10.3]),
        "low": array("d", [9.9, 10.0]),
        "close": array("d", [10.1, 10.2]),
        "volume": array("d", [100.0, 200.0]),
        "amount": array("d", [1000.0, 2000.0]),
    }
    bars = snapshot_to_bars(snap, 20261119)
    assert bars.height == 2
    assert bars.columns == ["day", "sid", "bar_idx", "open", "high", "low", "close", "volume", "amount"]
    assert bars["day"][0] == datetime.date(2026, 11, 19)
    assert bars["sid"].to_list() == [b"600000", b"600000"]
    assert bars["bar_idx"].to_list() == [0, 1]
    assert bars["close"][0] == 10.1

    # also accepts a datetime.date directly
    bars2 = snapshot_to_bars(snap, datetime.date(2026, 11, 19))
    assert bars2["day"][0] == datetime.date(2026, 11, 19)


def test_daily_hfq_scales_matches_apply_factor():
    # 2-for-1 split on 20240604: raw factor r=2.0 -> scale = 1/2 = 0.5
    factors = {b"600000": NS(raw_factors={20240604: 2.0})}
    assert daily_hfq_scales(factors, 20240605) == {"600000": 0.5}
    # before the ex-date the scale is identity
    assert daily_hfq_scales(factors, 20240603) == {"600000": 1.0}

    # multiple ex-dates accumulate multiplicatively
    multi = {b"600000": NS(raw_factors={20240110: 2.0, 20240210: 4.0})}
    assert daily_hfq_scales(multi, 20240301)["600000"] == 1.0 / (2.0 * 4.0)


def test_daily_hfq_scales_rejects_non_factor_result():
    # 传入裸 dict(会被误当 adj_factors)必须显式报错, 防止过度/不足复权
    with pytest.raises(TypeError, match="raw_factors"):
        daily_hfq_scales({b"600000": {20240604: 2.0}}, 20240605)
