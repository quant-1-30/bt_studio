#!/usr/bin/env python3

import math

from bt_studio.utils.math_func import derive_m_threshold_d, entry_bar_clock, entry_bar_idx, limit_pct


def test_entry_bar_clock_maps_end_labeled_bar_to_clock():
    # 240-bar END mapping: 09:31->0 .. 11:30->119 | 13:01->120 .. 15:00->239
    assert entry_bar_idx({"bars_per_day": 240, "exclude_bars": 10}) == 230
    assert entry_bar_clock({"bars_per_day": 240, "exclude_bars": 10}) == (14, 51)
    assert entry_bar_clock({"bars_per_day": 240, "exclude_bars": 0}) == (15, 0)
    assert entry_bar_clock({"bars_per_day": 240, "exclude_bars": 240}) == (9, 31)


def test_derive_m_threshold_d_clamps_invalid_inputs():
    m, threshold_d = derive_m_threshold_d(60, 0, 1.2)
    assert m == 60
    assert math.isfinite(threshold_d) and threshold_d > 0.0

    m2, threshold_d2 = derive_m_threshold_d(60, 4, -0.5)
    assert m2 == 15
    assert threshold_d2 == math.sqrt(2 * 15)


def test_limit_pct_band_by_board():
    assert limit_pct("300750") == 0.195   # ChiNext (20% band)
    assert limit_pct("688111") == 0.195   # STAR
    assert limit_pct("000001") == 0.098   # main board
    assert limit_pct("600519") == 0.098   # main board
    assert limit_pct("830799") == 0.295   # BSE (30% band)
