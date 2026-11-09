#!/usr/bin/env python3
"""utils.math_func 输入防御回归。"""

import math

from bt_studio.utils.math_func import derive_m_threshold_d


def test_derive_m_threshold_d_clamps_invalid_inputs():
    # downsample=0 不再除零; threshold_r>1 不再让 sqrt(负数) 产 NaN
    m, threshold_d = derive_m_threshold_d(60, 0, 1.2)
    assert m == 60
    assert math.isfinite(threshold_d) and threshold_d > 0.0

    # 负 threshold_r 按 0 处理: threshold_d = sqrt(2m)
    m2, threshold_d2 = derive_m_threshold_d(60, 4, -0.5)
    assert m2 == 15
    assert threshold_d2 == math.sqrt(2 * 15)
