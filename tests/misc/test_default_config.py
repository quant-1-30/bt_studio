#!/usr/bin/env python3
"""bt_studio.default_config 回归测试: 顶层配置白名单与契约函数。"""

import pytest

from bt_studio.default_config import (
    build_exp_config,
    common_params_of,
    resolve_feature_col,
    search_spec_of,
)


def test_build_exp_config_rejects_unknown_top_level_keys():
    with pytest.raises(ValueError):
        build_exp_config({"invalid_flat_key": "x"})


def test_resolve_feature_col_priority():
    # explicit > common_params > recipe[-1].name
    assert resolve_feature_col({"feature_col": "cc"}, None, "explicit") == "explicit"
    assert resolve_feature_col({"feature_col": "cc"}) == "cc"
    recipe = [{"name": "last_feat", "ast": {"op": "abs", "args": [{"col": "x"}]}}]
    assert resolve_feature_col({}, recipe) == "last_feat"

    with pytest.raises(ValueError, match="feature_col"):
        resolve_feature_col({}, None)


def test_common_params_of_dual_key_names():
    assert common_params_of({"common_params": {"a": 1}}) == {"a": 1}
    assert common_params_of({"common_config": {"b": 2}}) == {"b": 2}
    assert common_params_of({}) == {}


def test_search_spec_of_falls_back_to_ssot():
    """无覆盖时回落 SSOT 基准对象; 传裸 dict 时原样返回 (归一化由
    build_exp_config 完成)。"""
    from bt_studio.default_config import PRODUCTION_SEARCH_SPEC

    assert search_spec_of({}) is PRODUCTION_SEARCH_SPEC
    # 裸 exp_config (未经 build_exp_config): 原样透传
    assert search_spec_of({"search_spec": {"downsample": {"type": "choice", "values": [3, 5]}}}) == {
        "downsample": {"type": "choice", "values": [3, 5]}}
    # 标准入口: 旧列表写法被归一化, 未覆盖参数回落基准
    spec = search_spec_of(build_exp_config({"search_spec": {"downsample": {"type": "choice", "values": [3, 5]}}}))
    assert spec["downsample"] == {"type": "choice", "values": [3, 5]}
    assert spec["threshold_r"] == PRODUCTION_SEARCH_SPEC["threshold_r"]


def test_search_spec_rejects_execution_control_keys():
    """num_trials 等调度预算只能放 common_params, 不得混入搜索空间。"""
    with pytest.raises(ValueError, match="common_params"):
        build_exp_config({"search_spec": {"num_trials": 2}})
    with pytest.raises(ValueError, match="common_params"):
        build_exp_config({"search_spec": {"num_workers": {"type": "choice", "values": [2]}}})
