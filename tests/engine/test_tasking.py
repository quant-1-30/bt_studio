#!/usr/bin/env python3
"""engine.tasking 回归测试: task_id 内容寻址与产物目录分桶。

(2026-09 迁自 utils/tasking.py —— 模块现位于 bt_studio.engine.tasking)
"""

import os

from bt_studio.engine.executor import attach_task_dirs, resolve_task_id
from bt_studio.default_config import build_exp_config


def test_task_id_content_addressed_isolation():
    """task_id 解析与产物目录分桶。"""
    ast_a = {"op": "ref", "args": [{"col": "close"}, 2]}
    ast_b = {"op": "mean", "args": [{"col": "volume"}, 10]}

    # 内容寻址: 同 (feature, ast, 日期, 基准) 幂等; 不同 AST 隔离
    t1 = resolve_task_id(build_exp_config({"feature_col": "sig"}), ast_a)
    t2 = resolve_task_id(build_exp_config({"feature_col": "sig"}), ast_b)
    assert t1 and t2 and t1 != t2 and t1.startswith("sig-")
    assert t1 == resolve_task_id(build_exp_config({"feature_col": "sig"}), ast_a)

    # 显式优先级: task_id / run_name 覆盖内容寻址
    assert resolve_task_id(build_exp_config({"run_name": "manual run"}), ast_a) == "manual-run"
    assert resolve_task_id(build_exp_config({"task_id": "exp#1"}), ast_a) == "exp-1"

    # 无任务身份 (无 feature_col) -> None -> 旧扁平布局
    assert resolve_task_id(build_exp_config({}), ast_a) is None


def test_attach_task_dirs_injects_derived_dirs():
    ast_a = {"op": "ref", "args": [{"col": "close"}, 2]}
    e1 = build_exp_config({"feature_col": "sig"})
    tid = attach_task_dirs(e1, ast_a)
    cc = e1["common_params"]
    assert cc["_task_id"] == tid
    assert cc["feature_dir"].endswith(tid)
    assert cc["tune_model_dir"] == os.path.join(cc["tune_base_dir"], "models")
    assert cc["tune_score_dir"] == os.path.join(cc["tune_base_dir"], "scores")
    assert cc["tune_collapse_dir"] == os.path.join(cc["tune_base_dir"], "collapse")
    assert tid in cc["tune_base_dir"]


def test_task_id_uses_common_config_dates_too():
    """[键名兼容] common_config 形状的日期也必须参与分桶哈希。"""
    ast = {"op": "abs", "args": [{"col": "close"}]}
    a = resolve_task_id(
        {"common_config": {"feature_col": "sig", "start_date": 1, "end_date": 2}}, ast)
    b = resolve_task_id(
        {"common_config": {"feature_col": "sig", "start_date": 1, "end_date": 3}}, ast)
    assert a and b, "common_config 形状的 feature_col 也必须能解析出 task_id"
    assert a != b, "不同日期区间必须落入不同分桶"


def test_task_id_hash_includes_search_spec():
    """不同 HPO 搜索空间不得共用同一产物分桶。"""
    ast = {"op": "abs", "args": [{"col": "close"}]}
    a = resolve_task_id(
        build_exp_config({"feature_col": "sig", "search_spec": {"downsample": {"type": "choice", "values": [3]}}}), ast
    )
    b = resolve_task_id(
        build_exp_config({"feature_col": "sig", "search_spec": {"downsample": {"type": "choice", "values": [5]}}}), ast
    )
    assert a and b and a != b


def test_common_params_explicit_task_id_is_honored():
    """调用方显式提供的 common_params.task_id 是合法分桶覆盖。"""
    cfg = build_exp_config({"feature_col": "sig"})
    cfg["common_params"]["task_id"] = "manual bucket"
    assert resolve_task_id(cfg, None) == "manual-bucket"
