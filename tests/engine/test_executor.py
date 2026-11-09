#!/usr/bin/env python3
"""engine.executor 回归测试: MockNodes 干跑 (无 IO / 无 ray)。

覆盖: wfo_production 全流程与分支、wfo_hpo 剪枝、断点恢复、fail-fast、
output="model_id" 声明式追踪、warmup role 切片、短历史显式报错、
feature_col 必填前置校验。
"""

import os

import pytest

from bt_studio.engine.parser import parse_dag
from bt_studio.engine import executor as ex

MONTHS = [202101 + m for m in range(12)] + [202201, 202202, 202203]

EXP = {"common_params": {"train_window": 3, "oos_step": 3, "num_workers": 1,
                         "feature_col": "f"},
       "search_spec": {}}


def _write_dag_file(tmp, warm_variant=False, producer_output=False):
    """写一个最小 DAG: macro [+ warmup extract 节点 / output=model_id 生产者]。"""
    extra = ""
    if warm_variant:
        extra = ('\n      <node id="warm" fn="node_extract_feature_monthly" '
                 'role="warmup" deps="macro"/>')
    if producer_output:
        extra = ('\n      <node id="producer" fn="node_prepare_train" '
                 'output="model_id" deps="macro"/>')
    p = os.path.join(tmp, "t.xml")
    with open(p, "w", encoding="utf-8") as f:
        f.write(f"""<pipeline name="t">
          <window from="first"/>
          <node id="macro" fn="node_prepare_macro"/>{extra}
        </pipeline>""")
    return p


class MockNodes:
    def __init__(self):
        self.calls = []
        self.decay_verdict = True  # True = needs tuning, False = healthy decay
        self.seen_task_ids = set()

    def fake_macro(self, common_config):
        self.calls.append(("macro",))
        return {"dret_path": "/synthetic/dret.parquet", "universe_sids": {}}

    def fake_extract(self, ymonths, universe_sids, common_config, ast_recipe, feature_col=None, raw_hf_lf=None):
        self.calls.append(("extract", tuple(ymonths)))
        assert ast_recipe is not None, "ast_recipe is mandatory now"
        tid = common_config.get("_task_id")
        if tid:
            self.seen_task_ids.add(tid)
        return [f"/f/hf_feat_{ym}.parquet" for ym in ymonths]

    def fake_decay(self, prev_model_id, prev_oos_yms, dret_path, train_paths, common_config):
        self.calls.append(("decay", prev_model_id, tuple(prev_oos_yms)))
        # 关键不变量: executor 传入的 train_paths 必须覆盖 prev_oos_yms
        assert all(any(f"_{ym}.parquet" in p for p in train_paths) for ym in prev_oos_yms)
        # return True for tune branch, False for update branch
        return prev_model_id is None or self.decay_verdict

    def fake_train_data(self, dret_path, train_paths, common_config):
        self.calls.append(("train_data", len(train_paths)))
        return {"stub": True}

    def fake_tune(self, prev_model_id, model_id, train_data, exp_config):
        self.calls.append(("tune", prev_model_id, model_id))
        return model_id  # engine convention: model_id | None

    def fake_update(self, model_id, prev_model_id, dret_path, train_paths, common_config):
        self.calls.append(("update", model_id, prev_model_id))
        return model_id

    def fake_oos(self, model_id, dret_path, oos_yms, oos_paths, train_paths, common_config):
        self.calls.append(("oos", model_id, tuple(oos_yms)))
        # oos 自行从 train_paths 尾部推导 warmup 月, 这里只验证绑定到位
        assert oos_paths and train_paths


def _install_mocks(mocks):
    ex.NODE_REGISTRY.clear()
    ex.NODE_REGISTRY.update({
        "node_prepare_macro": mocks.fake_macro,
        "node_extract_feature_monthly": mocks.fake_extract,
        "node_check_decay_monthly": mocks.fake_decay,
        "node_prepare_train": mocks.fake_train_data,
        "node_tune_monthly": mocks.fake_tune,
        "node_update_fsm_matrix": mocks.fake_update,
        "node_oos_inference_monthly": mocks.fake_oos,
    })


@pytest.fixture()
def mock_engine(registry_preserved):
    """安装 MockNodes 并注入合成月份序列; yield mocks 供断言与行为切换。"""
    mocks = MockNodes()
    _install_mocks(mocks)
    ex.retrieve_available_months = lambda p: list(MONTHS)
    return mocks


def test_executor_wfo_production(mock_engine):
    mocks = mock_engine
    # production: 4 windows — w1 tune (cold start), w2-4 inherit model (update)
    mocks.decay_verdict = False
    res = ex.run_dag(parse_dag("wfo_production"), EXP, ast_recipe={"op": "x"})

    assert res.windows_executed == 4
    # cold start always tunes. Subsequent windows have decay_verdict=False, so they should UPDATE.
    kinds = [c[0] for c in mocks.calls]
    assert kinds.count("tune") == 1, f"Expected 1 tune, got {kinds.count('tune')}"
    assert kinds.count("update") == 3, f"Expected 3 updates, got {kinds.count('update')}"
    assert kinds.count("oos") == 4
    assert res.models_produced == [202103, 202106, 202109, 202112]
    assert res.last_model_id == 202112
    # 任务隔离: feature_col=f -> 内容寻址 task_id 注入 common_config
    assert len(mocks.seen_task_ids) == 1 and next(iter(mocks.seen_task_ids)).startswith("f-")


def test_executor_wfo_production_forced_decay_retunes(mock_engine):
    mocks = mock_engine
    mocks.decay_verdict = True
    res = ex.run_dag(parse_dag("wfo_production"), EXP, ast_recipe={"op": "x"})
    kinds = [c[0] for c in mocks.calls]
    assert res.windows_executed == 4
    assert kinds.count("tune") == 4, f"Expected 4 tunes due to decay, got {kinds.count('tune')}"
    assert kinds.count("update") == 0, f"Expected 0 updates, got {kinds.count('update')}"


def test_executor_wfo_hpo(mock_engine):
    mocks = mock_engine
    res = ex.run_dag(parse_dag("wfo_hpo"), EXP, ast_recipe={"op": "x"})
    kinds = [c[0] for c in mocks.calls]
    assert res.windows_executed == 1
    assert kinds.count("tune") == 1
    assert "decay" not in kinds and "update" not in kinds and "oos" not in kinds

    extracts = [c[1] for c in mocks.calls if c[0] == "extract"]
    # 15 个月 + train_window=3 + from="last-trainable" -> 尾部切片
    assert extracts == [(202201, 202202, 202203)]


def test_executor_fail_fast(mock_engine):
    def boom(*args, **kwargs):
        raise RuntimeError("decay exploded")

    ex.NODE_REGISTRY["node_check_decay_monthly"] = boom
    with pytest.raises(RuntimeError, match="decay exploded"):
        ex.run_dag(parse_dag("wfo_production"), EXP, ast_recipe={"op": "x"})
    # repr_graph 嵌入异常链 (✗ failed here 标记)
    try:
        ex.run_dag(parse_dag("wfo_production"), EXP, ast_recipe={"op": "x"})
    except RuntimeError as e:
        assert "✗ failed here" in str(e)


def test_executor_resume_grid_alignment(mock_engine):
    exp2 = {"common_params": {"train_window": 3, "oos_step": 3, "num_workers": 1,
                              "feature_col": "f", "last_model_id": 202106},
            "search_spec": {}}
    res = ex.run_dag(parse_dag("wfo_production"), exp2, ast_recipe={"op": "x"})
    # 202106 (idx=6) 已产出 -> 恢复后仅执行 idx=9 (202109) 和 idx=12 (202112)
    assert res.windows_executed == 2, f"Expected 2 windows, got {res.windows_executed}"
    assert res.models_produced == [202109, 202112], f"Got models {res.models_produced}"


def test_executor_output_model_id_binding(tmp_path, registry_preserved):
    """fn 不在硬编码名单 (非 tune/update), 仅凭 output="model_id" 声明就必须被追踪。"""
    calls = []

    def fake_producer(common_config, model_id, prev_model_id):
        calls.append(model_id)
        return 424242

    ex.NODE_REGISTRY.clear()
    ex.NODE_REGISTRY.update({
        "node_prepare_macro": lambda common_config: {
            "dret_path": "/synthetic/dret.parquet", "universe_sids": {}},
        "node_prepare_train": fake_producer,
    })
    ex.retrieve_available_months = lambda p_: [202101 + m for m in range(4)]

    p = _write_dag_file(str(tmp_path), producer_output=True)
    res = ex.run_dag(parse_dag(p), {"common_params": {
        "train_window": 3, "oos_step": 3, "feature_col": "f"}})
    assert res.models_produced == [424242], res.models_produced
    assert res.last_model_id == 424242


def test_executor_warmup_role_months(tmp_path, registry_preserved):
    """warmup role: 窗口内切片 = 训练窗前 warmup_months 个月; 头窗无历史 -> 空。"""
    calls = []

    def fake_extract(ymonths, universe_sids, common_config, ast_recipe,
                     feature_col=None, raw_hf_lf=None):
        calls.append(tuple(ymonths))
        return [f"/f/hf_{ym}.parquet" for ym in ymonths]

    ex.NODE_REGISTRY.clear()
    ex.NODE_REGISTRY.update({
        "node_prepare_macro": lambda common_config: {
            "dret_path": "/x.parquet", "universe_sids": {}},
        "node_extract_feature_monthly": fake_extract,
    })
    ex.retrieve_available_months = lambda p_: [202101 + m for m in range(7)]

    p = _write_dag_file(str(tmp_path), warm_variant=True)
    res = ex.run_dag(parse_dag(p), {"common_params": {
        "train_window": 3, "oos_step": 3, "warmup_months": 2, "feature_col": "f"}})
    assert res.windows_executed == 2
    # w1(idx=3): train=months[0:3], warmup 无更早历史 -> 空切片
    # w2(idx=6): train=months[3:6], warmup=months[1:3]
    assert calls == [(), (202102, 202103)], calls


def test_executor_rejects_history_shorter_than_train_window(tmp_path, registry_preserved):
    ex.NODE_REGISTRY.clear()
    ex.NODE_REGISTRY.update({
        "node_prepare_macro": lambda common_config: {
            "dret_path": "/x.parquet", "universe_sids": {}},
    })
    ex.retrieve_available_months = lambda p_: [202101, 202102]
    p = _write_dag_file(str(tmp_path))
    with pytest.raises(ValueError, match="train_window"):
        ex.run_dag(parse_dag(p), {"common_params": {
            "train_window": 3, "oos_step": 3, "feature_col": "f"}})


def test_run_dag_requires_feature_col(registry_preserved):
    """无 feature_col 且 recipe 未声明输出名 -> 在任何节点执行前报错。"""
    calls = []

    def spy_macro(common_config):
        calls.append("macro")
        return {"dret_path": "/x.parquet", "universe_sids": {}}

    ex.NODE_REGISTRY["node_prepare_macro"] = spy_macro
    with pytest.raises(ValueError, match="feature_col"):
        ex.run_dag(parse_dag("wfo_hpo"),
                   {"common_params": {"train_window": 3, "oos_step": 3}},
                   ast_recipe={"op": "abs", "args": [{"col": "close"}]})
    assert not calls, "校验必须先于 macro 节点 (fail before IO)"
