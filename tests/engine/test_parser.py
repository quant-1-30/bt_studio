#!/usr/bin/env python3
"""engine.parser 回归测试: DAG XML 解析 (图结构 / 条件依赖 / 属性 / 负例)。"""

import pytest

from bt_studio.engine.parser import parse_dag, list_dags, DagValidationError


def test_parser():
    prod = parse_dag("wfo_production")
    hpo = parse_dag("wfo_hpo")
    assert prod.name == "wfo_production" and hpo.name == "wfo_hpo"

    assert set(prod.graph.nodes) == {
        "macro", "extract_train", "extract_oos",
        "decay", "train_data", "tune", "update", "oos"}
    assert set(hpo.graph.nodes) == {"macro", "extract_train", "train_data", "tune"}

    # conditional edges carry cond/negated attrs
    tune_deps = prod.graph.nodes["tune"]["deps"]
    conds = {(t.cond, t.negated) for d in tune_deps for t in d.terms()}
    assert ("decay", False) in conds, conds
    update_deps = prod.graph.nodes["update"]["deps"]
    uconds = {(t.cond, t.negated) for d in update_deps for t in d.terms()}
    assert ("decay", True) in uconds, uconds
    oos_alts = [t for d in prod.graph.nodes["oos"]["deps"] for t in d.terms()
                if t.cond is None]
    assert {"tune", "update"} <= {t.node for t in oos_alts}

    assert set(list_dags()) >= {"wfo_production", "wfo_hpo"}


def test_parser_negatives(write_dag):
    cases = [
        ('<pipeline name="a"><node id="x" fn="node_prepare_macro" deps="ghost"/></pipeline>', "unknown dep"),
        ('<pipeline name="a"><node id="x" fn="f" deps="y"/><node id="y" fn="f" deps="x"/></pipeline>', "cycle"),
        ('<pipeline name="a"><node id="x" fn="node_prepare_macro" role="bogus"/></pipeline>', "bad role"),
        ('<pipeline name="a"><node id="x" fn="node_prepare_macro" deps="x>bad!"/></pipeline>', "bad term"),
    ]
    for body, why in cases:
        with pytest.raises(DagValidationError):
            parse_dag(write_dag(body))


def test_parser_reads_output_attr_and_accepts_warmup_role(write_dag):
    p = write_dag("""<pipeline name="t">
      <window from="first"/>
      <node id="macro" fn="node_prepare_macro"/>
      <node id="decay" fn="node_check_decay_monthly" output="gate" deps="macro"/>
      <node id="warm" fn="node_extract_feature_monthly" role="warmup" deps="macro"/>
    </pipeline>""")
    dag = parse_dag(p)
    assert dag.graph.nodes["decay"]["output"] == "gate"
    assert dag.graph.nodes["warm"]["role"] == "warmup"


def test_parser_rejects_invalid_output_name(write_dag):
    p = write_dag("""<pipeline name="t">
      <node id="macro" fn="node_prepare_macro" output="bad-name!"/>
    </pipeline>""")
    with pytest.raises(DagValidationError):
        parse_dag(p)
