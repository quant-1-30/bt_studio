#!/usr/bin/env python3


import importlib

import pytest

CORE_MODULES = [
    # 顶层
    "bt_studio.constant",
    "bt_studio.schema",
    "bt_studio.default_config",
    "bt_studio.tune",
    # pipeline 子包 (含兼容 shim)
    "bt_studio.pipeline",
    "bt_studio.pipeline.preprocess",
    "bt_studio.pipeline.preprocess.panel",
    "bt_studio.pipeline.preprocess.macro",
    "bt_studio.pipeline.preprocess.sample",
    "bt_studio.pipeline.patterns",
    "bt_studio.pipeline.patterns.astc",
    "bt_studio.pipeline.patterns.fsm",
    "bt_studio.pipeline.patterns.pattern",
    "bt_studio.pipeline.metrics",
    "bt_studio.pipeline.metrics.score",
    "bt_studio.pipeline.metrics.pareto",
    "bt_studio.pipeline.metrics.fanova",
    "bt_studio.pipeline.inference",
    # utils
    "bt_studio.utils.months",
    "bt_studio.utils.io",
    "bt_studio.utils.math_func",
    "bt_studio.pipeline.metrics.gates",
    "bt_studio.utils.manifest",
    "bt_studio.utils.diagnostics",
    # agent/engine
    "bt_studio.agent",
    "bt_studio.agent.harness",
    "bt_studio.agent.flywheel",
    "bt_studio.agent.prompt",
    "bt_studio.utils.diagnostics.hpo_check",
    # engine (编排 tune 节点)
    "bt_studio.engine",
    "bt_studio.engine.parser",
    "bt_studio.engine.executor",
    # visual & monitor & server & live
    "bt_studio.visual",
    "bt_studio.visual.collapse",
    "bt_studio.visual.bkh",
    "bt_studio.visual._streamlit",
    "bt_studio.monitor",
    "bt_studio.monitor.indexer",
    "bt_studio.server",
    "bt_studio.server.main",
    "bt_studio.server.monitor_api",
    "bt_studio.server.watcher",
    "bt_studio.trade",
    "bt_studio.trade.engine",
    "bt_studio.trade.strategy",
    "bt_studio.trade.execution",
    "bt_studio.trade.sizers",
    "bt_studio.trade.scorer",
    "bt_studio.trade.data",
]


@pytest.mark.parametrize("module", CORE_MODULES)
def test_module_imports(module):
    importlib.import_module(module)


def test_tune_uses_default_config_ssot():
    import bt_studio.default_config as dc
    import bt_studio.tune as tune_mod

    assert tune_mod.PRODUCTION_SEARCH_SPEC is dc.PRODUCTION_SEARCH_SPEC


def test_utils_do_not_depend_on_high_layers():
    import ast
    from pathlib import Path

    forbidden = ("bt_studio.engine", "bt_studio.agent", "bt_studio.tune", "bt_studio.pipeline")
    utils_dir = Path(__file__).resolve().parent.parent.parent / "bt_studio" / "utils"
    offenders = []
    for path in utils_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(forbidden):
                offenders.append(f"{path}: from {node.module} import ...")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(forbidden):
                        offenders.append(f"{path}: import {alias.name}")
    assert not offenders, "\n".join(offenders)

    pipe_forbidden = ("bt_studio.engine", "bt_studio.agent", "bt_studio.tune")
    pipe_dir = Path(__file__).resolve().parent.parent.parent / "bt_studio" / "pipeline"
    pipe_offenders = []
    for path in pipe_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(pipe_forbidden):
                pipe_offenders.append(f"{path}: from {node.module} import ...")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(pipe_forbidden):
                        pipe_offenders.append(f"{path}: import {alias.name}")
    assert not pipe_offenders, "\n".join(pipe_offenders)

    tune_path = Path(__file__).resolve().parent.parent.parent / "bt_studio" / "tune.py"
    tune_tree = ast.parse(tune_path.read_text(encoding="utf-8"))
    tune_offenders = [
        f"{tune_path}: from {node.module} import ..."
        for node in ast.walk(tune_tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(("bt_studio.engine", "bt_studio.agent"))
    ]
    assert not tune_offenders, "\n".join(tune_offenders)
