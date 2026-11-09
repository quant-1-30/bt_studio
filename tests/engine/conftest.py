#!/usr/bin/env python3
"""tests/engine 共享夹具。

registry_preserved: 自动保存/恢复 executor 的 NODE_REGISTRY 与
    retrieve_available_months —— 取代旧测试中每个用例手写的 try/finally。
write_dag: 向 tmp_path 写入 DAG XML 并返回路径。
"""

import os

import pytest

from bt_studio.engine import executor as ex


@pytest.fixture()
def registry_preserved():
    saved = dict(ex.NODE_REGISTRY)
    saved_lam = ex.retrieve_available_months
    try:
        yield
    finally:
        ex.NODE_REGISTRY.clear()
        ex.NODE_REGISTRY.update(saved)
        ex.retrieve_available_months = saved_lam


@pytest.fixture()
def write_dag(tmp_path):
    def _write(body: str, name: str = "t.xml") -> str:
        p = os.path.join(str(tmp_path), name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(body)
        return p

    return _write
