#!/usr/bin/env python3
"""engine.task_manager / ctx 回归测试。

覆盖 2026-09 代码审计项:
    H1  ctx.ensure_ray 异常透传 (不被 NameError 掩盖)
    H2  pending JSON 契约 (run_agent 真实产物形状可入队)
    M9  intake 先原子认领 (.consumed) 再入队; 消费成功后清理
    M17 _execute_pipeline 返回任务隔离的 model_dir
"""

import json
import os
import threading

import pytest

from bt_studio.engine import executor as ex
from bt_studio.engine import task_manager as tm_mod


def _make_tm(tmp):
    tm = tm_mod.AsyncTaskManager.__new__(tm_mod.AsyncTaskManager)
    tm._registry = tm_mod.TaskRegistry(max_queued=100, max_history=200)
    tm._queue = __import__("queue").Queue()
    tm._running = False   # 不启动 worker/intake 线程 (纯逻辑测试)
    tm._intake_dir = tmp
    return tm


AGENT_CONTRACT_TASK = {
    "ast_recipe": [{"name": "sig", "ast": {"op": "abs", "args": [{"col": "close"}]}}],
    "common_params": {
        "feature_col": "sig",
        "start_date": 20221001,
        "end_date": 20221231,
        "train_window": 1,
        "oos_step": 1,
    },
    "search_spec": {"downsample": {"type": "choice", "values": [4]}},
    "created_at": "2026-09-01T00:00:00",
    "dag": "wfo_hpo",
    "source": "llm_agent",
}


# ---------------------------------------------------------------------------
# H1: ctx logger / ensure_ray 异常透传
# ---------------------------------------------------------------------------

def test_ctx_module_has_logger():
    import logging
    from bt_studio.engine import ctx as ctx_mod
    assert isinstance(ctx_mod.logger, logging.Logger)


def test_ensure_ray_reports_original_error(monkeypatch):
    """ray.init 失败必须上抛原始 RuntimeError, 不得被 NameError 掩盖。"""
    from bt_studio.engine import ctx as ctx_mod

    def boom(**kwargs):
        raise RuntimeError("init failed")

    monkeypatch.setattr(ctx_mod.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(ctx_mod.ray, "init", boom)
    with pytest.raises(RuntimeError, match="init failed"):
        ctx_mod.ensure_ray(num_cpus=1)


def test_platform_concurrency_darwin_vs_linux(monkeypatch):
    """验证 macOS (Darwin) 与 Linux 64C 大内存工作站的动态参数派生。"""
    from bt_studio.engine import ctx as ctx_mod

    for k in ("HPO_NUM_WORKERS", "EVAL_PROCS", "TICK_BATCH_SIDS", "RAY_OBJECT_STORE_GB"):
        monkeypatch.delenv(k, raising=False)

    # 1. 模拟 Linux 64 核 + 512GB RAM (Threadripper 9995WX)
    monkeypatch.setattr(ctx_mod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(ctx_mod.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(ctx_mod, "_get_system_total_ram", lambda: 512 * (1024 ** 3))
    linux_cfg = ctx_mod.resolve_platform_concurrency()
    linux_ray = ctx_mod._resolve_platform_ray_defaults()
    assert linux_cfg["num_workers"] == 48       # 64 * 0.75 = 48 并发 Ray Trials
    assert linux_cfg["eval_procs"] == 21        # 64 // 3 = 21 窗口进程 (每进程 3 DTW 线程)
    assert linux_cfg["tick_batch_sids"] == 128  # 512GB 内存自动升至 128 批量
    assert linux_ray["plasma_limit"] <= 64 * (1024 ** 3)

    # 2. 模拟 macOS 10 核 + 32GB RAM
    monkeypatch.setattr(ctx_mod.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(ctx_mod.os, "cpu_count", lambda: 10)
    monkeypatch.setattr(ctx_mod, "_get_system_total_ram", lambda: 32 * (1024 ** 3))
    mac_cfg = ctx_mod.resolve_platform_concurrency()
    mac_ray = ctx_mod._resolve_platform_ray_defaults()
    assert mac_cfg["num_workers"] == 8
    assert mac_cfg["eval_procs"] == 5
    assert mac_cfg["tick_batch_sids"] == 8
    assert mac_ray["plasma_limit"] <= 16 * (1024 ** 3)


# ---------------------------------------------------------------------------
# H2 / M9: intake 契约与原子认领
# ---------------------------------------------------------------------------

def test_intake_accepts_agent_contract_shape(tmp_path):
    tmp = str(tmp_path)
    tm = _make_tm(tmp)
    path = os.path.join(tmp, "agent.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(AGENT_CONTRACT_TASK, f)

    assert tm._drain_intake_once() == 1
    # 消费成功: 原 JSON 被原子认领为 .consumed 且入队后清理 (不留尾巴)
    assert not os.path.exists(path)
    assert not os.path.exists(path + ".invalid")

    _, dag_ref, cfg, _ = tm._queue.get_nowait()
    assert dag_ref == "wfo_hpo"
    assert cfg["common_params"]["feature_col"] == "sig"


def test_intake_dag_refs_and_invalid_quarantine(tmp_path):
    tmp = str(tmp_path)
    good1, good2, bad = (os.path.join(tmp, n) for n in ("a.json", "b.json", "bad.json"))
    with open(good1, "w") as f:
        json.dump({"feature_col": "a", "dag": "wfo_production"}, f)
    with open(good2, "w") as f:
        json.dump({"feature_col": "b"}, f)
    with open(bad, "w") as f:
        f.write("{not json")

    tm = _make_tm(tmp)
    assert tm._drain_intake_once() == 2
    _, dag1, cfg1, _ = tm._queue.get_nowait()
    _, dag2, cfg2, _ = tm._queue.get_nowait()
    assert dag1 == "wfo_production" and cfg1.get("feature_col") == "a"
    assert dag2 == "wfo_hpo" and cfg2.get("feature_col") == "b"
    assert not os.path.exists(good1) and os.path.exists(bad + ".invalid")


def test_intake_rejects_absolute_dag_path(tmp_path):
    tmp = str(tmp_path)
    evil = os.path.join(tmp, "evil.json")
    with open(evil, "w") as f:
        json.dump({"feature_col": "a", "dag": "/etc/passwd.xml"}, f)

    tm = _make_tm(tmp)
    assert tm._drain_intake_once() == 0
    assert os.path.exists(evil + ".invalid")
    assert tm._queue.empty()


def test_intake_claims_file_before_enqueue(tmp_path, monkeypatch):
    tmp = str(tmp_path)
    tm = _make_tm(tmp)
    observed = {}

    real_submit = tm.submit_dag_task

    def spy_submit(dag_ref, config):
        observed["files_at_submit"] = sorted(os.listdir(tmp))
        return real_submit(dag_ref, config)

    monkeypatch.setattr(tm, "submit_dag_task", spy_submit)

    path = os.path.join(tmp, "a.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"feature_col": "a"}, f)

    assert tm._drain_intake_once() == 1
    # 入队发生时, 文件必须已被原子认领为 .consumed (崩溃窗口内不会重复消费)
    assert "a.json.consumed" in observed["files_at_submit"], observed
    # 消费成功后 .consumed 被清理, 原 JSON 不复存在
    assert not os.path.exists(path)


# ---------------------------------------------------------------------------
# submit / pipeline 执行
# ---------------------------------------------------------------------------

def test_submit_accepts_feature_col_in_common_params(tmp_path):
    tm = _make_tm(str(tmp_path))
    tid = tm.submit_dag_task(
        "wfo_hpo", {"common_params": {"feature_col": "sig", "train_window": 1}})
    assert tid in tm.snapshot()["tasks"]
    assert tm.snapshot(tid).feature_col == "sig"


def test_execute_pipeline_reports_task_model_dir(tmp_path, monkeypatch):
    """M17: 有 task_id 时 model_dir 必须指向任务隔离目录, 而非全局常量。"""
    tmp = str(tmp_path)
    tm = _make_tm(tmp)
    task_id = "t1"
    tm._registry.add(tm_mod.TaskResult(task_id=task_id,
                                       status=tm_mod.TaskStatus.QUEUED))
    model_dir = os.path.join(tmp, "task_models")
    exp = {
        "feature_col": "sig",
        "common_params": {
            "feature_col": "sig",
            "tune_model_dir": model_dir,
            "tune_collapse_dir": os.path.join(tmp, "collapse"),
        },
    }
    monkeypatch.setattr(
        tm_mod, "run_dag",
        lambda dag, cfg, ast: ex.EngineResult(dag_name="x", last_model_id=1,
                                              models_produced=[1]))
    tm._execute_pipeline(task_id, "wfo_hpo", exp, None)
    res = tm.snapshot(task_id)
    assert res.status == tm_mod.TaskStatus.COMPLETED
    assert res.result["model_dir"] == model_dir


def test_submit_does_not_inject_queue_uuid_into_engine_bucket(tmp_path):
    """队列 uuid 只用于 manager 状态跟踪, 不得成为 engine 内容寻址分桶 id。"""
    tm = _make_tm(str(tmp_path))
    tm.submit_dag_task(
        "wfo_hpo", {"common_params": {"feature_col": "sig", "train_window": 1}}
    )
    _, _, cfg, _ = tm._queue.get_nowait()
    assert "task_id" not in cfg["common_params"]


def test_submit_validates_ast_recipe_and_fingerprint(tmp_path):
    """服务端重算 AST 指纹并静态校验因果律, 拒绝篡改或非法 pending 负载。"""
    tm = _make_tm(str(tmp_path))

    # 1. 非法未来函数 AST 被入队预检拦截
    with pytest.raises(ValueError, match="静态编译校验失败"):
        tm.submit_dag_task(
            "wfo_hpo",
            {
                "feature_col": "sig",
                "ast_recipe": {"op": "ref", "args": [{"col": "close"}, -2]},
            },
        )

    # 2. f_{fp12} 命名特征但与 ast_recipe 真实指纹不匹配 -> 拒绝
    with pytest.raises(ValueError, match="与服务端重算指纹.*不匹配"):
        tm.submit_dag_task(
            "wfo_hpo",
            {
                "feature_col": "f_000000000000",
                "ast_recipe": {"op": "abs", "args": [{"col": "close"}]},
            },
        )
