#!/usr/bin/env python3
"""EvalResult 窗口缓存: 命中免计算 / 门控不在键内(阈值重放免费) / 版本隔离。"""

import json
import os
import sys

import polars as pl
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import bt_studio.agent.harness as H
from bt_studio.agent.harness import TwoStageAgentHarness


class _Recorder:
    calls = 0


def _fake_worker(dret_path, feat_paths, feats, cc, combos):
    _Recorder.calls += 1
    return {
        f: H.EvalResult(feature_name=f, status="success", score=1.5 + _Recorder.calls,
                        u_pval=0.05, trigger_count=100)
        for f in feats
    }


class _FakeFuture:
    def __init__(self, fn, args):
        self._fn, self._args = fn, args

    def result(self):
        return self._fn(*self._args)


class _FakePool:
    """编排层替身: submit 立即求值 (parent 侧调用, mock 可见)。"""

    def __init__(self, max_workers=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def submit(self, fn, *args, **kw):
        return _FakeFuture(fn, args)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # EVAL_CACHE_ROOT 是 RESULT_ROOT 派生的绝对路径; 测试必须显式隔离,
    # 否则缓存会写真实 result/cache/evals 并在用例/多次运行间穿透。
    cache_root = os.path.join(str(tmp_path), "evals")
    monkeypatch.setattr(H, "EVAL_CACHE_ROOT", cache_root)
    _Recorder.calls = 0
    monkeypatch.setattr(H, "_eval_window_worker", _fake_worker)
    monkeypatch.setattr(H, "ProcessPoolExecutor", _FakePool)


def _make_harness(extra=None, feats=("f1",), months=(202401, 202402)):
    h = TwoStageAgentHarness(common_config=dict(
        {"min_window_pass_ratio": 0.6, "eval_procs": 4,
         "min_excess_ret": 0.0, "u_pval": 0.5, "trigger": 5,
         "T1_rets": {"exit_5m": 5}, "eps": 1e-8, "exclude_bars": 10,
         "decay_minutes": 15.0, "dtw_window_frac": 0.1, "ranking_window": 5,
         "topk": 2, "alternative": "greater", "seed": 42,
         "days_since_ipo": 0, "_eval_data_fp": "d" * 16}, **(extra or {})))
    # 桩掉持久化与 dret 物化依赖: 造带正确命名的哑 parquet (worker 已 mock, 不读内容)
    paths = {}
    for f in feats:
        paths[f] = []
        for ym in months:
            p = os.path.join("result", "features", "agent", f"{f}-{'a'*16}",
                             f"hf_{f}_{ym}.parquet")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            pl.DataFrame({"x": [1]}).write_parquet(p)
            paths[f].append(p)
    h._last_feat_paths = paths
    h._last_fingerprints = {f: "a" * 16 for f in feats}
    h._eval_data_fp = "d" * 16
    return h, list(feats), list(months)


def _drive(h, feats, months):
    dret = pl.LazyFrame({"sid": [], "day": [], "close": []})
    return h._evaluate_windows_parallel(None, dret, feats, months)


def test_second_run_zero_compute(env):
    h, feats, months = _make_harness()
    r1 = _drive(h, feats, months)
    assert _Recorder.calls == 2  # 每月一次 worker
    # 全新 harness (新进程语义), 同 data_fp/网格 → 全命中
    _Recorder.calls = 0
    h2, feats2, months2 = _make_harness()
    r2 = _drive(h2, feats2, months2)
    assert _Recorder.calls == 0
    for f in feats:
        assert [r.score for r in r1[f]] == [r.score for r in r2[f]]


def test_gate_change_replays_free(env):
    h, feats, months = _make_harness()
    _drive(h, feats, months)
    _Recorder.calls = 0
    # 门控 (u_pval/比例) 不在缓存键内 → 调阈值零重算
    h3, feats3, months3 = _make_harness(extra={"u_pval": 0.5,
                                               "min_window_pass_ratio": 0.99})
    _drive(h3, feats3, months3)
    assert _Recorder.calls == 0


def test_version_invalidates(env, monkeypatch):
    h, feats, months = _make_harness()
    _drive(h, feats, months)
    _Recorder.calls = 0
    monkeypatch.setattr(H, "EVAL_LOGIC_VERSION", "v99")
    h4, feats4, months4 = _make_harness()
    _drive(h4, feats4, months4)
    assert _Recorder.calls == 2  # 版本变更 → 全重算


def test_internal_eval_config_change_recomputes(env):
    """worker/discover 内部口径 (min_excess_ret 等) 必须进缓存键, 否则 stale 复用。"""
    h, feats, months = _make_harness()
    _drive(h, feats, months)
    _Recorder.calls = 0
    h5, feats5, months5 = _make_harness(extra={"min_excess_ret": 0.02})
    _drive(h5, feats5, months5)
    assert _Recorder.calls == 2  # 非回放层语义变更 → 不允许免费命中
