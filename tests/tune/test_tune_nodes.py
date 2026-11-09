#!/usr/bin/env python3
"""tune.py 节点级回归测试 (TDD red→green)。

覆盖 2026-09 代码审计定位的逻辑 bug:
    M8  node_check_decay_monthly 未拼接 warmup 月 → OOS 头部 macro 态
        退化中性 (同文件 oos 节点已修的 train/serve skew 只修了一半)
    L   node_tune_monthly(train_data=None) 直接 TypeError 而非优雅失败
以及 2026-09 import 重构后的 SSOT 透传契约:
    - build_ray_tune_search_space 的 spec → Ray Tune 空间映射
    - 用户 search_spec 覆盖项经 search_spec_of(exp_config) 完整透传
"""

import datetime
import json
import os
import pickle

import numpy as np
import polars as pl
import pytest

import bt_studio.tune as tune_mod
from bt_studio.compiler.ast import ast_fingerprint
from bt_studio.default_config import PRODUCTION_SEARCH_SPEC, build_exp_config, search_spec_of


# ---------------------------------------------------------------------------
# 合成数据: 2 sids × 240 bars × (2024-01, 2024-02) 各 6 天
# ---------------------------------------------------------------------------

SIDS = ["600000", "600001"]
MONTH_DAYS = {
    202401: [datetime.date(2024, 1, i) for i in (2, 3, 4, 5, 8, 9)],
    202402: [datetime.date(2024, 2, i) for i in (1, 2, 5, 6, 7, 8)],
}


def _synth_frames(seed: int = 11):
    rng = np.random.RandomState(seed)
    hf_parts, daily_rows = [], []
    for si, sid in enumerate(SIDS):
        base = 100.0 + 5.0 * si
        for ym, days in MONTH_DAYS.items():
            for day in days:
                base *= 1.001
                close = base + rng.normal(0, 0.01, 240)
                hf_parts.append(pl.DataFrame({
                    "day": [day] * 240,
                    "sid": [sid] * 240,
                    "bar_idx": np.arange(240, dtype=np.int32),
                    "open": close,
                    "high": close * 1.001,
                    "low": close * 0.999,
                    "close": close,
                    "volume": np.full(240, 1000.0),
                    "amount": close * 1000.0,
                    "sig": rng.normal(0, 0.05, 240),
                }))
                daily_rows.append({
                    "day": day, "sid": sid, "close": float(base),
                    "high": float(base * 1.001), "low": float(base * 0.999),
                    "volume": 240000.0, "amount": 240000.0 * base,
                })
    return pl.concat(hf_parts), pl.DataFrame(daily_rows)


DECAY_CC = {
    "feature_col": "sig",
    "eps": 1e-8,
    "T1_rets": {"exit_5m": 5},
    "decay_minutes": 15,
    "exclude_bars": 10,
    "dtw_window_frac": 0.1,
    "ranking_window": 5,
    "trigger": 5,
    "alternative": "greater",
    "u_pval": 0.05,
}


def _month_ids(df: pl.DataFrame) -> list:
    return (df.select((pl.col("day").dt.year() * 100 + pl.col("day").dt.month())
                      .alias("_m"))["_m"].unique().to_list())


# ---------------------------------------------------------------------------
# M8: decay 节点 warmup 前拼
# ---------------------------------------------------------------------------

@pytest.fixture()
def decay_env(tmp_path):
    tmp = str(tmp_path)
    hf_df, daily_df = _synth_frames()

    feat_dir = os.path.join(tmp, "features")
    os.makedirs(feat_dir, exist_ok=True)
    train_paths = []
    for ym in (202401, 202402):
        p = os.path.join(feat_dir, f"hf_sig_{ym}.parquet")
        month_df = hf_df.with_columns(
            (pl.col("day").dt.year() * 100 + pl.col("day").dt.month()).alias("_m")
        ).filter(pl.col("_m") == ym).drop("_m")
        month_df.write_parquet(p)
        train_paths.append(p)

    dret_path = os.path.join(tmp, "daily.parquet")
    daily_df.write_parquet(dret_path)

    model_dir = os.path.join(tmp, "models")
    os.makedirs(model_dir, exist_ok=True)
    ckpt = {
        "config": {"downsample": 3, "motif_minutes": 30, "m": 10,
                   "threshold_r": 0.7, "threshold_d": 2.4},
        "motif": np.zeros((1, 10)),
        "fsm_matrix": {},
    }
    with open(os.path.join(model_dir, "model_202402.pkl"), "wb") as f:
        pickle.dump(ckpt, f)

    common = dict(DECAY_CC)
    common["tune_model_dir"] = model_dir
    return {"train_paths": train_paths, "dret_path": dret_path,
            "common": common, "hf_df": hf_df}


def test_decay_prepends_warmup_for_macro(decay_env, monkeypatch):
    env = decay_env
    captured = {}

    def fake_eval(panel_df, curves_2d, motif, tune_config, common_config,
                  skip_stats=False, stats_yms=None):
        captured["panel_months"] = sorted(_month_ids(panel_df))
        captured["stats_yms"] = stats_yms
        return {"status": "success", "u_pval": 0.01}

    monkeypatch.setattr(tune_mod, "evaluate_and_build_fsm", fake_eval)

    need_tune = tune_mod.node_check_decay_monthly(
        202402, [202401, 202402], env["dret_path"], env["train_paths"],
        {**env["common"], "decay_eval_months": 1})

    assert need_tune is False, "健康模型 (p=0.01<=0.05) 不应触发重训"
    # 送给评估器的面板 = warmup(202401) + eval_yms(202402) (macro 状态用全窗口回填)
    assert captured["panel_months"] == [202401, 202402], captured
    # 统计行经 stats_yms 限定在裁剪后的 eval_yms (202402), 而非完整 prev_oos_yms
    assert captured["stats_yms"] == [202402]
    # stats_yms 过滤的端到端正确性由 tests/tune/test_hpo_full_chain.py 的真实 decay 链路覆盖


def test_decay_cold_start_triggers_tune(decay_env):
    """prev_model_id=None (冷启动) / 模型文件缺失 -> 触发重训。"""
    env = decay_env
    assert tune_mod.node_check_decay_monthly(
        None, [202402], env["dret_path"], env["train_paths"], env["common"]) is True
    assert tune_mod.node_check_decay_monthly(
        209999, [202402], env["dret_path"], env["train_paths"], env["common"]) is True


# ---------------------------------------------------------------------------
# L: node_tune_monthly 对 train_data=None 优雅失败
# ---------------------------------------------------------------------------

def test_node_tune_rejects_none_train_data():
    assert tune_mod.node_tune_monthly(None, 202402, None, {}) is None


# ---------------------------------------------------------------------------
# 预计算复用: agent 命名空间约定 (剔除 _precomputed_paths 显式桥)
# ---------------------------------------------------------------------------

def test_node_extract_hits_agent_namespace(tmp_path, monkeypatch):
    """harness 预计算特征位于 FEATURE_DIR/agent/{feat}-{指纹8}/,
    extract 节点按 (feature_col, ast_recipe) 确定性推导并命中, 不再拉 tick。"""
    tmp = str(tmp_path)
    monkeypatch.setattr(tune_mod, "FEATURE_DIR", os.path.join(tmp, "features"))

    recipe = [{"name": "sig", "ast": {"op": "mul", "args": [{"col": "raw_sig"}, 1.0]}}]
    fp12 = str(ast_fingerprint(recipe))[:12]   # v2: 纯指纹键
    agent_dir = os.path.join(tmp, "features", fp12)
    os.makedirs(agent_dir, exist_ok=True)
    precomputed = os.path.join(agent_dir, "hf_sig_202401.parquet")
    pl.DataFrame({"day": [1], "sid": ["a"], "sig": [0.1]}).write_parquet(precomputed)

    def no_fetch(*a, **kw):
        raise AssertionError("命中 agent 预计算后不得再拉取 tick")

    monkeypatch.setattr(tune_mod, "_prepare_tick_for_months", no_fetch)

    paths = tune_mod.node_extract_feature_monthly(
        [202401], {202401: ["a"]}, {"eps": 1e-8}, recipe)
    assert paths == [precomputed]


# ---------------------------------------------------------------------------
# SSOT: 搜索空间 spec → Ray Tune 空间映射与透传
# ---------------------------------------------------------------------------

def test_build_ray_tune_search_space_type_mapping():
    """spec 类型必须显式映射到 Ray Tune 域; 未知类型显式报错。"""
    space = tune_mod.build_ray_tune_search_space({
        "downsample": {"type": "choice", "values": [3, 5]},
        "threshold_r": {"type": "uniform", "low": 0.6, "high": 0.8},
        "lookback": {"type": "int_range", "low": 10, "high": 100},
    })
    assert set(space) == {"downsample", "threshold_r", "lookback"}

    with pytest.raises(ValueError, match="Unsupported tune param type"):
        tune_mod.build_ray_tune_search_space({"bad": {"type": "loguniform"}})


def test_tune_search_spec_flows_through_exp_config():
    """用户经 build_exp_config(search_spec=...) 的覆盖项必须能被
    node_tune_monthly 的 search_spec_of(exp_config) 解析到 (透传契约)。"""
    exp_cfg = build_exp_config({
        "search_spec": {
            "downsample": {"type": "choice", "values": [3, 5]},
            "motif_minutes": {"type": "choice", "values": [30, 45]},
        },
    })
    spec = search_spec_of(exp_cfg)
    assert spec["downsample"] == {"type": "choice", "values": [3, 5]}
    assert spec["motif_minutes"] == {"type": "choice", "values": [30, 45]}
    # 未覆盖的参数回落到 SSOT 基准
    assert spec["threshold_r"] == PRODUCTION_SEARCH_SPEC["threshold_r"]

    # 解析出的 spec 必须能直接构建 Ray Tune 空间 (类型完备性)
    space = tune_mod.build_ray_tune_search_space(spec)
    assert set(space) >= {"downsample", "motif_minutes", "threshold_r"}


# ---------------------------------------------------------------------------
# M11a: prepare_macro 缓存覆盖检查 (迁自 tests/engine/test_bugfix_engine.py)
# ---------------------------------------------------------------------------

def test_prepare_macro_regenerates_when_cache_stale(tmp_path, monkeypatch):
    import polars as pl

    tmp = str(tmp_path)
    base = os.path.join(tmp, "tune_task")
    os.makedirs(base, exist_ok=True)
    stale_path = os.path.join(base, "global_daily.parquet")
    pl.DataFrame({
        "sid": ["a"],
        "day": ["20230630"],
        "close": [1.0], "high": [1.1], "low": [0.9],
        "volume": [100.0], "amount": [100.0],
    }).write_parquet(stale_path)

    calls = []

    def fake_prepare_macro(start_date, end_date, benchmark):
        calls.append((start_date, end_date))
        uni = pl.LazyFrame({"sid": ["a"], "first_trading": ["20200101"]})
        daily = pl.LazyFrame({
            "sid": ["a"], "day": ["20231229"],
            "close": [2.0], "high": [2.1], "low": [1.9],
            "volume": [100.0], "amount": [200.0],
        })
        return uni, daily

    monkeypatch.setattr(tune_mod, "prepare_macro", fake_prepare_macro)
    tune_mod.node_prepare_macro({
        "tune_base_dir": base,
        "start_date": 20230101,
        "end_date": 20231231,
        "benchmark": "1A0001",
        "warmup_months": 1,
        "days_since_ipo": 0,
    })
    assert calls, "缓存未覆盖 end_date 月份时必须重新拉取, 而不是静默 cache-hit"


def test_prepare_macro_cache_hit_when_covered(tmp_path, monkeypatch):
    import datetime
    import polars as pl

    tmp = str(tmp_path)
    base = os.path.join(tmp, "tune_task")
    os.makedirs(base, exist_ok=True)
    fresh_path = os.path.join(base, "global_daily.parquet")
    pl.DataFrame({
        "sid": ["a"],
        "day": [datetime.date(2023, 12, 29)],
        "close": [1.0], "high": [1.1], "low": [0.9],
        "volume": [100.0], "amount": [100.0],
    }).write_parquet(fresh_path)

    def fail_fetch(*a, **kw):
        raise AssertionError("缓存已覆盖请求区间, 不应重新拉取")

    monkeypatch.setattr(tune_mod, "prepare_macro", fail_fetch)
    out = tune_mod.node_prepare_macro({
        "tune_base_dir": base,
        "start_date": 20231201,
        "end_date": 20231231,
        "benchmark": "1A0001",
        "warmup_months": 0,   # required months 仅 202312, fresh_path 已覆盖
    })
    assert out["dret_path"] == fresh_path


def test_prepare_macro_intramonth_date_mismatch_regenerates(tmp_path, monkeypatch):
    """同月不同日 (如 20221116 vs 20221117) 不得因月份相同 (202211) 而误判为 Cache Hit。"""
    import polars as pl

    base = os.path.join(str(tmp_path), "tune_task")
    os.makedirs(base, exist_ok=True)
    calls = []

    def fake_prepare_macro(start_date, end_date, benchmark):
        calls.append((start_date, end_date))
        uni = pl.LazyFrame({"sid": ["a"], "first_trading": ["20200101"]})
        daily = pl.LazyFrame({
            "sid": ["a", "a"],
            "day": ["20221018", str(end_date)],
            "close": [1.0, 2.0], "high": [1.1, 2.1], "low": [0.9, 1.9],
            "volume": [100.0, 100.0], "amount": [100.0, 200.0],
        })
        return uni, daily

    monkeypatch.setattr(tune_mod, "prepare_macro", fake_prepare_macro)
    # 第 1 次: 拉取至 20221116 (周三)
    tune_mod.node_prepare_macro({
        "tune_base_dir": base,
        "start_date": 20221101,
        "end_date": 20221116,
        "benchmark": "1A0001",
        "warmup_months": 1,
        "days_since_ipo": 0,
    })
    assert len(calls) == 1

    # 第 2 次: 请求扩展至同月次日 20221117 (周四) -> 必须触发 Cache Miss 重拉!
    tune_mod.node_prepare_macro({
        "tune_base_dir": base,
        "start_date": 20221101,
        "end_date": 20221117,
        "benchmark": "1A0001",
        "warmup_months": 1,
        "days_since_ipo": 0,
    })
    assert len(calls) == 2, "同月内 end_date 从 20221116 推进到 20221117 时必须重新拉取"


# ---------------------------------------------------------------------------
# M11b: 模型路径 legacy 回退 (utils.months)
# ---------------------------------------------------------------------------

def test_find_model_path_falls_back_to_legacy_dir(tmp_path, monkeypatch):
    import bt_studio.utils.months as months_mod
    from bt_studio.utils.months import find_model_path

    tmp = str(tmp_path)
    task_dir = os.path.join(tmp, "task_models")
    legacy_dir = os.path.join(tmp, "legacy_models")
    os.makedirs(task_dir, exist_ok=True)
    os.makedirs(legacy_dir, exist_ok=True)
    legacy_model = os.path.join(legacy_dir, "model_202301.pkl")
    with open(legacy_model, "wb") as f:
        f.write(b"stub")

    monkeypatch.setattr(months_mod, "TUNE_MODEL_DIR", legacy_dir)
    common = {"tune_model_dir": task_dir}

    assert find_model_path(202301, common) == legacy_model
    assert find_model_path(202302, common) is None


def test_node_extract_requires_feature_col():
    with pytest.raises(ValueError, match="feature_col"):
        tune_mod.node_extract_feature_monthly(
            [202401], {}, {"eps": 1e-8},
            {"op": "abs", "args": [{"col": "close"}]})


def test_write_model_manifest_handoff_schema(tmp_path):
    from bt_studio.constant import RAW_PIT_VERSION
    from bt_studio.utils.manifest import safe_load_model_ckpt
    ckpt = {"train_end_month": 202412, "config": {"downsample": 4},
            "T1_rets": {"open_5m": 5}, "decay_minutes": 15}
    pkl_path = tmp_path / "model_202412.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(ckpt, f)
    common = {"feature_col": "f_abc123def456", "_eval_data_fp": "d" * 8,
              "_task_id": "t1", "exclude_bars": 10}
    tune_mod._write_model_manifest(str(tmp_path), ckpt, common, None, source="tune")
    data = json.loads((tmp_path / "model_202412.manifest.json").read_text())
    assert data["model_id"] == 202412
    assert data["feature_id"] == "f_abc123def456"
    assert len(data["pkl_sha256"]) == 64
    assert data["label_basis"]["execution_feasibility"] == "bt_core"
    assert data["label_basis"]["signal_cut_bar"] == 229
    assert data["code"]["raw_pit_version"] == RAW_PIT_VERSION

    # 正常签名验证通过
    loaded = safe_load_model_ckpt(str(pkl_path))
    assert loaded["train_end_month"] == 202412

    # 篡改 .pkl 内容 -> 签名校验拒绝反序列化
    with open(pkl_path, "ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        safe_load_model_ckpt(str(pkl_path))
