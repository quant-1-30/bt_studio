#!/usr/bin/env python3
"""tune.py 全链路 HPO 集成测试(无外部 RPC / 无真实行情依赖)。

覆盖链路:
    node_prepare_macro (cache-hit)
    -> node_extract_feature_monthly (ast_recipe 编译 + 按月 PIT 落盘)
    -> node_prepare_train (ray.put)
    -> node_tune_monthly (Ray Tune + Optuna TPE + collapse/fANOVA/Pareto 门控)
    -> node_check_decay_monthly
    -> node_update_fsm_matrix
    -> node_oos_inference_monthly (FSMPredictor)

数据为合成分钟 bar: 在约 40% 的 (sid, day) 上植入一个日内容积形态(motif))
并令这些样本的 T+1 早盘收益显著为正 —— HPO 应能从随机超参中恢复出该形态。

同时内置三条未来函数(lookahead)回归断言:
    1. compiler 因果守卫拒绝 shift(-N) / backward-fill 类 AST;
    2. prepare_curves 对尾部 exclude_bars 做 NaN 屏蔽(特征时点 <= 入场 bar);
    3. FSMPredictor 对"未来数据扰动不变": 篡改 T 日之后的全部数据,
       T-1 日及以前的打分必须逐行一致。
"""

import os
import sys
import pickle
import logging
import tempfile

import numpy as np
import pandas as pd
import polars as pl
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import ray
import mlflow

import bt_studio.tune as tune_mod
import bt_studio.utils.diagnostics.hpo_check as hpo_check_mod
import bt_studio.engine.executor as tasking_mod
from bt_studio.compiler.ast import compile_ast, ASTCompilationError
from bt_studio.default_config import build_exp_config
from bt_studio.engine import parse_dag, run_dag
from bt_studio.pipeline.inference import FSMPredictor
from bt_studio.pipeline.patterns.astc import prepare_curves
from bt_studio.pipeline.preprocess.panel import build_fsm_panel
from bt_studio.engine.executor import attach_task_dirs


# ============================================================================
# 合成数据
# ============================================================================

N_SIDS = 10
DAYS_PER_MONTH = 10
MONTHS = ["2024-01", "2024-02", "2024-03", "2024-04"]  # 前 3 月训练, 第 4 月 OOS
TRAIN_MONTHS = [202401, 202402, 202403]
OOS_MONTHS = [202404]
FEATURE_COL = "sig"          # ast_recipe 最后一步的名字
TRIGGER_RATE = 0.40


def _synth_days() -> list[pd.Timestamp]:
    """每月取前 DAYS_PER_MONTH 个工作日, 保持时间有序。"""
    cal = pd.bdate_range("2024-01-02", "2024-04-30")
    by_month: dict[str, list] = {}
    for d in cal:
        by_month.setdefault(d.strftime("%Y-%m"), []).append(d)
    out = []
    for m in MONTHS:
        out.extend(by_month.get(m, [])[:DAYS_PER_MONTH])
    return out


def _gen_hf_and_daily(seed: int = 7):
    """生成分钟 hf 帧 (含 raw_sig 列) 与日线帧。

    因果结构:
      - day D 为 trigger 日 => raw_sig 在 bars 20..85 植入形态, 且
        day D+1 早盘 open/5m/30m 价格较 D 日 14:50 入场价高 ~+2%;
      - 非 trigger 日 => 纯噪声特征, D+1 早盘 ~-1%。
      - 日线 close 缓慢上行 => regime_signal 恒为 1 (不误杀曲线)。
    """
    rng = np.random.RandomState(seed)
    days = _synth_days()
    sids = [f"60{i:04d}" for i in range(N_SIDS)]

    hf_parts, daily_rows = [], []
    for sid_i, sid in enumerate(sids):
        base = 100.0 + 3.0 * sid_i
        prev_trigger, prev_entry = False, base
        for d_i, day in enumerate(days):
            base = base * (1.0 + 0.001)          # 缓慢上行 -> regime=1
            trigger = rng.random() < TRIGGER_RATE

            # ---- T+1 早盘与前日入场价挂钩 (标签侧, 仅为 label) ----
            gap = (0.020 + rng.uniform(0, 0.004)) if prev_trigger \
                else -(0.008 + rng.uniform(0, 0.004))
            open_px = prev_entry * (1.0 + gap)
            drift = 0.003 if prev_trigger else -0.002

            close = np.empty(240)
            open_ = np.empty(240)
            close[:30] = open_px * (1.0 + drift * np.arange(30) / 30.0)
            open_[0] = open_px
            open_[1:30] = close[:29]
            # 午后向当日 base 回归, 尾盘(入场 bar 230)锚定 base
            close[30:230] = np.linspace(close[29], base, 200)
            close[230:] = base
            open_[30:] = close[29:239]   # open[i] ~= close[i-1]
            noise = rng.normal(0, 0.02, 240)
            noise[230:] = 0.0
            close = close + noise
            high = np.maximum(open_, close) * 1.0005
            low = np.minimum(open_, close) * 0.9995
            volume = np.full(240, 1000.0)
            amount = volume * close

            # ---- 特征侧: 日内形态 (causal, 仅用当日已发生 bar) ----
            sig = rng.normal(0, 0.05, 240)
            if trigger:
                ramp = np.linspace(0, 1, 41)          # bars 20..60 上行
                decay = np.linspace(1, 0, 25)         # bars 61..85 回落
                sig[20:61] += ramp
                sig[61:86] += decay

            hf_parts.append(pl.DataFrame({
                "day": [day.date()] * 240,
                "sid": [sid] * 240,
                "bar_idx": np.arange(240, dtype=np.int32),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "amount": amount,
                "raw_sig": sig,
            }))
            daily_rows.append({
                "day": day.date(), "sid": sid,
                "close": float(base), "high": float(base * 1.001),
                "low": float(base * 0.999), "volume": 240000.0,
                "amount": 240000.0 * base,
            })

            prev_trigger, prev_entry = trigger, float(close[230])

    hf_df = pl.concat(hf_parts)
    daily_df = pl.DataFrame(daily_rows)
    return hf_df, daily_df


AST_RECIPE = [
    {"name": FEATURE_COL, "ast": {"op": "mul", "args": [{"col": "raw_sig"}, 1.0]}},
]


def _build_exp_config(tmp: str) -> dict:
    return build_exp_config({
        "run_name": "hpo_full_chain_test",
        "common_params": {
            # 合成数据区间为 202401-202403 (OOS 测试另用 dret_full); 显式覆盖
            # PRODUCTION 2010 默认区间, 避免 macro 缓存把 2024 synthetic 误当历史缺口。
            "start_date": 20240101,
            "end_date": 20240331,
            "warmup_months": 0,
            "train_window": 3,
            "oos_step": 1,
            "T1_rets": {"exit_5m": 5, "exit_30m": 30},
            "decay_minutes": 15,
            "exclude_bars": 10,
            "dtw_window_frac": 0.1,
            "ranking_window": 5,
            "topk": 3,
            "trigger": 10,
            "alternative": "greater",
            "u_pval": 0.5,            # 测试放宽统计门控
            "min_excess_ret": 0.0,    # 测试关闭经济阈值硬门控 (专测见 test_score_gates.py)
            "eps": 1e-8,
            "num_workers": 1,       # 串行 trial: TPE(seed=42) 建议序列确定,
                                    # 消除并发完成顺序导致的统计门控随机翻转
            "n_startup_trials": 6,
            "num_trials": 22,
            "seed": 42,
            "storage_path": os.path.join(tmp, "ray_results"),
            "eval_workers": 2,
        },
        "search_spec": {
            "downsample": {"type": "choice", "values": [3, 5]},
            "motif_minutes": {"type": "choice", "values": [30, 45]},
            # threshold_r 上界 0.85 时强种植信号的最优解贴边 (0.845), 塌陷
            # 检测的邻域只剩单侧 → 误判 ISOLATED_SPIKE (经济真实评分的动态
            # 范围大于旧 z-clip 掩蔽后的尺度)。扩到 0.95 使最优解有双侧邻居,
            # 高原可测; 检测器严格度不变。
            "threshold_r": {"type": "uniform", "low": 0.65, "high": 0.95},
        },
    })


@pytest.fixture(scope="module")
def chain_env(tmp_path_factory):
    """准备缓存 parquet / monkeypatch 输出目录 / ray 本地集群。"""
    tmp = str(tmp_path_factory.mktemp("hpo_chain"))
    hf_df, daily_df = _gen_hf_and_daily()

    feat_dir = os.path.join(tmp, "features")
    TUNE_BASE_DIR = os.path.join(tmp, "tune")
    for d in (feat_dir, os.path.join(TUNE_BASE_DIR, "models"),
              os.path.join(TUNE_BASE_DIR, "scores"), os.path.join(TUNE_BASE_DIR, "collapse")):
        os.makedirs(d, exist_ok=True)

    # 训练用 dret(仅训练月, 决定 last-trainable 窗口) 与全量 dret(供 decay/oos 用)
    # run_name 会激活任务分桶 -> macro 缓存预置到分桶路径, 避免触发真实 RPC
    exp_cfg = _build_exp_config(tmp)
    dret_full = os.path.join(tmp, "daily_full.parquet")
    month_expr = (pl.col("day").dt.year() * 100 + pl.col("day").dt.month()).alias("_m")
    daily_train = daily_df.with_columns(month_expr).filter(
        pl.col("_m").is_in(TRAIN_MONTHS)).drop("_m")
    daily_df.write_parquet(dret_full)

    # RPC mock: prepare_tick 返回按 sid 分桶的合成分钟帧。
    # [确定性] 不要用 set 保存 want: set 迭代序受 PYTHONHASHSEED 影响,
    # 会改变返回 dict 的插入序 → pl.concat 行序 → 特征 parquet 逐进程漂移
    # (曾导致 collapse 门控在全量套件中随机翻转)。
    def fake_prepare_tick(start_date, end_date, sids, *a, **kw):
        want = list(dict.fromkeys(
            s.decode() if isinstance(s, bytes) else s for s in sids
        ))
        frames = [
            hf_df.filter(
                (pl.col("sid") == sid)
                & pl.col("day").is_in(
                    daily_df.filter(pl.col("sid") == sid)["day"].to_list()
                )
            ).lazy()
            for sid in want
        ]
        return pl.concat(frames) if frames else None

    saved_paths = {
        "FEATURE_DIR": tune_mod.FEATURE_DIR,
        "TUNE_BASE_DIR": tune_mod.TUNE_BASE_DIR,
        "TUNE_MODEL_DIR": tune_mod.TUNE_MODEL_DIR,
        "TUNE_SCORE_DIR": tune_mod.TUNE_SCORE_DIR,
        "TASKfeature_dir": tasking_mod.FEATURE_DIR,
        "TASKtune_base_dir": tasking_mod.TUNE_BASE_DIR,
    }
    saved_tick = tune_mod.prepare_tick
    saved_collapse_dir = hpo_check_mod.TUNE_COLLAPSE_DIR

    tune_mod.FEATURE_DIR = feat_dir
    tune_mod.TUNE_BASE_DIR = TUNE_BASE_DIR
    tune_mod.TUNE_MODEL_DIR = os.path.join(TUNE_BASE_DIR, "models")
    tune_mod.TUNE_SCORE_DIR = os.path.join(TUNE_BASE_DIR, "scores")
    # [任务隔离] 分桶根目录常量归 tasking 模块所有, 必须一并 patch,
    # 否则 run_name 激活的分桶会逃逸到仓库真实 result/ 目录
    tasking_mod.FEATURE_DIR = feat_dir
    tasking_mod.TUNE_BASE_DIR = TUNE_BASE_DIR
    tune_mod.prepare_tick = fake_prepare_tick
    hpo_check_mod.TUNE_COLLAPSE_DIR = os.path.join(TUNE_BASE_DIR, "collapse")

    # run_name -> task_id 分桶; 预置分桶内 macro 缓存 (cache-hit 路径)
    task_id = attach_task_dirs(exp_cfg, None, common_config=exp_cfg["common_params"])
    assert task_id == "hpo_full_chain_test"
    task_root = os.path.join(TUNE_BASE_DIR, task_id)
    os.makedirs(task_root, exist_ok=True)
    daily_train.write_parquet(os.path.join(task_root, "global_daily.parquet"))

    # 隔离外部 MLflow: .env 里的 http tracking URI 会让 node_tune_monthly
    # 挂上 MLflowLoggerCallback, 而本机服务/代理返回 403 时回调直接把 tune 节点炸掉
    saved_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"file://{os.path.join(tmp, 'mlruns')}")

    # [确定性] 单线程 env 由 tests/conftest.py 在 polars/numba import 之前
    # 注入 driver 进程; raylet 在此处继承 driver env, worker 亦为单线程。
    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True,
             logging_level=logging.ERROR)
    try:
        yield {
            "tmp": tmp, "hf_df": hf_df, "daily_df": daily_df,
            "dret_full": dret_full,
            "feat_dir": os.path.join(feat_dir, task_id),
            "model_dir": os.path.join(task_root, "models"),
            "score_dir": os.path.join(task_root, "scores"),
            "TUNE_BASE_DIR": TUNE_BASE_DIR, "exp_config": exp_cfg,
        }
    finally:
        ray.shutdown()
        mlflow.set_tracking_uri(saved_uri)
        tune_mod.FEATURE_DIR = saved_paths["FEATURE_DIR"]
        tune_mod.TUNE_BASE_DIR = saved_paths["TUNE_BASE_DIR"]
        tune_mod.TUNE_MODEL_DIR = saved_paths["TUNE_MODEL_DIR"]
        tune_mod.TUNE_SCORE_DIR = saved_paths["TUNE_SCORE_DIR"]
        tasking_mod.FEATURE_DIR = saved_paths["TASKfeature_dir"]
        tasking_mod.TUNE_BASE_DIR = saved_paths["TASKtune_base_dir"]
        tune_mod.prepare_tick = saved_tick
        hpo_check_mod.TUNE_COLLAPSE_DIR = saved_collapse_dir


# ============================================================================
# 1. compiler 因果守卫
# ============================================================================

def test_causal_guard_rejects_lookahead():
    for bad in (
        {"op": "ref", "args": [{"col": "close"}, -1]},
        {"op": "delta", "args": [{"col": "close"}, -5]},
        {"col": "next_close"},
    ):
        with pytest.raises(ASTCompilationError):
            compile_ast(bad, check_causal=True)
    # 合法回看窗口可以通过
    expr, name = compile_ast({"op": "ref", "args": [{"col": "close"}, 1]})
    assert name == "ref(close,1)" and expr is not None


# ============================================================================
# 2. 全链路: wfo_hpo DAG (macro -> extract -> train_data -> tune)
# ============================================================================

def test_hpo_dag_end_to_end(chain_env):
    env = chain_env
    res = run_dag(parse_dag("wfo_hpo"), env["exp_config"], ast_recipe=AST_RECIPE)

    assert res.windows_executed == 1
    assert res.models_produced == [TRAIN_MONTHS[-1]], res.models_produced

    # 特征按月落盘 (compiler 产物, PIT 分区)
    for ym in TRAIN_MONTHS:
        p = os.path.join(env["feat_dir"], f"hf_{FEATURE_COL}_{ym}.parquet")
        assert os.path.exists(p), p
        part = pl.read_parquet(p)
        assert part.height == N_SIDS * DAYS_PER_MONTH * 240
        assert {"day", "sid", "bar_idx", FEATURE_COL} <= set(part.columns)

    # 模型 checkpoint 结构
    model_id = res.last_model_id
    with open(os.path.join(env["model_dir"], f"model_{model_id}.pkl"), "rb") as f:
        ckpt = pickle.load(f)
    assert ckpt["config"]["m"] > 0 and ckpt["config"]["threshold_d"] > 0
    assert np.asarray(ckpt["motif"]).size > 0
    macro_mat = np.asarray(ckpt["fsm_matrix"]["P(exit_5m|Macro)"])
    assert macro_mat.shape == (3, 5)          # 3 宏观态 x 5 收益分位
    # [回归] 状态收益向量长度必须与转移矩阵列数一致 (FSMPredictor np.dot)
    for vec in ckpt["fsm_matrix"]["state_return_vectors"].values():
        assert len(vec) == macro_mat.shape[1]


# ============================================================================
# 3. decay / matrix update / OOS 推理
# ============================================================================

def test_decay_update_and_oos_inference(chain_env):
    env = chain_env
    common = env["exp_config"]["common_params"]
    model_id = TRAIN_MONTHS[-1]
    train_paths = [os.path.join(env["feat_dir"], f"hf_{FEATURE_COL}_{ym}.parquet")
                   for ym in TRAIN_MONTHS]

    # OOS 月特征: 走 node_extract_feature_monthly 的 raw_hf_lf 分支 (同样过 compiler)
    oos_hf = env["hf_df"].filter(
        (pl.col("day").dt.year() * 100 + pl.col("day").dt.month()).is_in(OOS_MONTHS)
    ).lazy()
    universe_sids = {ym: env["daily_df"]["sid"].unique().to_list()
                     for ym in TRAIN_MONTHS + OOS_MONTHS}
    oos_paths = tune_mod.node_extract_feature_monthly(
        OOS_MONTHS, universe_sids, common, AST_RECIPE, raw_hf_lf=oos_hf)
    assert len(oos_paths) == 1 and os.path.exists(oos_paths[0])

    # --- decay: 模型在训练尾月应保持健康 ---
    need_tune = tune_mod.node_check_decay_monthly(
        model_id, [TRAIN_MONTHS[-1]], env["dret_full"], train_paths, common)
    assert need_tune is False, "planted-signal 模型在训练尾月不应判定衰减"

    # --- matrix update: 继承 motif, 只更新转移矩阵 ---
    new_id = tune_mod.node_update_fsm_matrix(
        OOS_MONTHS[-1], model_id, env["dret_full"], train_paths, common)
    assert new_id == OOS_MONTHS[-1]
    with open(os.path.join(env["model_dir"], f"model_{new_id}.pkl"), "rb") as f:
        upd = pickle.load(f)
    with open(os.path.join(env["model_dir"], f"model_{model_id}.pkl"), "rb") as f:
        old = pickle.load(f)
    assert np.allclose(np.asarray(upd["motif"]), np.asarray(old["motif"]))

    # --- OOS 推理: FSMPredictor 全链路 (此处曾因 4/5 状态数不一致必崩) ---
    # 传 train_paths: 节点内部按 ceil(ranking_window/15) 取训练尾月作
    # macro_state 滚动分位回填历史; 打分输出必须仍只含 OOS 月
    # (下方 months 断言守护 warmup 泄漏)
    tune_mod.node_oos_inference_monthly(
        model_id, env["dret_full"], OOS_MONTHS, oos_paths, common,
        train_paths=train_paths)
    score_path = os.path.join(env["score_dir"], f"scores_{model_id}.parquet")
    assert os.path.exists(score_path)
    scores = pl.read_parquet(score_path)
    assert {"day", "sid", "fsm_score"} <= set(scores.columns)
    assert scores.height >= 5
    assert scores["fsm_score"].abs().sum() > 0
    # 触发日集中于 OOS 月内 (月份过滤生效)
    months = scores.with_columns(
        (pl.col("day").dt.year() * 100 + pl.col("day").dt.month()).alias("_m")
    )["_m"].unique().to_list()
    assert set(months) == set(OOS_MONTHS)


# ============================================================================
# 4. 未来函数回归: 尾部屏蔽 + FSMPredictor 对未来扰动不变
# ============================================================================

def test_curve_tail_masked(chain_env):
    env = chain_env
    common = env["exp_config"]["common_params"]
    with open(os.path.join(env["model_dir"],
                           f"model_{TRAIN_MONTHS[-1]}.pkl"), "rb") as f:
        ckpt = pickle.load(f)
    ds = ckpt["config"]["downsample"]
    oos_hf = pl.scan_parquet(
        os.path.join(env["feat_dir"], f"hf_{FEATURE_COL}_{OOS_MONTHS[0]}.parquet"))
    panel = build_fsm_panel(oos_hf, pl.scan_parquet(env["dret_full"]),
                            ckpt["config"], common, is_train=False).collect()
    curves = prepare_curves(panel, ckpt["config"], common)
    n_mask = common["exclude_bars"] // ds
    assert n_mask >= 1
    assert np.isnan(curves[:, -n_mask:]).all(), "入场 bar 之后的数据必须被 NaN 屏蔽"


def test_predictor_invariant_to_future_perturbation(chain_env):
    """篡改 T 日之后的全部行情/特征, T-1 日及更早的打分必须不变。"""
    env = chain_env
    common = env["exp_config"]["common_params"]
    model_id = TRAIN_MONTHS[-1]
    with open(os.path.join(env["model_dir"], f"model_{model_id}.pkl"), "rb") as f:
        ckpt = pickle.load(f)
    predictor = FSMPredictor(ckpt, common)

    oos_hf = pl.scan_parquet(
        os.path.join(env["feat_dir"], f"hf_{FEATURE_COL}_{OOS_MONTHS[0]}.parquet"))
    days_sorted = oos_hf.select("day").unique().sort("day").collect()["day"]
    cut_day = days_sorted[len(days_sorted) // 2]           # 中间一天为 T

    full_scores = predictor.predict(build_fsm_panel(
        oos_hf, pl.scan_parquet(env["dret_full"]), ckpt["config"], common,
        is_train=False))

    # 未来侧暴力扰动: 特征 x100, 价格路径打乱
    hf_future = oos_hf.filter(pl.col("day") > cut_day)
    hf_future = hf_future.with_columns(
        pl.col(FEATURE_COL) * 100.0
    ).with_columns(
        (pl.col("close") * 2.0).alias("close"),
        (pl.col("open") * 2.0).alias("open"),
    )
    perturbed = pl.concat([oos_hf.filter(pl.col("day") <= cut_day), hf_future])
    pert_scores = predictor.predict(build_fsm_panel(
        perturbed, pl.scan_parquet(env["dret_full"]), ckpt["config"], common,
        is_train=False))

    left = full_scores.filter(pl.col("day") < cut_day).sort(["day", "sid"])
    right = pert_scores.filter(pl.col("day") < cut_day).sort(["day", "sid"])
    assert left.height == right.height > 0
    assert left["sid"].to_list() == right["sid"].to_list()
    np.testing.assert_allclose(
        left["fsm_score"].to_numpy(), right["fsm_score"].to_numpy(), rtol=1e-9,
        err_msg="T 日之后的数据影响了 T-1 日及以前的打分 -> 存在未来函数")
