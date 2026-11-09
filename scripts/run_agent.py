#!/usr/bin/env python3

import os
import sys
from datetime import datetime

# [确定性 + 性能] 必须在任何 polars/numba/stumpy import 之前固定:
# 1. 多线程浮点归约顺序漂移 -> Stage2 score/u_pval 不可复现 (与 tests/conftest.py 同源问题);
# 2. polars_async executor 线程池 (10 线程) 对海量小 streaming collect 的
#    park/unpark 派发 -> 实测 342K Mach traps/s + 62K ctx switch/s 的系统调用风暴。
# 单线程实测代价仅 ~4% (工作负载以单帧算子为主, 线程协调开销反而占优)。
for _var in ("NUMBA_NUM_THREADS", "POLARS_MAX_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import polars as pl
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# .env 必须在任何 provider 构造前加载 (flywheel 默认从 env 读 LLM_PROVIDER)
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

from bt_studio.pipeline.preprocess import prepare_macro, prepare_tick, universe_sample
from bt_studio.agent import TwoStageAgentHarness, RLFeatureFlywheel
from bt_studio.constant import LLM_PENDING_DIR, LLM_RUN_DIR
from bt_studio.default_config import PRODUCTION_COMMON_PARAMS, PRODUCTION_SEARCH_SPEC
from bt_studio.engine.ctx import resolve_platform_concurrency
from bt_studio.utils.io import atomic_save_json, read_json_safe, sha1_hex
from bt_studio.utils.months import calc_warmup_start_date, ym_range

# --------------------------------------------------------------------------- #
# Stage 3 → spool: agent loop stays self-contained.
#
# The LLM agent and the task_manager run as two INDEPENDENT loops, decoupled
# by a filesystem contract: Stage3 winners are written as pending task JSONs
# under LLM_PENDING_DIR (fire-and-forget). When hardware resources are ready,
# start the orchestrator with BT_STUDIO_LLM_INTAKE=1 — it will consume this
# directory into the serial queue and run_dag for each feature.
# --------------------------------------------------------------------------- #
def real_hpo_trigger(feature_name, ast_recipe, common_config) -> bool:
    print(f"  [Stage3] feature '{feature_name}' passed walk-forward → spooling HPO task")
    is_prod = os.environ.get("BT_STUDIO_ENV", "").lower() == "prod"
    n_months = len(ym_range(common_config["start_date"] // 100, common_config["end_date"] // 100))

    if is_prod:
        train_win = min(int(common_config.get("train_window", PRODUCTION_COMMON_PARAMS["train_window"])), max(1, n_months))
        oos_step = int(common_config.get("oos_step", PRODUCTION_COMMON_PARAMS["oos_step"]))
        common_params = {
            **PRODUCTION_COMMON_PARAMS,
            **{k: v for k, v in common_config.items() if k in PRODUCTION_COMMON_PARAMS},
            "benchmark": (common_config["benchmark"].decode()
                          if isinstance(common_config["benchmark"], bytes)
                          else common_config["benchmark"]),
            "train_window": train_win,
            "oos_step": oos_step,
            "feature_col": feature_name,
            "num_workers": resolve_platform_concurrency()["num_workers"],
            "n_startup_trials": int(os.environ.get("HPO_STARTUP_TRIALS", PRODUCTION_COMMON_PARAMS["n_startup_trials"])),
            "num_trials": int(os.environ.get("HPO_NUM_TRIALS", PRODUCTION_COMMON_PARAMS["num_trials"])),
            "storage_path": os.environ.get("RAY_STORAGE_PATH", PRODUCTION_COMMON_PARAMS["storage_path"]),
        }
        search_spec = PRODUCTION_SEARCH_SPEC
    else:
        common_params = {
            "start_date": common_config["start_date"],
            "end_date": common_config["end_date"],
            "benchmark": (common_config["benchmark"].decode()
                          if isinstance(common_config["benchmark"], bytes)
                          else common_config["benchmark"]),
            "train_window": 1, "oos_step": 1,
            "feature_col": feature_name,
            "T1_rets": common_config["T1_rets"],
            "topk": 5, "trigger": 5,
            "u_pval": 0.50,   # relaxed for the mini dataset
            "num_workers": 2, "n_startup_trials": 2,
            # [契约] trial 预算属于 common_params 控制参数; 放入 search_bounds
            # 会被归一化成搜索维度且被生产基线 num_trials=500 静默覆盖。
            "num_trials": 2,
            "storage_path": "/tmp/ray_results_agent_test",
        }
        search_spec = {
            "downsample": {"type": "choice", "values": [4]},
            "motif_minutes": {"type": "choice", "values": [60]},
            "threshold_r": {"type": "uniform", "low": 0.55, "high": 0.75},
        }

    if isinstance(common_config.get("prior_config"), dict) and common_config["prior_config"]:
        common_params["prior_config"] = dict(common_config["prior_config"])

    task = {
        "feature_col": feature_name,
        "ast_recipe": ast_recipe,
        "common_params": common_params,
        "search_spec": search_spec,
        # 预计算复用走纯指纹约定: FEATURE_DIR/{fp12}/,
        # 由 (feature_col, ast_recipe) 推导, 无需显式路径
        "created_at": datetime.now().isoformat(),
        "dag": os.environ.get("AGENT_STAGE3_DAG", "wfo_hpo"),
        "source": "llm_agent",
    }

    try:
        os.makedirs(LLM_PENDING_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(LLM_PENDING_DIR, f"hpo_{feature_name}_{ts}.json")
        # [原子写] 写一半崩溃会留下截断 JSON, 被 intake 判 .invalid 静默丢任务
        atomic_save_json(task, path)
        print(f"  [Stage3] spooled → {path} (consumed by task_manager when intake enabled)")
        return True
    except Exception as e:
        print(f"  [Stage3] spool failed: {e}")
        return False


def _require_prod_env() -> None:
    """生产模式 (BT_STUDIO_ENV=prod) 下挖掘范围必须显式下发 ——
    缺省回退 demo 档 (2022Q4×50) 在生产是静默跑错数据的隐患。"""
    if os.environ.get("BT_STUDIO_ENV", "").lower() != "prod":
        return
    missing = [k for k in ("FROM_DATE", "TO_DATE", "N_SIDS") if not os.environ.get(k)]
    if missing:
        raise SystemExit(
            f"[prod] BT_STUDIO_ENV=prod 需显式配置 {missing} (FROM_DATE/TO_DATE/N_SIDS); "
            f"demo 档请显式设 BT_STUDIO_ENV=dev")


def main():
    _require_prod_env()
    """End-to-end agent demo: RLFeatureFlywheel drives the whole loop.

    Flow: flywheel.step() → (dummy) LLM → parse → TwoStageAgentHarness.run()
        → Stage3 callback submits a real (mini) WFO HPO job.
    """

    common_config = {
        # 生产规模可经环境变量覆盖 (demo 默认: 2022Q4 × 50 标的):
        #   FROM_DATE=20220101 TO_DATE=20230630 N_SIDS=100 AGENT_STEPS=4
        "start_date": int(os.environ.get("FROM_DATE", 20221001)),
        "end_date": int(os.environ.get("TO_DATE", 20221231)),
        "benchmark": b"1A0001",
        "warmup_months": 3,

        "eps": 1e-8,
        "days_since_ipo": 120,

        "exclude_bars": 10,
        "T1_rets":
            {
            "exit_5m": 5,
             "exit_15m": 15
             },

        "ranking_window": 5,
        # ranking_ratio 已移除: 收益分箱在 fsm.py 固化为 5 桶

        "trigger": 30,
        "topk": 5,
        "alternative": "greater",

        # 经济门 (中位数超额硬门): 5 分钟持有 5bp / 15 分钟持有 10bp,
        # 防御"物理成功但无经济含义"的弱特征进入 RL success 池
        "min_excess_ret": {"exit_5m": 5e-4, "exit_15m": 1e-3},

        "decay_minutes": 15,
        "dtw_window_frac": 0.1,

        # walk-forward verdict knob consumed by the harness
        "min_window_pass_ratio": 0.6,
        "seed": 42,

        # [窗口级多进程] Stage2 月份间独立并行 (worker 内单线程, 结果与串行
        # 逐位等价); 根据平台 (macOS vs Linux 64C) 自动适配进程数
        "eval_procs": resolve_platform_concurrency()["eval_procs"],

        # Stage1 物理预筛统计去 warmup 比例 (长窗口因果特征防误杀; 0=旧行为)
        "stage1_trim_warmup_frac": float(os.environ.get("AGENT_STAGE1_TRIM_WARMUP_FRAC", 0.20)),
    }

    # [screen/promote 语义] Stage2 只是 RL 粗筛器; 生产调阈值用运维旋钮,
    # 不改研究默认。promote 到 tune 仍由 AGENT_STAGE3 + 严格 WFO 另行决定。
    common_config["stage2_max_pval"] = float(os.environ.get(
        "AGENT_SCREEN_MAX_PVAL", common_config.get("stage2_max_pval", 0.10)))
    common_config["min_window_pass_ratio"] = float(os.environ.get(
        "AGENT_SCREEN_MIN_WINDOW_PASS_RATIO", common_config["min_window_pass_ratio"]))

    raw_start_date = calc_warmup_start_date(
    common_config["start_date"], 
    common_config.get("warmup_months", 1)
    )

    print("[run_agent] fetching macro data...")
    universe_lf, daily_lf = prepare_macro(
        start_date=raw_start_date,
        end_date=common_config["end_date"],
        benchmark=common_config["benchmark"],
    )
    filtered = universe_sample(universe_lf, daily_lf, common_config)
    # [可复现性] unique() 无序 (polars 哈希聚合, 跨进程漂移), [:50] 截断前必须
    # 排序 —— 否则每次运行选出不同的 50 只标的, 特征/触发/pval 全变,
    # Stage2 判定不可复现 (实测同配置两次重估 4/6 vs 3/6 摆动的根因)
    valid_sids = (
        filtered.select(pl.col("sid").cast(pl.Binary))
        .unique()
        .sort("sid")
        .collect()
        .to_series()
        .to_list()
    )

    print(f"[run_agent] universe sids: {len(valid_sids)}")

    n_sids = int(os.environ.get("N_SIDS", 50))
    if len(valid_sids) > n_sids:
        rng = np.random.default_rng(int(common_config.get("seed", 42)))
        chosen = sorted(rng.choice(valid_sids, size=n_sids, replace=False).tolist())
    else:
        chosen = valid_sids
    hf_lf = prepare_tick(
        start_date=raw_start_date,
        end_date=common_config["end_date"],
        sids=chosen,
    )
    if isinstance(hf_lf, pl.LazyFrame) and hf_lf.collect_schema().names() and hf_lf.select(pl.len()).collect().item() == 0:
        print("[run_agent] no tick data; abort")
        return

    # tune 暂不纳入时设 AGENT_STAGE3=off: Stage2 结束后不写 pending、不触发 HPO 契约。
    # 默认 spool 保持现有文件契约 (仅落盘 result/llm/pending, 由 intake 显式决定是否消费)。
    stage3_mode = os.environ.get("AGENT_STAGE3", "spool").lower()
    stage3_callback = None if stage3_mode == "off" else real_hpo_trigger
    print(f"[run_agent] Stage3 mode: {stage3_mode} (off=特征挖掘+RL only, no HPO spool)")

    # [身份与缓存] 两个指纹分职:
    #   data_fp = 标的池+区间 (EvalResult 缓存的数据身份, 门控不在键内 → 阈值重放免费)
    #   run fp  = data_fp ∪ 门控/网格/vendor (resume 的 run 身份, 改任一即断点失效)
    import json as _json

    data_fp = sha1_hex({"start": common_config["start_date"],
                        "end": common_config["end_date"],
                        "sids": sorted(s.decode() for s in chosen)})
    # run_fp 只含研究/数据语义; 资源与路径旋钮 (eval_procs/num_workers/storage_path/
    # task_id/_*) 变更不应打断 resume 或改变 run 身份。
    _RESOURCE_KEYS = {
        "eval_procs", "num_workers", "storage_path", "last_model_id",
        "task_id", "run_name", "_task_id", "_eval_data_fp",
    }
    run_common = {
        k: v for k, v in common_config.items()
        if k not in _RESOURCE_KEYS and not str(k).startswith("_")
    }
    run_fp = sha1_hex({"data": data_fp, "common": run_common,
                       "vendor": os.environ.get("LLM_PROVIDER", "dummy")})
    common_config["_eval_data_fp"] = data_fp

    harness = TwoStageAgentHarness(
        common_config=common_config,
        stage3_callback=stage3_callback,
    )

    # [resume] RESUME=auto 扫描 runs/ 找指纹匹配且未跑完的 run 目录续跑
    run_dir = None
    if os.environ.get("RESUME", "auto") == "auto":
        import glob as _glob
        for d in sorted(_glob.glob(os.path.join(LLM_RUN_DIR, f"*_{run_fp[:8]}")), reverse=True):
            st_obj = read_json_safe(os.path.join(d, "state.json"))
            if isinstance(st_obj, dict):
                n_done = int(st_obj.get("step_counter", 0) or 0)
                if n_done < int(os.environ.get("AGENT_STEPS", "3")):
                    run_dir = d
                    break

    warm_start = os.environ.get("AGENT_WARM_START", "1").lower() not in ("0", "off", "false")
    flywheel = RLFeatureFlywheel(
        harness=harness, run_dir=run_dir, config_fingerprint=run_fp, warm_start=warm_start)

    target = ("挖掘 A 股尾盘 (14:30 后) 的微观结构 alpha: "
              "偏好 order-flow imbalance / momentum / volatility 类因子")

    # RL 闭环: 每步的成败经 ReplayBuffer 注入下一轮 prompt (LLM 据此演化假设)
    n_steps = int(os.environ.get("AGENT_STEPS", "3"))
    results = flywheel.run(target, hf_lf, daily_lf.lazy(), n_steps=n_steps)

    print("\n=== Flywheel Results (累积) ===")
    for r in results:
        print(f"{r.feature_name}: stage={r.stage_passed} "
              f"ratio={r.window_pass_ratio:.0%} score={r.final_score:.2f}")
    print(f"replay buffer: {len(flywheel.buffer.successful_history)} success / "
          f"{len(flywheel.buffer.failed_history)} failure")


if __name__ == "__main__":
    main()