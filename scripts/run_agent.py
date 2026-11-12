#!/usr/bin/env python3

from __future__ import annotations

import argparse
import cProfile
import glob
import io
import os
import pstats
from datetime import datetime
from typing import Any, Dict

import numpy as np
import polars as pl
from dotenv import load_dotenv

from bt_studio.agent import RLFeatureFlywheel, TwoStageAgentHarness
from bt_studio.constant import LLM_PENDING_DIR, LLM_PROFILE_DIR, LLM_RUN_DIR
from bt_studio.default_config import PRODUCTION_COMMON_PARAMS, PRODUCTION_SEARCH_SPEC
from bt_studio.engine.ctx import resolve_platform_concurrency
from bt_studio.tune import node_prepare_macro
from bt_studio.utils.io import atomic_save_json, read_json_safe, sha1_hex
from bt_studio.utils.months import resolve_walk_forward_windows, ym_range

_RESOURCE_KEYS = {
    "eval_procs", "num_workers", "storage_path", "last_model_id",
    "task_id", "run_name", "_task_id", "_eval_data_fp",
}

DEFAULT_MINING_TARGET = (
    "挖掘 A 股尾盘 (14:30 后) 的微观结构 Alpha: "
    "聚焦订单流失衡 (OFI)、量价背离、动量加速度与波动率压缩突破，预测 T 日 14:50 入场至 T+1 早盘的超额收益。"
)


def spool_stage3_hpo_task(feature_name: str, ast_recipe: Any, common_config: Dict[str, Any]) -> bool:
    """Spool a Stage 2 winner feature as an atomic pending HPO task JSON for AsyncTaskManager."""
    n_months = len(ym_range(common_config["start_date"] // 100, common_config["end_date"] // 100))
    train_win = min(int(common_config.get("train_window", PRODUCTION_COMMON_PARAMS["train_window"])), max(1, n_months))

    common_params: Dict[str, Any] = {
        **PRODUCTION_COMMON_PARAMS,
        **{k: v for k, v in common_config.items() if k in PRODUCTION_COMMON_PARAMS},
        "benchmark": str(common_config["benchmark"]),
        "train_window": train_win,
        "oos_step": int(common_config.get("oos_step", PRODUCTION_COMMON_PARAMS["oos_step"])),
        "feature_col": feature_name,
        "num_workers": resolve_platform_concurrency()["num_workers"],
    }
    if isinstance(common_config.get("prior_config"), dict) and common_config["prior_config"]:
        common_params["prior_config"] = dict(common_config["prior_config"])

    task = {
        "feature_col": feature_name,
        "ast_recipe": ast_recipe,
        "common_params": common_params,
        "search_spec": PRODUCTION_SEARCH_SPEC,
        "created_at": datetime.now().isoformat(),
        "dag": os.environ.get("AGENT_STAGE3_DAG", "wfo_hpo"),
        "source": "llm_agent",
    }

    try:
        os.makedirs(LLM_PENDING_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(LLM_PENDING_DIR, f"hpo_{feature_name}_{ts}.json")
        atomic_save_json(task, path)
        print(f"  [Stage3] spooled -> {path}")
        return True
    except Exception as e:
        print(f"  [Stage3] spool failed: {e}")
        return False


def build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run long-running LLM + RL Feature Mining Flywheel")
    p.add_argument("--start-date", type=int, default=int(os.environ.get("FROM_DATE", PRODUCTION_COMMON_PARAMS["start_date"])))
    p.add_argument("--end-date", type=int, default=int(os.environ.get("TO_DATE", PRODUCTION_COMMON_PARAMS["end_date"])))
    p.add_argument("--train-window", type=int, default=int(os.environ.get("TRAIN_WINDOW", PRODUCTION_COMMON_PARAMS["train_window"])), help="Walk-Forward training window (months)")
    p.add_argument("--oos-step", type=int, default=int(os.environ.get("OOS_STEP", PRODUCTION_COMMON_PARAMS["oos_step"])), help="Walk-Forward OOS step interval (months)")
    p.add_argument("--n-sids", type=int, default=int(os.environ.get("N_SIDS", "0")), help="Max sampled sids (0 = full PIT universe)")
    p.add_argument("--steps", type=int, default=int(os.environ.get("AGENT_STEPS", "100")), help="Total RL flywheel iterations")
    p.add_argument("--vendor", default=os.environ.get("LLM_PROVIDER", "zhipu"), help="LLM provider (zhipu | google | kimi | dummy)")
    p.add_argument("--stage3", choices=("spool", "off"), default=os.environ.get("AGENT_STAGE3", "off"))
    p.add_argument("--resume", choices=("auto", "off"), default=os.environ.get("RESUME", "auto"))
    p.add_argument("--target", default=DEFAULT_MINING_TARGET, help="Economic mining objective prompt")
    p.add_argument("--profile", action="store_true", help="Enable cProfile sampling and save to LLM_PROFILE_DIR")
    return p


def run_mining_flywheel(args: argparse.Namespace) -> None:
    load_dotenv()
    os.environ["LLM_PROVIDER"] = args.vendor

    common_config: Dict[str, Any] = {
        **PRODUCTION_COMMON_PARAMS,
        "start_date": int(args.start_date),
        "end_date": int(args.end_date),
        "train_window": int(args.train_window),
        "oos_step": int(args.oos_step),
        "u_pval": float(os.environ.get("AGENT_SCREEN_MAX_PVAL", PRODUCTION_COMMON_PARAMS["u_pval"])),
        "min_window_pass_ratio": float(
            os.environ.get("AGENT_SCREEN_MIN_WINDOW_PASS_RATIO", PRODUCTION_COMMON_PARAMS["min_window_pass_ratio"])
        ),
        "eval_procs": resolve_platform_concurrency()["eval_procs"],
    }

    macro_res = node_prepare_macro(common_config)
    dret_lf = pl.scan_parquet(macro_res["dret_path"])
    universe_sids: Dict[int, list[bytes]] = {
        int(ym): [s if isinstance(s, bytes) else str(s).encode() for s in sids]
        for ym, sids in macro_res["universe_sids"].items()
    }
    all_sids = sorted({s for sids in universe_sids.values() for s in sids})
    if not all_sids:
        raise RuntimeError("[run_agent] Empty PIT universe returned by node_prepare_macro")

    if args.n_sids > 0 and len(all_sids) > args.n_sids:
        rng = np.random.default_rng(int(common_config["seed"]))
        chosen = sorted(rng.choice(all_sids, size=args.n_sids, replace=False).tolist())
        chosen_set = set(chosen)
        universe_sids = {
            ym: [s for s in sids if s in chosen_set]
            for ym, sids in universe_sids.items()
        }
        universe_sids = {ym: sids for ym, sids in universe_sids.items() if sids}
    else:
        chosen = all_sids

    wf_windows = resolve_walk_forward_windows(list(universe_sids.keys()), common_config)
    if not wf_windows:
        raise RuntimeError("[run_agent] No valid walk-forward windows resolved from universe_sids")
    print(
        f"[run_agent] Selected {len(chosen)}/{len(all_sids)} universe sids | "
        f"Walk-Forward windows: {len(wf_windows)} "
        f"(train={common_config['train_window']}m, step={common_config['oos_step']}m, "
        f"{wf_windows[0][0]}..{wf_windows[-1][-1]})"
    )

    data_fp = sha1_hex({
        "start": common_config["start_date"],
        "end": common_config["end_date"],
        "sids": [s.decode() for s in chosen],
    })
    run_common = {
        k: v for k, v in common_config.items()
        if k not in _RESOURCE_KEYS and not str(k).startswith("_")
    }
    run_fp = sha1_hex({"data": data_fp, "common": run_common, "vendor": args.vendor})
    common_config["_eval_data_fp"] = data_fp

    harness = TwoStageAgentHarness(
        common_config=common_config,
        stage3_callback=None if args.stage3 == "off" else spool_stage3_hpo_task,
        universe_sids=universe_sids,
    )

    run_dir = None
    if args.resume == "auto":
        for d in sorted(glob.glob(os.path.join(LLM_RUN_DIR, f"*_{run_fp[:8]}")), reverse=True):
            st_obj = read_json_safe(os.path.join(d, "state.json"))
            if isinstance(st_obj, dict) and int(st_obj.get("step_counter", 0)) < args.steps:
                run_dir = d
                break

    warm_start = os.environ.get("AGENT_WARM_START", "1").lower() not in ("0", "off", "false")
    flywheel = RLFeatureFlywheel(
        harness=harness,
        vendor=args.vendor,
        run_dir=run_dir,
        config_fingerprint=run_fp,
        warm_start=warm_start,
    )

    results = flywheel.run(args.target, None, dret_lf, n_steps=args.steps)
    print("\n=== Flywheel Summary ===")
    for r in results:
        print(f"  {r.feature_name}: stage={r.stage_passed} pass_ratio={r.window_pass_ratio:.0%} score={r.final_score:.2f}")
    print(f"ReplayBuffer: {len(flywheel.buffer.successful_history)} success / {len(flywheel.buffer.failed_history)} failed")


def main() -> None:
    args = build_cli_parser().parse_args()
    if not args.profile:
        run_mining_flywheel(args)
        return

    prof = cProfile.Profile()
    try:
        prof.enable()
        run_mining_flywheel(args)
    finally:
        prof.disable()
        os.makedirs(LLM_PROFILE_DIR, exist_ok=True)
        out = os.path.join(LLM_PROFILE_DIR, f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        stream = io.StringIO()
        stats = pstats.Stats(prof, stream=stream)
        stats.sort_stats("cumulative").print_stats(30)
        stats.dump_stats(out.replace(".txt", ".pstats"))
        with open(out, "w", encoding="utf-8") as f:
            f.write(stream.getvalue())
        print(f"\n[profile] Top-30 cumulative saved: {out}")


if __name__ == "__main__":
    main()
