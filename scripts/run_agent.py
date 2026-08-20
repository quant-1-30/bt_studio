#!/usr/bin/env python3

import json
import os
import sys
from datetime import datetime

import polars as pl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bt_studio.pipeline.preprocess import prepare_macro, prepare_tick, universe_sample
from bt_studio.factor_mining_agent.mining_agent import TwoStageAgentHarness
from bt_studio.factor_mining_agent.llm_loop import RLFeatureFlywheel
from bt_studio.constant import LLM_PENDING_DIR


# --------------------------------------------------------------------------- #
# Stage 3 → spool: agent loop stays self-contained.
#
# The LLM agent and the task_manager run as two INDEPENDENT loops, decoupled
# by a filesystem contract: Stage3 winners are written as pending task JSONs
# under LLM_PENDING_DIR (fire-and-forget). When hardware resources are ready,
# start the orchestrator with BT_STUDIO_LLM_INTAKE=1 — it will consume this
# directory into the serial queue and run_dag for each feature.
# --------------------------------------------------------------------------- #
def real_hpo_trigger(feature_name, ast_recipe, common_config, persisted_paths=None) -> bool:

    print(f"  [Stage3] feature '{feature_name}' passed walk-forward → spooling HPO task")
    task = {
        "ast_recipe": ast_recipe,
        "common_params": {
            "start_date": common_config["start_date"],
            "end_date": common_config["end_date"],
            "benchmark": (common_config["benchmark"].decode()
                          if isinstance(common_config["benchmark"], bytes)
                          else common_config["benchmark"]),
            "train_window": 1, "oss_step": 1,
            "feature_col": feature_name,
            "regime_filter": common_config["regime_filter"],
            "T1_rets": common_config["T1_rets"],
            "topk": 5, "trigger": 5,
            "u_pval": 0.50,   # relaxed for the mini dataset
            "num_workers": 2, "n_startup_trials": 2,
            "storage_path": "/tmp/ray_results_agent_test",
        },
        "search_bounds": {
            "downsample": [4], "motif_minutes": [60],
            "threshold_r": [0.55, 0.75], "num_trials": 2,
        },
        "precomputed_features": persisted_paths or [],
        "created_at": datetime.now().isoformat(),
        "dag": "wfo_hpo",
        "source": "llm_agent",
    }

    try:
        os.makedirs(LLM_PENDING_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(LLM_PENDING_DIR, f"hpo_{feature_name}_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(task, f, ensure_ascii=False, indent=2, default=str)
        print(f"  [Stage3] spooled → {path} (consumed by task_manager when intake enabled)")
        return True
    except Exception as e:
        print(f"  [Stage3] spool failed: {e}")
        return False

# --------------------------------------------------------------------------- #
# Dummy LLM: returns a fixed OFI-reconstruction recipe (markdown-wrapped)
# --------------------------------------------------------------------------- #
def dummy_llm_call(system_prompt: str, user_prompt: str) -> str:
    return """```json
{
    "hypothesis_id": "hyp_reconstructed_ofi",
    "economic_reasoning": "sign(close.diff)*amount 累计占比,截面去均值还原 OFI。",
    "sub_features": [
        [
            {"name": "ofi_dir", "ast": {"op": "sign", "args": [{"op": "delta", "args": [{"col": "close"}, 1]}]}},
            {"name": "ofi_signed_amt", "ast": {"op": "mul", "args": [{"col": "ofi_dir"}, {"col": "amount"}]}},
            {"name": "ofi_cum_sa", "ast": {"op": "cum_sum", "args": [{"col": "ofi_signed_amt"}]}},
            {"name": "ofi_cum_amt", "ast": {"op": "cum_sum", "args": [{"col": "amount"}]}},
            {"name": "final_ofi", "ast": {"op": "cs_demean", "args": [{"op": "div", "args": [{"col": "ofi_cum_sa"}, {"col": "ofi_cum_amt"}]}]}}
        ]
    ]
}
```"""


def main():
    """End-to-end agent demo: RLFeatureFlywheel drives the whole loop.

    Flow: flywheel.step() → (dummy) LLM → parse → TwoStageAgentHarness.run()
        → Stage3 callback submits a real (mini) WFO HPO job.
    """

    common_config = {
        "start_date": 20221001, "end_date": 20221231,
        "benchmark": b"1A0001",
        "eps": 1e-8, "min_factor_weight": 0.05,
        "top_k_ratio": 0.25, "days_since_ipo": 120,
        "exclude_bars": 10,
        "T1_rets": {"open_5m": 5, "open_15m": 15},
        "ranking_window": 5, "ranking_ratio": 0.25,
        "trigger": 30, "topk": 5, "alternative": "greater",
        "decay_minutes": 15, "dtw_window_frac": 0.1,
        "regime_filter": {"ma_window": 5},
        # walk-forward verdict knob consumed by the harness
        "min_window_pass_ratio": 0.6,
    }

    print("[run_agent] fetching macro data...")
    universe_lf, daily_lf = prepare_macro(
        start_date=common_config["start_date"],
        end_date=common_config["end_date"],
        benchmark=common_config["benchmark"],
    )
    filtered = universe_sample(universe_lf, daily_lf, common_config)
    sids = (filtered.select(pl.col("sid").cast(pl.Binary)).unique()
            .collect().to_series().to_list())
    print(f"[run_agent] universe sids: {len(sids)}")

    snapshot = prepare_tick(
        start_date=common_config["start_date"],
        end_date=common_config["end_date"],
        sids=sids[:50],          # cap for demo speed
    )
    if not snapshot:
        print("[run_agent] no tick data; abort")
        return
    hf_lf = (pl.concat(list(snapshot.values()))
             .rename({"minute_idx": "bar_idx"}).lazy())

    harness = TwoStageAgentHarness(
        common_config=common_config,
        stage3_callback=real_hpo_trigger,
    )
    flywheel = RLFeatureFlywheel(llm_call_fn=dummy_llm_call, harness=harness)

    target = ("挖掘 A 股尾盘 (14:30 后) 的微观结构 alpha: "
              "偏好 order-flow imbalance / momentum / volatility 类因子")
    results = flywheel.step(target, hf_lf, daily_lf.lazy())

    print("\n=== Flywheel Results ===")
    for r in results:
        print(f"{r.feature_name}: stage={r.stage_passed} "
              f"ratio={r.window_pass_ratio:.0%} score={r.final_score:.2f}")
    print(f"replay buffer: {len(flywheel.buffer.successful_history)} success / "
          f"{len(flywheel.buffer.failed_history)} failure")


if __name__ == "__main__":
    main()