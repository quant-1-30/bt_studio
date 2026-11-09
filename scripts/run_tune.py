#!/usr/bin/env python3
"""Production WFO entry — wfo_production / wfo_hpo DAG via the engine.

支持两种运行方式:
1. 显式指定特征名 / viable_features.json 中的指纹或 feature_id:
   poetry run python scripts/run_tune.py --feature f_a1b2c3d4e5f6
2. 环境变量或命令行覆盖起止日期、DAG、搜索预算:
   poetry run python scripts/run_tune.py --feature close --dag wfo_production --start-date 20200101 --end-date 20221231
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional, Tuple

# Pin single-thread execution before importing polars/numba/stumpy to guarantee
# bit-identical floating-point reduction order across processes (see docs/AGENTS.md §2).
for _var in ("NUMBA_NUM_THREADS", "POLARS_MAX_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import mlflow

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bt_studio.agent.flywheel import load_viable_features
from bt_studio.default_config import build_exp_config
from bt_studio.engine import parse_dag, run_dag


def _lookup_viable_feature(feature_ref: Optional[str]) -> Tuple[Optional[str], Any, Dict[str, Any]]:
    """从 viable_features.json 解析 (feature_col, ast_recipe, prior_config)。"""
    features = load_viable_features()

    if not feature_ref:
        if features:
            top = features[0]
            fc = top.get("feature_id") or top.get("feature_name")
            return fc, top.get("ast_recipe"), dict(top.get("prior_config") or {})
        return None, None, {}

    for item in features:
        candidates = {
            str(item.get("feature_id") or ""),
            str(item.get("feature_name") or ""),
            str(item.get("fingerprint") or ""),
            str(item.get("display_name") or ""),
        }
        if feature_ref in candidates:
            fc = item.get("feature_id") or item.get("feature_name") or feature_ref
            return str(fc), item.get("ast_recipe"), dict(item.get("prior_config") or {})

    return feature_ref, None, {}


def build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run bt_studio WFO / HPO DAG")
    p.add_argument(
        "--feature",
        default=os.environ.get("BT_STUDIO_FEATURE_COL"),
        help="Feature column name, feature_id (f_<fp12>), or fingerprint from viable_features.json",
    )
    p.add_argument(
        "--dag",
        default=os.environ.get("BT_STUDIO_DAG", "wfo_production"),
        help="DAG name in dags/ (default: wfo_production)",
    )
    p.add_argument("--start-date", type=int, default=None, help="Override start_date (YYYYMMDD)")
    p.add_argument("--end-date", type=int, default=None, help="Override end_date (YYYYMMDD)")
    p.add_argument("--train-window", type=int, default=None, help="Override train_window (months)")
    p.add_argument("--oos-step", type=int, default=None, help="Override oos_step (months)")
    p.add_argument("--num-trials", type=int, default=None, help="Override Ray Tune num_trials")
    p.add_argument("--num-workers", type=int, default=None, help="Override Ray Tune max_concurrent_trials")
    p.add_argument("--last-model-id", type=int, default=None, help="Resume from last_model_id (YYYYMM)")
    p.add_argument("--task-id", default=None, help="Explicit task_id bucket override")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_cli_parser().parse_args(argv)

    # [FIX] Force local SQLite MLflow backend unless MLFLOW_TRACKING_URI is explicitly set
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(f"sqlite:///{os.path.join(os.getcwd(), 'mlflow.db')}")
    print(f"  [MLflow] Tracking URI: {mlflow.get_tracking_uri()}")

    feature_col, ast_recipe, prior_config = _lookup_viable_feature(args.feature)
    if not feature_col:
        raise SystemExit(
            "未指定特征且 viable_features.json 为空: 请通过 --feature <feature_id|col> "
            "或环境变量 BT_STUDIO_FEATURE_COL 显式指定要运行的特征。"
        )

    common_overrides: Dict[str, Any] = {"feature_col": feature_col}
    if prior_config:
        common_overrides["prior_config"] = prior_config
    if args.start_date is not None:
        common_overrides["start_date"] = args.start_date
    if args.end_date is not None:
        common_overrides["end_date"] = args.end_date
    if args.train_window is not None:
        common_overrides["train_window"] = args.train_window
    if args.oos_step is not None:
        common_overrides["oos_step"] = args.oos_step
    if args.num_trials is not None:
        common_overrides["num_trials"] = args.num_trials
    if args.num_workers is not None:
        common_overrides["num_workers"] = args.num_workers
    if args.last_model_id is not None:
        common_overrides["last_model_id"] = args.last_model_id

    user_cfg: Dict[str, Any] = {
        "feature_col": feature_col,
        "common_params": common_overrides,
    }
    if ast_recipe is not None:
        user_cfg["ast_recipe"] = ast_recipe
    if args.task_id:
        user_cfg["task_id"] = args.task_id

    exp_config = build_exp_config(user_cfg)
    res = run_dag(parse_dag(args.dag), exp_config, ast_recipe=ast_recipe)
    print(
        f"✅ [{res.dag_name}] task_id={res.task_id} windows={res.windows_executed} "
        f"models={res.models_produced} last_model_id={res.last_model_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
