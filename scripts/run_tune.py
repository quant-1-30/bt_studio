#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional, Tuple

import mlflow

from bt_studio.agent.flywheel import load_viable_features
from bt_studio.default_config import build_exp_config
from bt_studio.engine import parse_dag, run_dag


def _lookup_viable_feature(feature_ref: Optional[str]) -> Tuple[Optional[str], Any, Dict[str, Any]]:
    """Resolve (feature_col, ast_recipe, prior_config) from viable_features.json."""
    features = load_viable_features()

    if not feature_ref:
        if features:
            top = features[0]
            fc = top.get("feature_id", top.get("feature_name"))
            return fc, top.get("ast_recipe"), dict(top.get("prior_config", {}))
        return None, None, {}

    for item in features:
        candidates = {
            str(item.get("feature_id", "")),
            str(item.get("feature_name", "")),
            str(item.get("fingerprint", "")),
            str(item.get("display_name", "")),
        }
        if feature_ref in candidates:
            fc = item.get("feature_id", item.get("feature_name", feature_ref))
            return str(fc), item.get("ast_recipe"), dict(item.get("prior_config", {}))

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

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(f"sqlite:///{os.path.join(os.getcwd(), 'mlflow.db')}")
    print(f"  [MLflow] Tracking URI: {mlflow.get_tracking_uri()}")

    feature_col, ast_recipe, prior_config = _lookup_viable_feature(args.feature)
    if not feature_col:
        raise SystemExit(
            "viable_features.json null and --feature <feature_id|col> "
            "or env set BT_STUDIO_FEATURE_COL is required"
        )

    common_overrides: Dict[str, Any] = {"feature_col": feature_col}
    if prior_config:
        common_overrides["prior_config"] = prior_config
    for cli_val, key in (
        (args.start_date, "start_date"),
        (args.end_date, "end_date"),
        (args.train_window, "train_window"),
        (args.oos_step, "oos_step"),
        (args.num_trials, "num_trials"),
        (args.num_workers, "num_workers"),
        (args.last_model_id, "last_model_id"),
    ):
        if cli_val is not None:
            common_overrides[key] = cli_val

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
