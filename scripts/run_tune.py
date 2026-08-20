#!/usr/bin/env python3
"""Production WFO entry — wfo_production DAG via the engine."""

import os
import sys

import mlflow

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bt_studio.engine import parse_dag, run_dag
from bt_studio.default_config import build_exp_config


if __name__ == "__main__":

    # [FIX] Force local SQLite MLflow backend (avoid Docker HTTP 403 + retry hang)
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(os.getcwd(), 'mlflow.db')}")
    print(f"  [MLflow] Tracking URI: {mlflow.get_tracking_uri()}")

    # Baseline lives in bt_studio/default_config.py (single source of truth).
    # Override per-run knobs here, e.g.:
    #   exp_config = build_exp_config({"common_params": {"end_date": 20221231}})
    exp_config = build_exp_config({})

    run_dag(parse_dag("wfo_production"), exp_config)