#!/usr/bin/env python3

import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULT_ROOT = os.environ.get(
    "BT_STUDIO_RESULT_DIR", os.path.join(_PROJECT_ROOT, "result")
)

# --------------------------------------------------------------------------- #
# SHARED feature cache (llm <-> tune reuse)
# --------------------------------------------------------------------------- #
FEATURE_DIR = os.path.join(RESULT_ROOT, "features")

# --------------------------------------------------------------------------- #
# WFO pipeline (tune.py) directories
# --------------------------------------------------------------------------- #
TUNE_BASE_DIR = os.path.join(RESULT_ROOT, "tune")
TUNE_MODEL_DIR = os.path.join(TUNE_BASE_DIR, "models")
TUNE_SCORE_DIR = os.path.join(TUNE_BASE_DIR, "scores")
TUNE_COLLAPSE_DIR = os.path.join(TUNE_BASE_DIR, "collapse")  # HPO 参数空间塌陷自动校验报告

# --------------------------------------------------------------------------- #
# LLM agent directories
# --------------------------------------------------------------------------- #
LLM_BASE_DIR = os.path.join(RESULT_ROOT, "llm")
LLM_RUN_DIR = os.path.join(LLM_BASE_DIR, "runs")         # per-step iteration JSON
LLM_PENDING_DIR = os.path.join(LLM_BASE_DIR, "pending")   # agent → task_manager 契约目录
LLM_PROFILE_DIR = os.path.join(LLM_BASE_DIR, "profile")  # cProfile artifacts

# --------------------------------------------------------------------------- #
# LLM RL 
# --------------------------------------------------------------------------- #
RL_TOPK = 3

# --------------------------------------------------------------------------- #
# Fixed behaviour constants (owned here — do not redefine elsewhere)
# --------------------------------------------------------------------------- #
MAX_AST_DEPTH = 4                 # AST depth cap (overfitting / cyclic guard)
MIN_WINDOW_PASS_RATIO = 0.6       # walk-forward verdict threshold (Stage 2)

# --------------------------------------------------------------------------- #
# Pipeline Dag
# --------------------------------------------------------------------------- #

DAGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..","dags")

BASE_MARKET_COLS = [
    "day",
    "sid",
    "bar_idx",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]
