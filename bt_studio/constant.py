#!/usr/bin/env python3

import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULT_ROOT = os.environ.get(
    "BT_STUDIO_RESULT_DIR", os.path.join(_PROJECT_ROOT, "result")
)


ARTIFACT_ROOT = os.path.join(RESULT_ROOT, "artifacts")
CACHE_ROOT = os.path.join(RESULT_ROOT, "cache")

# --- artifacts --------------------------------------------------------------- #
FEATURE_DIR = os.path.join(ARTIFACT_ROOT, "features")
REVIEW_DIR = os.path.join(ARTIFACT_ROOT, "review")

LLM_RUN_DIR = os.path.join(ARTIFACT_ROOT, "runs")
LLM_USAGE_DIR = os.path.join(ARTIFACT_ROOT, "usage")
LLM_PENDING_DIR = os.path.join(ARTIFACT_ROOT, "pending")
LLM_PROFILE_DIR = os.path.join(ARTIFACT_ROOT, "profile") 

TUNE_BASE_DIR = os.path.join(ARTIFACT_ROOT, "tune")
TUNE_MODEL_DIR = os.path.join(TUNE_BASE_DIR, "models")          
TUNE_SCORE_DIR = os.path.join(TUNE_BASE_DIR, "scores")
TUNE_COLLAPSE_DIR = os.path.join(TUNE_BASE_DIR, "collapse")


# --- cache --------------------------------------------------------------- #
BT_STUDIO_VERSION = "0.2.1"
RAW_PIT_VERSION = "v2"
EVAL_LOGIC_VERSION = "v2"
RAW_PIT_DIR = os.path.join(CACHE_ROOT, "bars", RAW_PIT_VERSION)  
EVAL_CACHE_ROOT = os.path.join(CACHE_ROOT, "evals")
MONITOR_DB = os.path.join(CACHE_ROOT, "monitor.sqlite")


# --------------------------------------------------------------------------- #
# LLM RL
# --------------------------------------------------------------------------- #
RL_TOPK = 3

# --------------------------------------------------------------------------- #
# Fixed behaviour constants (owned here — do not redefine elsewhere)
# --------------------------------------------------------------------------- #
MAX_AST_DEPTH = 4                 # AST depth cap (overfitting / cyclic guard)
MAX_RECIPE_DEPTH = 6              # Recipe List[AST]
MIN_WINDOW_PASS_RATIO = 0.6       # walk-forward verdict threshold (Stage 2)
TRADING_BARS_PER_DAY = 240        # standard intraday 1-min bars per trading day
TRADING_DAYS_PER_MONTH = 20       # approximate A-share trading days per calendar month
DEFAULT_WARMUP_MONTHS = 3         # default rolling feature / universe warmup months
DEFAULT_EXCLUDE_BARS = 10         # default pre-close bars masked after signal entry

MORNING_FIRST_MINUTE = 9 * 60 + 31     # 09:31 (END-labeled bar 0)
MORNING_LAST_MINUTE = 11 * 60 + 30     # 11:30 (END-labeled bar 119)
AFTERNOON_FIRST_MINUTE = 13 * 60 + 1   # 13:01 (END-labeled bar 120)
AFTERNOON_LAST_MINUTE = 15 * 60        # 15:00 (END-labeled bar 239)

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

DEFAULT_GRID = {
    "downsample": [3, 4, 5],
    "motif_minutes": [30, 45, 60],
    "threshold_r": [0.55, 0.65, 0.75],
}

PREFILTER_THRESHOLDS = {
    "min_std": 1e-6,
    "max_null_ratio": 0.20,
}

EVAL_SEMANTIC_KEYS = (
    "T1_rets", "decay_minutes", "eps", "exclude_bars",
    "dtw_window_frac", "ranking_window", "topk", "trigger",
    "alternative", "min_excess_ret", "seed", "min_curve_valid_ratio",
    "positive_motif_quantile", "max_t1_missing_ratio",
)

TASK_DIR_KEYS = (
    "_task_id", "feature_dir", "tune_base_dir",
    "tune_model_dir", "tune_score_dir", "tune_collapse_dir",
)
