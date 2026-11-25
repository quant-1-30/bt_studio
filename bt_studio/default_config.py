from __future__ import annotations

from typing import Any, Dict

from bt_studio.constant import (
    DEFAULT_EXCLUDE_BARS,
    DEFAULT_WARMUP_MONTHS,
    MIN_WINDOW_PASS_RATIO,
)


# ---------------------------------------------------------------------------
# Production Baseline Configuration (Single Source of Truth)
# ---------------------------------------------------------------------------

PRODUCTION_COMMON_PARAMS: Dict[str, Any] = {
    # Date range & benchmark (full 2010-2025 multi-cycle Walk-Forward horizon)
    "start_date": 20100101,
    "end_date": 20251231,
    "benchmark": "1A0001",
    "warmup_months": DEFAULT_WARMUP_MONTHS,

    # Point-in-time stock universe filtering (min_amount_rank=0.90 selects M -> M+1 causal Top 10% by avg_amount)
    "days_since_ipo": 120,
    "min_active_ratio": 0.90,
    "min_amount_rank": 0.90,

    # Walk-forward windowing & two-stage early-stop
    "train_window": 12,
    "oos_step": 6,
    "min_window_pass_ratio": MIN_WINDOW_PASS_RATIO,
    "max_consec_failures": 3,

    # T+1 exit horizons from open (minutes) & exponential decay half-life
    "T1_rets": {
        "exit_5m": 5,
        "exit_15m": 15,
        "exit_30m": 30,
    },
    "decay_minutes": 15,

    # STUMPY / DTW constraints
    "exclude_bars": DEFAULT_EXCLUDE_BARS,
    "dtw_window_frac": 0.1,

    # Macro quantiles & trigger threshold
    "ranking_window": 5,
    "topk": 5,
    "trigger": 30,
    
    # candidate_multiplier × max_positions
    "candidate_multiplier": 1.5,
    "auto_allowed": False,

    # trade/drift.py warning threshold: |v − median| / |median| > drift_rel
    "drift_rel": 0.5,

    # Statistical & economic gates.
    "alternative": "greater",
    "u_pval": 0.10,
    "min_excess_ret": {
        "exit_5m": 2e-3,
        "exit_15m": 2.5e-3,
        "exit_30m": 3e-3,
    },
    "eps": 1e-8,

    # Ray Tune scheduling & resources
    "num_workers": 12,
    "n_startup_trials": 20,
    "num_trials": 100,
    "seed": 42,
    "storage_path": "/tmp/ray_results",
}


# ---------------------------------------------------------------------------
# SSOT: Search Space Specification
# ---------------------------------------------------------------------------

PRODUCTION_SEARCH_SPEC: Dict[str, Dict[str, Any]] = {
    "downsample": {
        "type": "choice",
        "values": [3, 4, 5],
    },
    "motif_minutes": {
        "type": "choice",
        "values": [30, 45, 60, 90],
    },
    "threshold_r": {
        "type": "uniform",
        "low": 0.70,
        "high": 0.92,
    },
}
