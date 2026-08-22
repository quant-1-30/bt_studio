from __future__ import annotations

import copy
from typing import Any, Dict, Optional

_CONTROL_KEYS = frozenset({"run_name", "precomputed_features", "ast_recipe", "feature_col"})


def _deep_merge(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not override:
        return copy.deepcopy(base)

    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# ---------------------------------------------------------------------------
# Production baseline
# ---------------------------------------------------------------------------

PRODUCTION_COMMON_PARAMS: Dict[str, Any] = {
    # interval and benchmark
    "start_date": 20100101,
    "end_date": 20201231,
    "benchmark": "1A0001",
    "warmup_months": 3, 

    # filter universe
    "days_since_ipo": 120,        

    # Walk-forward 
    "train_window": 12,
    "oss_step": 6,

    # market state
    "regime_filter": {"ma_window": 20},

    # T+1 Ret
    "T1_rets": {
        "open_5m": 5,
        "open_15m": 15,
        "open_30m": 30,
    },
    "decay_minutes": 15,          

    "eps": 1e-8,
    "min_factor_weight": 0.05,

    # Stumpy / DTW
    "exclude_bars": 10,           # eg.14:50
    "dtw_window_frac": 0.1,

    # macro and trigger
    "ranking_window": 5,
    "ranking_ratio": 0.25,
    "topk": 5,
    "trigger": 30,

    # stats
    "alternative": "greater",
    "u_pval": 0.10,

    # concurrency
    "num_workers": 12,
    "n_startup_trials": 40,
    "num_trials": 500,            
    "seed": 42,
    "storage_path": "/tmp/ray_results",
}

# key -> [bounds / discrete choices]
PRODUCTION_SEARCH_BOUNDS: Dict[str, Any] = {
    "downsample": [3, 4, 5],              
    "motif_minutes": [30, 45, 60, 90],    
    "threshold_r": [0.65, 0.85],          
}


def build_exp_config(user_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
        exp_config

    Parameters
    ----------
    user_config : Optional[Dict[str, Any]]
        
        - common_params
        - search_bounds
        - _CONTROL_KEYS (run_name, ast_recipe)

    Returns
    -------
    Dict[str, Any]
    """
    if user_config is None:
        user_config = {}
    elif not isinstance(user_config, dict):
        raise TypeError(f"build_exp_config: user_config 必须为 dict 或 None, 收到 {type(user_config).__name__}")

    illegal = [
        k for k in user_config
        if k not in ("common_params", "search_bounds") and k not in _CONTROL_KEYS
    ]
    if illegal:
        raise ValueError(
            f"build_exp_config: 检测到未知的顶层配置键 {illegal}。\n"
            f"允许的键: 'common_params', 'search_bounds', 以及控制字段 {sorted(_CONTROL_KEYS)}。\n"
            "通用运行参数请放入 common_params, 超参搜索范围请放入 search_bounds。"
        )

    common = user_config.get("common_params") or {}
    search = user_config.get("search_bounds") or {}

    if not isinstance(common, dict) or not isinstance(search, dict):
        raise TypeError("common_params 与 search_bounds 必须为 dict 类型")

    exp_config: Dict[str, Any] = {
        "common_params": _deep_merge(PRODUCTION_COMMON_PARAMS, common),
        "search_bounds": _deep_merge(PRODUCTION_SEARCH_BOUNDS, search),
    }

    for key in _CONTROL_KEYS:
        if key in user_config and user_config[key] is not None:
            exp_config[key] = copy.deepcopy(user_config[key])

    return exp_config
