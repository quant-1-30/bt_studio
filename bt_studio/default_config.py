from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Union

import mlflow
import optuna
import requests
from ray.air.integrations.mlflow import MLflowLoggerCallback

from bt_studio.utils.manifest import safe_load_model_ckpt
from bt_studio.utils.months import find_model_path

_CONTROL_KEYS = frozenset({"run_name", "task_id", "ast_recipe", "feature_col"})

_SEARCH_CONTROL_KEYS = frozenset({
    "num_trials",
    "max_concurrent_trials",
    "metric",
    "mode",
    "seed",
    "storage_path",
    "num_workers",
    "n_startup_trials",
})


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
# Production Baseline Configuration
# ---------------------------------------------------------------------------

PRODUCTION_COMMON_PARAMS: Dict[str, Any] = {
    # Date range & benchmark
    "start_date": 20100101,
    "end_date": 20201231,
    "benchmark": "1A0001",
    "warmup_months": 3,

    # Point-in-time stock universe filtering
    "days_since_ipo": 120,
    "min_active_ratio": 0.90,
    "min_amount_rank": 0.10,

    # Walk-forward windowing
    "train_window": 12,
    "oos_step": 6,

    # T+1 exit horizons from open
    "T1_rets": {
        "exit_5m": 5,
        "exit_15m": 15,
        "exit_30m": 30,
    },
    "decay_minutes": 15,

    # STUMPY / DTW constraints
    "exclude_bars": 10,
    "dtw_window_frac": 0.1,

    # Macro quantiles & trigger threshold
    "ranking_window": 5,
    "topk": 5,
    "trigger": 30,

    # Statistical & economic gates
    "alternative": "greater",
    "u_pval": 0.10,
    "min_excess_ret": 0.02,
    "eps": 1e-8,

    # Ray Tune scheduling & resources
    "num_workers": 12,
    "n_startup_trials": 40,
    "num_trials": 500,
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
        "low": 0.65,
        "high": 0.85,
    },
}


_SPEC_TYPES = ("choice", "uniform", "int_range")
_SPEC_REQUIRED_FIELDS = {"choice": ("values",), "uniform": ("low", "high"),
                         "int_range": ("low", "high")}


def normalize_search_spec(raw_spec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:

    normalized: Dict[str, Dict[str, Any]] = {}

    for param_name, spec in raw_spec.items():
        if param_name in _SEARCH_CONTROL_KEYS:
            raise ValueError(
                f"normalize_search_spec: {param_name!r} belongs in common_params"
            )
        if not (isinstance(spec, dict) and spec.get("type") in _SPEC_TYPES):
            raise TypeError(
                f"normalize_search_spec: {param_name!r} unavailable: {spec!r}。"
                f"{{'type': 'choice'|'uniform'|'int_range', ...}}"
            )
        missing = [f for f in _SPEC_REQUIRED_FIELDS[spec["type"]] if f not in spec]
        if missing:
            raise TypeError(
                f"normalize_search_spec: {param_name!r} ({spec['type']}) {missing}。"
            )
        normalized[param_name] = copy.deepcopy(spec)

    return normalized


# ---------------------------------------------------------------------------
# Configuration Builder
# ---------------------------------------------------------------------------

def build_exp_config(user_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

    if user_config is None:
        user_config = {}
    elif not isinstance(user_config, dict):
        raise TypeError(f"build_exp_config: user_config dict / None, but got {type(user_config).__name__}")

    allowed_top_keys = {"common_params", "search_spec"}.union(_CONTROL_KEYS)
    illegal = [k for k in user_config if k not in allowed_top_keys]
    if illegal:
        raise ValueError(
            f"build_exp_config: {illegal} \n"
            f"allowed key: 'common_params', 'search_spec' and {sorted(_CONTROL_KEYS)}。\n"
        )

    # 1. extract and merge common args
    common_override = user_config.get("common_params") or {}
    if not isinstance(common_override, dict):
        raise TypeError("common_params not dict")
    final_common = _deep_merge(PRODUCTION_COMMON_PARAMS, common_override)

    # 2. normalize search_spec
    search_override = user_config.get("search_spec") or {}
    if not isinstance(search_override, dict):
        raise TypeError("search_spec 必须为 dict 类型")
    final_search_spec = copy.deepcopy(PRODUCTION_SEARCH_SPEC)
    if search_override:
        final_search_spec.update(normalize_search_spec(search_override))

    # 3. compose
    exp_config: Dict[str, Any] = {
        "common_params": final_common,
        "search_spec": final_search_spec,
    }

    # 4. mount process control params
    for key in _CONTROL_KEYS:
        if key in user_config and user_config[key] is not None:
            exp_config[key] = copy.deepcopy(user_config[key])

    return exp_config


def resolve_feature_col(
    common_config: Optional[Dict[str, Any]],
    ast_recipe: Any = None,
    explicit: Any = None,
) -> str:

    fc = explicit or (common_config or {}).get("feature_col")
    if fc is None and isinstance(ast_recipe, list) and ast_recipe \
            and isinstance(ast_recipe[-1], dict):
        fc = ast_recipe[-1].get("name")
    if not fc:
        raise ValueError(
            "feature_col or common_params.feature_col, "
            "or recipe name (ast_recipe[-1]['name'])。"
        )
    return fc


def search_spec_to_collapse_bounds(search_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    normalized spec: {"downsample": {"type":"choice","values":[3,4,5]}}
    legacy bounds:   {"downsample": [3,4,5], "threshold_r": [0.65,0.85]}
    """
    bounds: Dict[str, Any] = {}
    if not isinstance(search_config, dict):
        return bounds

    for name, value in search_config.items():
        if name in _SEARCH_CONTROL_KEYS:
            continue

        if isinstance(value, dict) and "type" in value:
            p_type = value.get("type")
            if p_type == "choice":
                values = list(value.get("values", []))
                if values:
                    bounds[name] = values
            elif p_type in ("uniform", "int_range"):
                bounds[name] = [value.get("low"), value.get("high")]
            continue

        if not isinstance(value, (list, tuple)) or len(value) < 2:
            continue
        raw_list = list(value)
        if any(isinstance(x, str) for x in raw_list):
            bounds[name] = list(dict.fromkeys(raw_list))
            continue
        try:
            sorted_vals = sorted(raw_list)
            all_int = all(isinstance(x, (int, bool)) for x in sorted_vals)
            bounds[name] = sorted_vals if all_int else [sorted_vals[0], sorted_vals[-1]]
        except Exception:
            continue

    return bounds


def search_spec_of(exp_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return exp_config.get("search_spec") or PRODUCTION_SEARCH_SPEC


def common_params_of(exp_config: Dict[str, Any]) -> Dict[str, Any]:
    return exp_config.get("common_params") or exp_config.get("common_config") or {}


def build_ray_tune_search_space(search_spec: Dict[str, Dict[str, Any]]) -> dict:
    """normalized search_spec tranform to Ray Tune (lazy ray import)"""
    from ray import tune

    space = {}
    for param_name, meta in search_spec.items():
        p_type = meta["type"]
        if p_type == "choice":
            space[param_name] = tune.choice(meta["values"])
        elif p_type == "uniform":
            space[param_name] = tune.uniform(meta["low"], meta["high"])
        elif p_type == "int_range":
            space[param_name] = tune.randint(meta["low"], meta["high"])
        else:
            raise ValueError(f"Unsupported tune param type: {p_type}")
    return space


def clamp_seed_point_to_spec(
    raw_seed: Dict[str, Any],
    search_spec: Dict[str, Dict[str, Any]],
    search_space: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Stage2 prior_config clip to align illegal values to valid search space as Optuna's warm start point."""
    target_keys = set(search_space.keys()) if search_space is not None else set(search_spec.keys())
    seed_pt: Dict[str, Any] = {}
    for k, meta in search_spec.items():
        if k not in raw_seed or k not in target_keys:
            continue
        val = raw_seed[k]
        p_type = meta.get("type")
        if p_type == "choice" and val in meta.get("values", []):
            seed_pt[k] = val
        elif p_type == "uniform":
            seed_pt[k] = max(float(meta["low"]), min(float(meta["high"]), float(val)))
        elif p_type == "int_range":
            seed_pt[k] = max(int(meta["low"]), min(int(meta["high"]), int(val)))
    return seed_pt if len(seed_pt) == len(target_keys) and seed_pt else None


def resolve_warm_start_points(
    prev_model_id: Optional[Union[int, str]],
    common_config: dict,
    search_spec: Dict[str, Dict[str, Any]],
    search_space: Dict[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    """Optuna HPO  warm seed or fallback to Stage2 prior_config)。"""
    if prev_model_id:
        prior_model_path = find_model_path(prev_model_id, common_config)
        if prior_model_path is not None:
            try:
                prior_cfg = safe_load_model_ckpt(prior_model_path)["config"]
                return [{k: prior_cfg[k] for k in search_space if k in prior_cfg}]
            except Exception:
                return None
    elif isinstance(common_config.get("prior_config"), dict):
        seed_pt = clamp_seed_point_to_spec(common_config["prior_config"], search_spec, search_space)
        if seed_pt:
            return [seed_pt]
    return None


def create_mlflow_callback(common_config: dict) -> Optional[MLflowLoggerCallback]:
    try:
        uri = mlflow.get_tracking_uri()
        if uri and uri.startswith("http"):
            resp = requests.head(uri, timeout=1.5)
            if resp.status_code < 400:
                return MLflowLoggerCallback(
                    tracking_uri=uri,
                    experiment_name=common_config.get("_task_id") or "FSM_Production_Models",
                    save_artifact=False,
                )
            print(f"  [MLflow] Service unavailable (HTTP {resp.status_code}) -> Skip callback")
    except Exception:
        pass
    return None


def build_optuna_distributions(
    valid_df: Any,
    search_spec: Dict[str, Dict[str, Any]],
) -> tuple[Dict[str, Any], List[str]]:
    """search_spec and trial DataFrame ---> Optuna for fANOVA"""

    distributions: Dict[str, Any] = {}
    active_params: List[str] = []

    for pname, meta in search_spec.items():
        col = f"config/{pname}"
        if col not in valid_df.columns or valid_df[col].n_unique() <= 1:
            continue

        p_type = meta["type"]
        if p_type == "choice":
            distributions[pname] = optuna.distributions.CategoricalDistribution(choices=meta["values"])
        elif p_type == "uniform":
            distributions[pname] = optuna.distributions.FloatDistribution(
                low=float(meta["low"]), high=float(meta["high"])
            )
        elif p_type == "int_range":
            distributions[pname] = optuna.distributions.IntDistribution(
                low=int(meta["low"]), high=int(meta["high"]), step=meta.get("step", 1)
            )
        active_params.append(pname)

    return distributions, active_params
