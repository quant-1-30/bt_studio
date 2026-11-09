#! /usr/bin/env python3
# -*- encondig: utf-8 -*-

import os
import numpy as np

from bt_studio.constant import DEFAULT_EXCLUDE_BARS, TRADING_BARS_PER_DAY


# ===========================================================================
# Math: Entry Cutoff, T+1 Target Decay & DTW Threshold Derivation
# ===========================================================================

def entry_bar_idx(common_config: dict | None = None) -> int:
    """Compute the 0-based intraday entry bar index (default 230 for 240 bars, exclude_bars=10)."""
    cc = common_config or {}
    total_bars = int(cc.get("bars_per_day", TRADING_BARS_PER_DAY))
    exclude_bars = int(cc.get("exclude_bars", DEFAULT_EXCLUDE_BARS))
    return min(total_bars - 1, max(0, total_bars - exclude_bars))


def causal_keep_bars(common_config: dict | None = None, downsample: int = 1) -> int:
    """Compute the number of downsampled intraday bars strictly before the excluded post-entry tail."""
    cc = common_config or {}
    total_bars = int(cc.get("bars_per_day", TRADING_BARS_PER_DAY))
    exclude_bars = max(0, int(cc.get("exclude_bars", 0)))
    ds = max(1, int(downsample))
    return max(1, (total_bars - exclude_bars) // ds)


def resolve_dtw_workers(common_config: dict) -> int:
    """Resolve DTW thread count: explicit max_workers > Ray worker assigned CPU quota (1 in Tune) > full host cores on driver."""
    explicit = common_config.get("max_workers")
    if explicit is not None:
        return max(1, int(explicit))
    try:
        import ray
        if ray.is_initialized():
            rt_ctx = ray.get_runtime_context()
            worker_mode = getattr(getattr(ray, "_private", None), "worker", None)
            expected_mode = getattr(worker_mode, "WORKER_MODE", 1)
            if getattr(getattr(rt_ctx, "worker", None), "mode", 0) == expected_mode:
                return max(1, int(rt_ctx.get_assigned_resources().get("CPU", 1)))
    except Exception:
        pass
    return os.cpu_count() or 1

def calculate_decay_weights(
    rets_window: dict[str, int], 
    half_life_minutes: float = 15.0
) -> dict[str, float]:
    """
    T+1 open Offset Targets exp weight

    Parameters
    ----------
    rets_window : dict[str, int]
        {"exit_5m": 5, "exit_15m": 15, "exit_30m": 30}
    half_life_minutes : float, optional
        default: 15.0

    Returns
    -------
    dict[str, float]
    """
    if not rets_window:
        return {}

    safe_half_life = max(1e-4, float(half_life_minutes))
    decay_const = np.log(2) / safe_half_life

    names = list(rets_window.keys())
    minutes = np.array(list(rets_window.values()), dtype=np.float64)

    raw_weights = np.exp(-decay_const * minutes)

    sum_w = np.sum(raw_weights)
    norm_weights = raw_weights / sum_w

    return {name: float(w) for name, w in zip(names, norm_weights)}


def derive_m_threshold_d(
    motif_minutes: int, downsample: int, threshold_r: float,
) -> tuple[int, float]:
    """threshold_d = sqrt(2m(1-r))
    """
    ds = max(1, int(downsample))
    r = float(np.clip(float(threshold_r), 0.0, 0.9999))
    m = max(1, int(motif_minutes) // ds)
    threshold_d = float(np.sqrt(2 * m * (1.0 - r)))
    return m, threshold_d


def resolve_search_hyperparams(tune_config: dict) -> dict:
    """Clone tune_config and derive m and threshold_d from scalar motif_minutes, downsample, and threshold_r."""
    cfg = tune_config.copy()
    r = float(cfg["threshold_r"])
    m, threshold_d = derive_m_threshold_d(cfg["motif_minutes"], cfg["downsample"], r)
    cfg["threshold_r"] = r
    cfg["m"] = m
    cfg["threshold_d"] = threshold_d
    return cfg


def group_combos_by_downsample(combos: list[dict]) -> dict[int, list[dict]]:
    """Group hyperparameter grid combinations by downsample factor to share curve extraction."""
    grouped: dict[int, list[dict]] = {}
    for c in combos:
        grouped.setdefault(c.get("downsample", 1), []).append(c)
    return grouped
