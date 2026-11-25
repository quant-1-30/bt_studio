#! /usr/bin/env python3
# -*- encondig: utf-8 -*-

import os
import numpy as np
import polars as pl

from bt_studio.constant import (
    AFTERNOON_FIRST_MINUTE,
    DEFAULT_EXCLUDE_BARS,
    LIMIT_PCT_BY_PREFIX,
    LIMIT_PCT_MAIN,
    MORNING_FIRST_MINUTE,
    TRADING_BARS_PER_DAY,
)


# ===========================================================================
# Math: Entry Cutoff, T+1 Target Decay & DTW Threshold Derivation
# ===========================================================================

def entry_bar_idx(common_config: dict | None = None) -> int:
    """Compute the 0-based intraday entry bar index (default 230 for 240 bars, exclude_bars=10)."""
    cc = common_config or {}
    total_bars = int(cc.get("bars_per_day", TRADING_BARS_PER_DAY))
    exclude_bars = int(cc.get("exclude_bars", DEFAULT_EXCLUDE_BARS))
    return min(total_bars - 1, max(0, total_bars - exclude_bars))


def entry_bar_clock(common_config: dict | None = None) -> tuple[int, int]:
    """Clock time (hour, minute) of the entry bar's END-labeled close.

    Inverse of the 240-bar END mapping in ``constant.py`` (09:31->0 ..
    11:30->119 | 13:01->120 .. 15:00->239). For the default ``exclude_bars=10``
    the entry bar 230 closes at 14:51 — the backtest TRADE timer must fire on
    that bar, not two minutes earlier.
    """
    idx = entry_bar_idx(common_config)
    if idx < 120:
        minute = MORNING_FIRST_MINUTE + idx
    else:
        minute = AFTERNOON_FIRST_MINUTE + (idx - 120)
    return minute // 60, minute % 60


def limit_pct(sid: str) -> float:
    """Entry-side price-limit band for a ticker (shared by tune OOS filter & trade build_topk).

    Reads the shared ``LIMIT_PCT_BY_PREFIX`` table in ``constant.py``; the
    board→band mapping has no re-implementation elsewhere (tune.py consumes this
    scalar via ``map_elements``).
    """
    for prefixes, pct in LIMIT_PCT_BY_PREFIX:
        if sid.startswith(prefixes):
            return pct
    return LIMIT_PCT_MAIN


def causal_keep_bars(common_config: dict | None = None, downsample: int = 1) -> int:
    """Compute the number of downsampled intraday bars strictly before the excluded post-entry tail."""
    cc = common_config or {}
    total_bars = int(cc.get("bars_per_day", TRADING_BARS_PER_DAY))
    exclude_bars = max(0, int(cc.get("exclude_bars", 0)))
    ds = max(1, int(downsample))
    return max(1, (total_bars - exclude_bars) // ds)


def entry_close_expr(entry_idx: int) -> pl.Expr:
    """Entry-price expression: the last non-null ``close`` at ``bar_idx <= entry_idx``.

    缺 entry bar 时向前取最近 bar 收盘 (forward fill); 不用当日收盘兜底 (14:50 缺 bar
    取 15:00 收盘会引入前视)。训练面板 (``build_static_panel``) 与 OOS 推理
    (``node_oos_inference_monthly``) 共用, 保证 train/serve 一致。
    """
    cond = (pl.col("bar_idx") <= entry_idx) & pl.col("close").is_not_null()
    return (
        pl.struct(["bar_idx", "close"])
        .filter(cond)
        .sort()
        .last()
        .struct.field("close")
    )


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
