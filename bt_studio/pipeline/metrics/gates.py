#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats


def daily_paired_pval(
    triggers: pl.DataFrame,
    complementary_df: pl.DataFrame,
    alternative: str,
    min_days: int = 3,
) -> float:
    cond_daily = (
        triggers.group_by("day")
        .agg(pl.col("rank_trajectory").mean().alias("cond_mean"))
    )
    uncond_daily = (
        complementary_df.group_by("day")
        .agg(pl.col("rank_trajectory").mean().alias("uncond_mean"))
    )
    daily = cond_daily.join(uncond_daily, on="day", how="inner").drop_nulls()
    diffs = (daily["cond_mean"] - daily["uncond_mean"]).to_numpy()

    # Wilcoxon zero-difference test requires at least 3 non-zero differences to compute a valid p-value.
    valid_diffs = diffs[np.abs(diffs) > 1e-8]
    if len(valid_diffs) < min_days:
        return 1.0

    try:
        _, pval = stats.wilcoxon(valid_diffs, alternative=alternative)
        return float(pval)
    except Exception:
        return 1.0


def daily_trigger_autocorr(eval_df: pl.DataFrame) -> float:
    daily_rate = (
        eval_df.group_by("day")
        .agg(pl.col("is_trigger").mean().alias("rate"))
        .sort("day")
    )
    if daily_rate.height < 3:
        return 0.0

    corr = (
        daily_rate.select(
            pl.corr("rate", pl.col("rate").shift(1))
            .fill_nan(0.0).fill_null(0.0)
        ).to_series()[0]
    )
    return float(corr)


def raw_excess_gate(
    triggers: pl.DataFrame,
    complementary_df: pl.DataFrame,
    common_config: dict,
) -> tuple[bool, str]:
    """Economic Excess Gate (raw_trajectory / per-target raw_* hard gate)."""
    min_excess = common_config.get("min_excess_ret", 0.0)
    if not min_excess:
        return True, ""

    alt = common_config.get("alternative", "greater")

    def _excess(col: str) -> float:
        cond_series = triggers.get_column(col).drop_nulls()
        uncond_series = complementary_df.get_column(col).drop_nulls()
        if len(cond_series) == 0 or len(uncond_series) == 0:
            return float("-inf")
        raw_diff = float(cond_series.median()) - float(uncond_series.median())
        if alt == "less":
            return -raw_diff
        return abs(raw_diff) if alt == "two-sided" else raw_diff

    thresholds = (
        {f"raw_{name}": (name, float(thr)) for name, thr in min_excess.items()}
        if isinstance(min_excess, dict)
        else {"raw_trajectory": ("raw_trajectory", float(min_excess))}
    )
    details = []
    for col, (label, thr) in thresholds.items():
        if col not in triggers.columns:
            continue
        e = _excess(col)
        details.append(f"{label}: {e:.4%} vs {thr:.4%}")
        if e < thr:
            msg = f"raw excess below threshold ({'; '.join(details)})" if isinstance(min_excess, dict) else f"raw excess {e:.4%} < min_excess_ret {thr:.4%}"
            return False, msg
    return True, "; ".join(details) if isinstance(min_excess, dict) else ""
