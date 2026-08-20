#! /usr/bin/env python3

from __future__ import annotations

import re

import polars as pl

# Feature parquet naming convention: hf_{feature_col}_{YYYYMM}.parquet
_YM_SUFFIX_RE = re.compile(r"_(\d{6})\.parquet$")


def list_available_months(dret_path: str) -> list[int]:
    """All distinct YYYYMM month ids in the daily parquet (sorted asc)."""
    daily_lazy = pl.scan_parquet(dret_path)
    schema = daily_lazy.collect_schema().names()
    date_col = "day" if "day" in schema else "date"
    
    if date_col == "day":
        if daily_lazy.collect_schema()["day"] in [pl.Int32, pl.Int64]:
             daily_lazy = daily_lazy.with_columns(pl.col("day").cast(pl.String).str.to_date("%Y%m%d").alias("day"))
        elif daily_lazy.collect_schema()["day"] == pl.String:
             daily_lazy = daily_lazy.with_columns(pl.col("day").str.to_date("%Y%m%d").alias("day"))

    daily_lazy = daily_lazy.select([date_col])
    month_id_expr = (pl.col(date_col).dt.year() * 100
                     + pl.col(date_col).dt.month()).cast(pl.Int32).alias("month_id")
    return (daily_lazy.select(month_id_expr).unique().sort("month_id")
            .collect(engine="streaming").get_column("month_id").to_list())


def select_train_months(dret_path: str, train_window: int = 12):
    """Pick the latest ``train_window`` months.

    Returns ``(train_yms, model_id)`` where ``model_id`` is the last
    available month (None when the cache is empty).
    """
    yms = list_available_months(dret_path)
    train_yms = yms[-train_window:] if len(yms) > train_window else yms
    model_id = yms[-1] if yms else (train_yms[-1] if train_yms else None)
    return train_yms, model_id


def paths_for_months(paths: list[str], ymonths: list[int]) -> list[str]:
    """Filter an already-materialized path list down to the given months.

    Feature parquets follow ``hf_{feature_col}_{YYYYMM}.parquet``; the month
    is anchored at the filename tail so any feature-col naming is matched.
    Used to reuse the train-window paths for the decay-check OOS slice
    instead of re-invoking ``node_extract_feature_monthly``.
    """
    wanted = {int(ym) for ym in ymonths}
    out = []
    for p in paths:
        m = _YM_SUFFIX_RE.search(p)
        if m and int(m.group(1)) in wanted:
            out.append(p)
    return out
