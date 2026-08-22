#! /usr/bin/env python3

from __future__ import annotations

import re
import polars as pl


# Feature parquet naming convention: hf_{feature_col}_{YYYYMM}.parquet
_YM_SUFFIX_RE = re.compile(r"_(\d{6})\.parquet$")


def calc_warmup_start_date(start_date: int, warmup_months: int = 1) -> int:
    year = start_date // 10000
    month = (start_date % 10000) // 100
    
    total_months = year * 12 + (month - 1) - warmup_months
    warm_year = total_months // 12
    warm_month = total_months % 12 + 1
    
    return warm_year * 10000 + warm_month * 100 + 1


# ==============================================================================
# Date & Month ID Parsing
# ==============================================================================

def _expr_month_id(col_name: str = "day") -> pl.Expr:
    """Date, Datetime, YYYYMMDD Int, YYYY-MM-DD Utf8 ---> Int32 YYYYMM"""
    col = pl.col(col_name)
    return (
        pl.coalesce([
            col.dt.year() * 100 + col.dt.month(),
            col.cast(pl.Int64) // 100,
            col.cast(pl.Utf8).str.replace_all("-", "").str.slice(0, 6).cast(pl.Int64),
        ])
        .cast(pl.Int32)
        .alias("_month_id")
    )


def list_available_months(dret_path: str) -> list[int]:
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
    
    yms = list_available_months(dret_path)
    train_yms = yms[-train_window:] if len(yms) > train_window else yms
    model_id = yms[-1] if yms else (train_yms[-1] if train_yms else None)
    return train_yms, model_id


def paths_for_months(paths: list[str], ymonths: list[int]) -> list[str]:

    wanted = {int(ym) for ym in ymonths}
    out = []
    for p in paths:
        m = _YM_SUFFIX_RE.search(p)
        if m and int(m.group(1)) in wanted:
            out.append(p)
    return out
