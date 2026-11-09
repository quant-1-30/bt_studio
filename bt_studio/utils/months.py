#! /usr/bin/env python3

from __future__ import annotations

import calendar
import datetime as _dt
import math
import os
import re
from typing import List, Optional

import polars as pl

from bt_studio.constant import TRADING_DAYS_PER_MONTH, TUNE_MODEL_DIR
from bt_studio.utils.io import read_json_safe


# Feature parquet naming convention: hf_{feature_col}_{YYYYMM}.parquet
_YM_SUFFIX_RE = re.compile(r"_(\d{6})\.parquet$")


def month_id_expr(col: str = "day") -> pl.Expr:
    """Date/Datetime YYYYMM"""
    return (pl.col(col).dt.year() * 100 + pl.col(col).dt.month()).cast(pl.Int32)


def next_month_id_expr(col: str = "date") -> pl.Expr:
    """Date/Datetime Next YYYYMM (M -> M+1 PIT )"""
    return (
        (pl.col(col).dt.year() + (pl.col(col).dt.month() == 12).cast(pl.Int32)) * 100
        + (pl.col(col).dt.month() % 12 + 1)
    ).cast(pl.Int32)


def align_date_col(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Normalize LazyFrame day -> pl.Date and sid -> stripped pl.String."""
    day_dt = lf.collect_schema()["day"].base_type()
    day_expr = (
        pl.col("day").cast(pl.Date)
        if day_dt in (pl.Date, pl.Datetime)
        else pl.col("day").cast(pl.String).str.to_date("%Y%m%d")
    )
    return lf.with_columns([
        pl.col("sid").cast(pl.String).str.strip_chars(" \x00\t\n"),
        day_expr.alias("day"),
    ])


def parse_ym_from_path(path: str) -> Optional[int]:
    m = _YM_SUFFIX_RE.search(str(path))
    return int(m.group(1)) if m else None


def day_month_id_expr(lf: "pl.LazyFrame", col: str = "day") -> pl.Expr:
    dt_ = lf.collect_schema()[col]
    if dt_.base_type() in (pl.Date, pl.Datetime):
        return month_id_expr(col)
    return pl.col(col).cast(pl.String).str.slice(0, 6).cast(pl.Int32)


def month_bounds(ym: int) -> tuple[int, int]:
    y, m = ym // 100, ym % 100
    if not 1 <= m <= 12:
        raise ValueError(f"illegal month: {ym}")
    last_day = calendar.monthrange(y, m)[1]
    return ym * 100 + 1, ym * 100 + last_day


def ym_range(start_ym: int, end_ym: int) -> list[int]:
    y, m = start_ym // 100, start_ym % 100
    end_y, end_m = end_ym // 100, end_ym % 100
    if not (1 <= m <= 12 and 1 <= end_m <= 12) or (y, m) > (end_y, end_m):
        raise ValueError(f"illegal range: {start_ym}->{end_ym}")
    out: list[int] = []
    while (y, m) <= (end_y, end_m):
        out.append(y * 100 + m)
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def calc_warmup_start_date(start_date: int, warmup_months: int = 1) -> int:
    year = start_date // 10000
    month = (start_date % 10000) // 100

    total_months = year * 12 + (month - 1) - warmup_months
    warm_year = total_months // 12
    warm_month = total_months % 12 + 1

    return warm_year * 10000 + warm_month * 100 + 1


def retrieve_available_months(dret_path: str) -> list[int]:
    daily_lazy = pl.scan_parquet(dret_path)
    date_col = "day" if "day" in daily_lazy.collect_schema().names() else "date"
    return (
        daily_lazy.select(day_month_id_expr(daily_lazy, date_col).alias("month_id"))
        .unique()
        .sort("month_id")
        .collect(engine="streaming")
        .get_column("month_id")
        .to_list()
    )


def resolve_oos_paths_with_warmup(
    all_paths: list[str],
    oos_yms: list[int],
    ranking_window: int,
) -> list[str]:

    if not oos_yms or not all_paths:
        return []

    path_map = {}
    for p in all_paths:
        ym = parse_ym_from_path(p)
        if ym is not None:
            path_map[ym] = p

    sorted_yms = sorted(path_map.keys())

    oos_paths = [path_map[ym] for ym in oos_yms if ym in path_map]
    if not oos_paths:
        return []

    first_oos_ym = min(oos_yms)
    # Convert trading-day rolling ranking_window into calendar warmup months
    n_warm = max(1, math.ceil(int(ranking_window) / TRADING_DAYS_PER_MONTH))
    warm_yms = [ym for ym in sorted_yms if ym < first_oos_ym][-n_warm:]
    warmup_paths = [path_map[ym] for ym in warm_yms]

    return warmup_paths + oos_paths


def find_model_path(model_id: int, common_config: dict) -> Optional[str]:
    for d in (common_config.get("tune_model_dir"), TUNE_MODEL_DIR):
        p = os.path.join(str(d), f"model_{model_id}.pkl") if d else None
        if p and os.path.exists(p):
            return p
    return None


def partition_cached_and_missing_months(
    ymonths: List[int],
    feature_col: str,
    feature_dir: str,
    agent_dir: Optional[str] = None,
) -> tuple[List[str], List[int]]:
    """Agent Fingerprints and save parquet by month"""
    cached_paths: List[str] = []
    missing_ymonths: List[int] = []

    for ym in ymonths:
        if agent_dir:
            ap = os.path.join(agent_dir, f"hf_{feature_col}_{ym}.parquet")
            if os.path.exists(ap):
                cached_paths.append(ap)
                continue
        p = f"{feature_dir}/hf_{feature_col}_{ym}.parquet"
        if os.path.exists(p):
            cached_paths.append(p)
            continue
        missing_ymonths.append(ym)

    return cached_paths, missing_ymonths


def _parse_ymd_date(ymd: int) -> _dt.date:
    return _dt.datetime.strptime(str(int(ymd)), "%Y%m%d").date()


def _extract_cached_max_date(dret_path: str) -> Optional[_dt.date]:
    try:
        lf = pl.scan_parquet(dret_path)
        schema = lf.collect_schema()
        col = "day" if "day" in schema else ("date" if "date" in schema else None)
        if col is None:
            return None
        val = lf.select(pl.col(col).max()).collect().item()
        if val is None:
            return None
        if isinstance(val, _dt.datetime):
            return val.date()
        if isinstance(val, _dt.date):
            return val
        s = str(val).replace("-", "")[:8]
        return _dt.datetime.strptime(s, "%Y%m%d").date()
    except Exception:
        return None


def is_macro_cache_valid(
    dret_path: str,
    fetch_start: int,
    end_date: int,
    common_config: dict,
) -> tuple[bool, List[int]]:
    if not os.path.exists(dret_path):
        return False, ym_range(fetch_start // 100, end_date // 100)

    cached_months = retrieve_available_months(dret_path)
    required_months = ym_range(fetch_start // 100, end_date // 100)
    missing_months = sorted(set(required_months) - set(cached_months))
    if not cached_months or missing_months:
        return False, missing_months

    meta = read_json_safe(dret_path + ".meta.json")
    if isinstance(meta, dict):
        if (
            int(meta.get("fetch_start", 0)) != int(fetch_start)
            or int(meta.get("end_date", 0)) != int(end_date)
            or str(meta.get("benchmark", "")) != str(common_config.get("benchmark", ""))
            or int(meta.get("days_since_ipo", -1)) != int(common_config.get("days_since_ipo", 120))
        ):
            return False, []
        return True, []

    _, month_last_ymd = month_bounds(int(end_date) // 100)
    if int(end_date) < month_last_ymd:
        max_dt = _extract_cached_max_date(dret_path)
        if max_dt is None or max_dt < _parse_ymd_date(end_date):
            return False, []

    return True, []
