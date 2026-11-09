#!/usr/bin/env python3

from __future__ import annotations

import os

import polars as pl

from bt_sdk.core.factor import apply_factor
from bt_sdk.ctx import external_mdapi_context
from bt_protocol._protocol import QueryBody
from bt_protocol.constant import FactorTopic, RpcTopic

from bt_studio.constant import (
    AFTERNOON_FIRST_MINUTE,
    AFTERNOON_LAST_MINUTE,
    MORNING_FIRST_MINUTE,
    MORNING_LAST_MINUTE,
    RAW_PIT_DIR,
    TRADING_BARS_PER_DAY,
)
from bt_studio.utils.io import (
    RawPitCacheStore,
    _collect_stream_sync,
    fetch_hfq_factors_safe,
)
from bt_studio.utils.months import month_bounds, ym_range

_BARS_SCHEMA = {
    "sid": pl.Binary,
    "day": pl.Date,
    "bar_idx": pl.Int32,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "amount": pl.Float64,
}


def prepare_macro(
    start_date: int,
    end_date: int,
    benchmark: bytes = b"1A0001",
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    with external_mdapi_context() as mdapi:
        inst_df = mdapi.get_instrument()
        # In rpc_feed (provider.pyx), delist is int32: 0 (or null) = active, YYYYMMDD = delisted date
        valid_meta = inst_df.filter(
            (pl.col("first_trading") <= end_date)
            & (pl.col("delist").is_null() | (pl.col("delist") == 0) | (pl.col("delist") >= start_date))
        )

        universe_lazy = valid_meta.select([
            pl.col("sid").cast(pl.Binary),
            pl.col("first_trading").cast(pl.Int32),
        ]).lazy()

        universe = valid_meta["sid"].cast(pl.Binary).to_list()
        body = QueryBody(start_date=start_date, end_date=end_date, sid=universe)
        raw = _collect_stream_sync(mdapi.subscribe(body, RpcTopic.Daily))

        lazy_frames = [
            df.with_columns(pl.lit(sid).cast(pl.Binary).alias("sid")).lazy()
            for sid, df in raw.items()
            if df.height > 0
        ]
        return universe_lazy, (pl.concat(lazy_frames) if lazy_frames else pl.LazyFrame())


def prepare_tick(
    start_date: int,
    end_date: int,
    sids: list[bytes | str],
) -> pl.LazyFrame:
    """Point-in-time minute bar cache loader with write-time validation and atomic monthly pack merging."""
    months = ym_range(start_date // 100, end_date // 100)
    batch = max(1, int(os.environ.get("TICK_BATCH_SIDS", "32")))
    store = RawPitCacheStore(RAW_PIT_DIR, _BARS_SCHEMA)
    sids_bytes = [s.encode() if isinstance(s, str) else bytes(s) for s in sids]

    with external_mdapi_context() as mdapi:
        for ym in months:
            covered = store.covered_sids(ym)
            missing = [s for s in sids_bytes if s.decode() not in covered]
            if not missing and os.path.exists(store.pack_path(ym)):
                continue

            new_frames: list[pl.DataFrame] = []
            added_sids: set[str] = set()
            lo, hi = month_bounds(ym)

            for i in range(0, len(missing), batch):
                chunk = missing[i : i + batch]
                body = QueryBody(start_date=lo, end_date=hi, sid=chunk)
                raw = _collect_stream_sync(mdapi.subscribe(body, RpcTopic.Tick))
                factors = fetch_hfq_factors_safe(mdapi, body, FactorTopic.Hfq)
                if factors:
                    raw = apply_factor(raw, factors, FactorTopic.Hfq)
                for sid in chunk:
                    tick_df = raw.get(sid)
                    if tick_df is not None and tick_df.height > 0 and "tick" in tick_df.columns:
                        df = align_skeleton(
                            tick_df.with_columns(pl.lit(sid).alias("sid").cast(pl.Binary)).lazy()
                        ).collect(engine="streaming")
                    else:
                        df = pl.DataFrame(schema=_BARS_SCHEMA)

                    sid_str = sid.decode()
                    if store.is_valid_sid_month(df):
                        new_frames.append(df)
                        added_sids.add(sid_str)
                    else:
                        n_days = df.height // TRADING_BARS_PER_DAY if df.height else 0
                        print(f"[raw_pit] {(sid_str, ym)} validation failed (days={n_days}), skipped")

            if added_sids:
                store.merge_and_save_month(ym, new_frames, added_sids)

    return store.verify_and_scan(months, sids_bytes, start_date, end_date)


def align_skeleton(tick_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Align raw intraday ticks onto the 240-bar END-labeled skeleton (09:31..11:30, 13:01..15:00)."""
    processed_lf = tick_lf.with_columns(
        (pl.col("tick") * 1000)
        .cast(pl.Int64)
        .cast(pl.Datetime("ms"))
        .alias("tick_dt")
    ).with_columns(
        pl.col("tick_dt").dt.date().alias("day")
    )

    # Normalize UTC timestamps to UTC+8 (Asia/Shanghai)
    processed_lf = processed_lf.with_columns(
        pl.when(
            pl.col("tick_dt").dt.hour().min().over(["day"]) < 8
        )
        .then(pl.col("tick_dt").dt.offset_by("8h"))
        .otherwise(pl.col("tick_dt"))
        .alias("tick_dt")
    ).with_columns(
        pl.col("tick_dt").dt.date().alias("day")
    )

    # END-label minute mapping: 09:31->0 .. 11:30->119 | 13:01->120 .. 15:00->239
    processed_lf = (
        processed_lf
        .with_columns(
            (
                pl.col("tick_dt").dt.hour().cast(pl.Int32) * 60
                + pl.col("tick_dt").dt.minute().cast(pl.Int32)
            ).alias("to_minutes"),
        )
        .filter(
            ((pl.col("to_minutes") >= MORNING_FIRST_MINUTE) & (pl.col("to_minutes") <= MORNING_LAST_MINUTE))
            | ((pl.col("to_minutes") >= AFTERNOON_FIRST_MINUTE) & (pl.col("to_minutes") <= AFTERNOON_LAST_MINUTE))
        )
        .with_columns(
            pl.when(pl.col("to_minutes") <= MORNING_LAST_MINUTE)
            .then(pl.col("to_minutes") - MORNING_FIRST_MINUTE)
            .otherwise((pl.col("to_minutes") - AFTERNOON_FIRST_MINUTE) + 120)
            .cast(pl.Int32)
            .alias("bar_idx")
        )
    )

    # Raw 240-bar skeleton construction for all observed (sid, day) pairs
    unique_pairs = processed_lf.select(["sid", "day"]).unique()
    skeleton_lf = unique_pairs.with_columns(
        pl.int_ranges(0, TRADING_BARS_PER_DAY, dtype=pl.Int32).alias("bar_idx")
    ).explode("bar_idx", empty_as_null=True)

    # Causally forward-fill close within each (day, sid) session
    return (
        skeleton_lf.join(processed_lf, on=["day", "sid", "bar_idx"], how="left")
        .sort(["day", "sid", "bar_idx"])
        .with_columns(
            pl.col("close").forward_fill().over(["day", "sid"]).alias("close")
        )
        .with_columns([
            pl.col("open").fill_null(pl.col("close")),
            pl.col("high").fill_null(pl.col("close")),
            pl.col("low").fill_null(pl.col("close")),
            pl.col("amount").fill_null(0.0),
            pl.col("volume").fill_null(0.0),
        ])
        .select([
            "sid", "day", "bar_idx", "open", "high", "low", "close",
            "volume", "amount",
        ])
    )
