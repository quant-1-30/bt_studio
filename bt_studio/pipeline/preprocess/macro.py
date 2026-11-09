#! /usr/bin/env python3


import polars as pl


def compute_macro_states(panel_df: pl.DataFrame, common_config: dict) -> pl.DataFrame:
    """Compute daily macro regime states (0, 1, 2) using shifted prior-day median OFI rolling quantiles."""
    rank_window = int(common_config["ranking_window"])
    min_samples = max(1, rank_window // 2)

    daily = (
        panel_df.lazy()
        .select([
            pl.col("day"),
            pl.col("lag_0").list.sum().alias("sid_ofi_sum"),
        ])
        .group_by("day")
        .agg(pl.col("sid_ofi_sum").median().alias("daily_ofi_median"))
        .sort("day")
        .with_columns(pl.col("daily_ofi_median").shift(1).alias("prev_ofi_median"))
        .with_columns([
            pl.col("prev_ofi_median")
            .rolling_quantile(quantile=0.33, window_size=rank_window, min_samples=min_samples)
            .alias("p33"),
            pl.col("prev_ofi_median")
            .rolling_quantile(quantile=0.67, window_size=rank_window, min_samples=min_samples)
            .alias("p67"),
        ])
        .collect(engine="streaming")
    )

    return daily.with_columns(
        pl.when(
            pl.col("prev_ofi_median").is_null()
            | pl.col("p33").is_null()
            | pl.col("p67").is_null()
        )
        .then(1)
        .when(pl.col("prev_ofi_median") <= pl.col("p33")).then(0)
        .when(pl.col("prev_ofi_median") <= pl.col("p67")).then(1)
        .otherwise(2)
        .cast(pl.Int32)
        .alias("macro_state")
    ).select(["day", "macro_state"])
