"""Intraday volatility regime feature."""

import polars as pl


def _demean_expr(col_name: str) -> pl.Expr:
    median = pl.col(col_name).median().over(["day", "bar_idx"])
    return pl.col(col_name) - median


def build_vol(aligned_lf: pl.LazyFrame, common_config: dict) -> pl.LazyFrame:
    """Build intraday volatility regime feature (all causal).

    Pipeline:
      1. Per-bar normalized range: (high - low) / (close + eps)
      2. Cumulative sum within day
      3. Cross-sectional demean -> vol_ratio
    """
    eps = common_config["eps"]

    step1_lf = (
        aligned_lf
        .sort(["day", "sid", "bar_idx"])
        .with_columns([
            ((pl.col("high") - pl.col("low")) / (pl.col("close") + eps)).alias("bar_range"),
        ])
    )

    step2_lf = (
        step1_lf
        .with_columns([
            pl.col("bar_range").cum_sum().over(["day", "sid"]).alias("cum_vol"),
        ])
    )

    final_lf = (
        step2_lf
        .with_columns([
            _demean_expr("cum_vol").alias("vol_ratio"),
        ])
        .rename({"bar_idx": "bar_idx"})
        .select(["day", "sid", "bar_idx", "open", "close", "vol_ratio"])
        .sort(["day", "sid", "bar_idx"])
    )
    return final_lf
