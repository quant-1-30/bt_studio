#! /usr/bin/env python3

import polars as pl

from bt_studio.utils.months import month_id_expr, next_month_id_expr


def universe_sample(
    universe_lf: pl.LazyFrame,
    daily_lf: pl.LazyFrame,
    common_config: dict,
) -> pl.LazyFrame:
    """Point-in-Time (PIT) align: based on T month -> select T+1 month."""
    start_date = common_config.get("start_date")
    min_ipo_days = int(common_config.get("days_since_ipo", 120))
    min_active_ratio = float(common_config.get("min_active_ratio", 0.90))
    min_amount_rank = float(common_config.get("min_amount_rank", 0.10))
    
    meta_lf = universe_lf.select(["sid", "first_trading"])

    # 1. normalize date
    uni_lf = (
        daily_lf.join(meta_lf, on="sid", how="left")
        .with_columns([
            pl.col("day").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d").alias("date"),
            pl.col("first_trading").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d").alias("ipo_date"),
        ])
        .with_columns([
            month_id_expr("date").alias("month_id"),
            next_month_id_expr("date").alias("trade_month_id"),
        ])
    )

    # 2. actual month trading days 
    market_month_days = (
        uni_lf.select(["month_id", "date"])
        .unique()
        .group_by("month_id")
        .agg(pl.len().alias("market_trade_days"))
    )

    # 3. filter with active days / IPO ordinary days / amount ranking (based on T month)
    sample_lf = (
        uni_lf.group_by(["sid", "month_id"])
        .agg([
            pl.col("volume").filter((pl.col("volume").is_not_null()) & (pl.col("volume") > 0) & (pl.col("high") > pl.col("low"))).count().alias("active_days"),
            (pl.col("date").max() - pl.col("ipo_date").max()).dt.total_days().alias("days_since_ipo"),
            pl.col("amount").mean().alias("avg_amount"),
            pl.col("trade_month_id").first().alias("trade_month_id")
        ])
        .join(market_month_days, on="month_id", how="left")
        .with_columns(
            (pl.col("active_days") / pl.col("market_trade_days")).alias("active_ratio")
        )
        .filter(
            (pl.col("days_since_ipo") >= min_ipo_days)
            & (pl.col("active_ratio") >= min_active_ratio)
        )
        .with_columns(
            (pl.col("avg_amount").rank(method="average").over("month_id") / pl.len().over("month_id")).alias("amount_rank")
        )
        .filter(pl.col("amount_rank") >= min_amount_rank) #descending=False
        .select(["sid", "trade_month_id"])
    )

    # 4. PIT join: M ---> M+1 trade_month_id (filter out suspended zero-volume days)
    filtered_uni_lf = (
        uni_lf.join(
            sample_lf,
            left_on=["sid", "month_id"],
            right_on=["sid", "trade_month_id"], # next
            how="inner",
        )
        .filter(pl.col("volume").is_not_null() & (pl.col("volume") > 0))
    )

    if start_date is not None:
        filtered_uni_lf = filtered_uni_lf.filter(pl.col("day").cast(pl.Int64) >= int(start_date))

    return filtered_uni_lf
