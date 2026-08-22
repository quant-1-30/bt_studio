#! /usr/bin/env python3

import polars as pl


def universe_sample(
    universe_lf: pl.LazyFrame,
    daily_lf: pl.LazyFrame,
    common_config: dict,
) -> pl.LazyFrame:
    """Point-in-Time(PIT) align
    
    - month_id (202209) 
    - trade_month_id = 202210 
    - start_date clip
    """
    meta_lf = universe_lf.select(["sid", "first_trading"])
    eps = common_config.get("eps", 1e-8)
    start_date = common_config.get("start_date")  

    # 1. canonalize
    uni_lf = (
        daily_lf.join(meta_lf, on="sid", how="left")
        .with_columns([
            pl.col("day").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d").alias("date"),
            pl.col("first_trading").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d").alias("ipo_date"),
            pl.when(pl.col("sid").cast(pl.String).str.starts_with("688"))
            .then(pl.lit("688"))
            .otherwise(pl.col("sid").cast(pl.String).str.slice(0, 1))
            .alias("board"),
        ])
    )

    # 2. month_id / trade_month_id
    uni_lf = (
        uni_lf.sort(["sid", "day"])
        .with_columns([
            (pl.col("date").dt.year() * 100 + pl.col("date").dt.month()).alias("month_id"),
            (
                (pl.col("date").dt.year() + (pl.col("date").dt.month() == 12).cast(pl.Int32)) * 100
                + (pl.col("date").dt.month() % 12 + 1)
            ).alias("trade_month_id"),
            ((pl.col("high") - pl.col("low")) / (pl.col("close").shift(1).over("sid") + eps)).alias("amplitude"),
        ])
    )

    # 3. filter by rank and theta
    sample_lf = (
        uni_lf.group_by(["sid", "month_id", "board"])
        .agg([
            pl.col("amount").mean().alias("avg_amount"),
            (pl.col("volume").filter(pl.col("volume") > 0).count() / pl.col("volume").count()).alias("active_ratio"),
            ((pl.col("date").max() - pl.col("ipo_date").max()).dt.total_days()).alias("days_since_ipo"),
            pl.col("trade_month_id").first().alias("trade_month_id"),
        ])
        .filter(
            (pl.col("days_since_ipo") >= common_config.get("days_since_ipo", 120))
            & (pl.col("active_ratio") >= 0.90)
        )
        .with_columns([
            (pl.col("avg_amount").rank(method="average") / pl.len()).over("month_id").alias("amount_rank")
        ])
        # is_between
        .filter(pl.col("amount_rank") >= 0.10) 
        .select(["sid", "trade_month_id"])
    )

    # 4. align and filter by warmup 
    filtered_uni_lf = (
        uni_lf.join(
            sample_lf,
            left_on=["sid", "month_id"],        
            right_on=["sid", "trade_month_id"], 
            how="inner",
        )
        .filter((pl.col("volume") > 0) & (pl.col("high") > pl.col("low")))
    )

    if start_date is not None:
        filtered_uni_lf = filtered_uni_lf.filter(pl.col("day").cast(pl.Int64) >= start_date)

    return filtered_uni_lf
