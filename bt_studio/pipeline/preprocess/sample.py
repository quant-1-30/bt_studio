import polars as pl
from bt_studio.utils.common import _collect_stream_sync


def universe_sample(universe_lf: pl.LazyFrame, daily_lf: pl.LazyFrame, common_config: dict) -> pl.LazyFrame:
    meta_lf = universe_lf.select(["sid", "first_trading"])
    eps = 1e-6
    
    uni_lf = (
        daily_lf
        .join(meta_lf, on="sid", how="left")
        .with_columns([
            pl.col("day").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d").alias("date"),
            pl.col("first_trading").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d").alias("ipo_date"),
            pl.when(pl.col("sid").cast(pl.String).str.starts_with("688")).then(pl.lit("688"))
              .otherwise(pl.col("sid").cast(pl.String).str.slice(0, 1))
              .alias("board")
        ])
    )
    
    uni_lf = (
        uni_lf.sort(["sid", "day"])
        .with_columns([
            pl.col("date").dt.strftime("%Y%m").cast(pl.Int32).alias("month_id"),
            pl.col("date").dt.offset_by("1mo").dt.strftime("%Y%m").cast(pl.Int32).alias("trade_month_id"),
            # Daily Amplitude)
            ((pl.col("high") - pl.col("low")) / (pl.col("close").shift(1).over("sid") + eps)).alias("amplitude")
        ])
    )
    
    sample_lf = (
        uni_lf
        .group_by(["sid", "month_id", "board"])
        .agg([
            pl.col("amount").mean().alias("avg_amount"),
            pl.col("amplitude").mean().alias("avg_amplitude"),
            (pl.col("volume").filter(pl.col("volume") > 0).count() / pl.col("volume").count()).alias("active_ratio"),
            pl.col("date").max().alias("last_trade_date"), 
            ((pl.col("date").max() - pl.col("ipo_date").max()).dt.total_days()).alias("days_since_ipo"),
            pl.col("trade_month_id").first().alias("trade_month_id") 
        ])
        .filter(
            # suspend within 10% by month
            (pl.col("days_since_ipo") >= common_config["days_since_ipo"]) 
            & (pl.col("active_ratio") >= 0.90) 
        ) 
        .with_columns([
            (pl.col("avg_amount").rank(method="average") / pl.len()).over(["month_id", "board"]).alias("amount_pct")
        ])
        .filter(pl.col("amount_pct").is_between(0.20, 0.85))
        .with_columns([
            (pl.col("avg_amplitude").rank(descending=True, method="average") / pl.len()).over(["month_id", "board"]).alias("volatility_rank")
        ])
        .filter(pl.col("volatility_rank") <= common_config["top_k_ratio"])
        .select(["sid", "trade_month_id"])
    )
    
    filtered_uni_lf = (
        uni_lf
        .join(sample_lf, left_on=["sid", "month_id"], right_on=["sid", "trade_month_id"], how="inner")
        .filter((pl.col("volume") > 0) & (pl.col("high") > pl.col("low"))) 
    )
    return filtered_uni_lf

