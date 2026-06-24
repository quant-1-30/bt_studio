import polars as pl
from bt_studio.utils.common import _collect_stream_sync


def universe_sample(universe_lf: pl.LazyFrame, daily_lf: pl.LazyFrame, exceed=120, topk=0.80) -> pl.LazyFrame:
    meta_lf = universe_lf.select(["sid", "first_trading"])
    
    uni_lf = (
        daily_lf
        .join(meta_lf, on="sid", how="left")
        .with_columns([
            pl.col("day").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d").alias("date"),
            pl.col("first_trading").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d").alias("ipo_date")
        ])
    )
    
    uni_lf = (
        uni_lf
        .sort(["sid", "day"])
        .with_columns([
            pl.col("date").dt.strftime("%Y%m").cast(pl.Int32).alias("month_id"),
            # dt.offset_by("1mo") solve 12 to 13 bug
            pl.col("date").dt.offset_by("1mo").dt.strftime("%Y%m").cast(pl.Int32).alias("trade_month_id"),
        ])
    )
    
    sample_lf = (
        uni_lf
        .group_by(["sid", "month_id"])
        .agg([
            pl.col("amount").mean().alias("avg_amount"),
            pl.col("date").max().alias("last_trade_date"), 
            ((pl.col("date").max() - pl.col("ipo_date").max()).dt.total_days()).alias("days_since_ipo"),
            pl.col("trade_month_id").first().alias("trade_month_id") 
        ])
        .filter(pl.col("days_since_ipo") >= exceed)
        .with_columns(
            rank = pl.col("avg_amount").rank(descending=True).over("month_id"),
            total_rank = pl.col("sid").count().over("month_id")
        )
        .filter(pl.col("rank") <= pl.col("total_rank") * topk)
        .select(["sid", "trade_month_id"])
    )
    
    filtered_uni_lf = (
        uni_lf
        .join(sample_lf, left_on=["sid", "month_id"], right_on=["sid", "trade_month_id"], how="inner")
        .filter(pl.col("volume") > 0)
        .filter(pl.col("high") > pl.col("low"))
    )
    return filtered_uni_lf

