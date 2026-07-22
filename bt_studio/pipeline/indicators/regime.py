import polars as pl


def regime_indicator(daily_lf: pl.LazyFrame, config: dict) -> pl.LazyFrame:
    ma_window = config.get("ma_window", 20)
    
    indicator_lf = (
        daily_lf.sort(["sid", "day"])
        .with_columns([
            pl.col("close").rolling_mean(window_size=ma_window, min_periods=max(1, ma_window//2))
            .over("sid").alias("ma_trend")
        ])
        .with_columns([
            (pl.col("close") > pl.col("ma_trend")).cast(pl.Int8).alias("regime_raw_signal")
        ])
        .with_columns([
            pl.col("regime_raw_signal").shift(1).over("sid")
            .fill_null(0) 
            .alias("regime_signal")
        ])
        .drop(["ma_trend", "regime_raw_signal"])
    )
    
    return indicator_lf

