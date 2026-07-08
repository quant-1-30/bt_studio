import polars as pl


def build_ofi(aligned_lf: pl.LazyFrame, common_config: dict) -> pl.LazyFrame:
    ofi_expr = (
        ((pl.col("close") * 2 - pl.col("high") - pl.col("low")) / 
         (pl.col("high") - pl.col("low") + 1e-8)) * pl.col("amount")
    )
    
    feat_lf = (
        aligned_lf
        .with_columns([
            # ((pl.col("minute_idx") -1) // downsample_m).cast(pl.Int32).alias("bar_idx"),
            pl.col("minute_idx").alias("bar_idx"),
            ofi_expr.alias("raw_ofi")
        ])
        .group_by(["day", "sid", "bar_idx"])
        .agg([
            pl.col("raw_ofi").sum().alias("agg_ofi"),
            pl.col("amount").sum().alias("agg_amount"),
            pl.col("close").last().alias("close") 
        ])
        .sort(["day", "sid", "bar_idx"])
        .with_columns([
            (pl.col("agg_ofi").cum_sum().over(["day", "sid"]) / 
             (pl.col("agg_amount").cum_sum().over(["day", "sid"]) + 1e-8)).alias("ofi_ratio")
        ])
    )
    return feat_lf


def build_ofi_vol(aligned_lf: pl.LazyFrame, common_config: dict) -> pl.LazyFrame:
    ofi_expr = (
        ((pl.col("close") * 2 - pl.col("high") - pl.col("low")) / 
         (pl.col("high") - pl.col("low") + 1e-8)) * pl.col("amount")
    )
    
    vol_expr = (pl.col("high") - pl.col("low")) / (pl.col("close") + 1e-8)
    
    feat_lf = (
        aligned_lf
        .with_columns([
            pl.col("minute_idx").alias("bar_idx"),
            ofi_expr.alias("raw_ofi"),
            vol_expr.alias("raw_vol")
        ])
        .group_by(["day", "sid", "bar_idx"])
        .agg([
            pl.col("raw_ofi").sum().alias("agg_ofi"),
            pl.col("amount").sum().alias("agg_amount"),
            pl.col("raw_vol").sum().alias("volatility"), 
            pl.col("close").last().alias("close") 
        ])
        .sort(["day", "sid", "bar_idx"])
        .with_columns([
            (pl.col("agg_ofi").cum_sum().over(["day", "sid"]) / 
             (pl.col("agg_amount").cum_sum().over(["day", "sid"]) + 1e-8)).alias("ofi_ratio")
        ])
    )
    return feat_lf
