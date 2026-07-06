import polars as pl


def build_fsm_panel(all_feat_lf: list[pl.LazyFrame], daily_lf: pl.LazyFrame, config: dict) -> pl.DataFrame:
    # =========================================================================
    # config 
    # =========================================================================
    ds = config["downsample"]
    bars_per_day = 240 // ds

    # =========================================================================
    # schema align with aligned_lf
    # =========================================================================
    daily_schema = daily_lf.collect_schema()

    if daily_schema["day"] in [pl.Int32, pl.Int64]:
        daily_lf = daily_lf.with_columns(pl.col("day").cast(pl.String).str.to_date("%Y%m%d"))
    elif daily_schema["day"] == pl.String:
        daily_lf = daily_lf.with_columns(pl.col("day").str.to_date("%Y%m%d"))
    elif daily_schema["day"] == pl.Datetime:
        daily_lf = daily_lf.with_columns(pl.col("day").cast(pl.Date))

    daily_lf = daily_lf.with_columns([
        pl.col("sid").cast(pl.String).str.strip_chars(" \x00\t\n"), 
        pl.col("day").cast(pl.Date)
    ])
    
    # =========================================================================
    # schema align with aligned_lf
    # =========================================================================

    # all_feat_lf = pl.concat(aligned_lfs)
    feat_schema = all_feat_lf.collect_schema()
    
    if feat_schema["day"] in [pl.Int32, pl.Int64]:
        all_feat_lf = all_feat_lf.with_columns(pl.col("day").cast(pl.String).str.to_date("%Y%m%d"))
    elif feat_schema["day"] == pl.String:
        all_feat_lf = all_feat_lf.with_columns(pl.col("day").str.to_date("%Y%m%d"))
    elif feat_schema["day"] == pl.Datetime:
        all_feat_lf = all_feat_lf.with_columns(pl.col("day").cast(pl.Date))

    all_feat_lf = all_feat_lf.with_columns([
        pl.col("sid").cast(pl.String).str.strip_chars(" \x00\t\n"), 
        pl.col("day").cast(pl.Date)
    ])

    # =========================================================================
    # daily_ret and vol
    # =========================================================================
    daily_ret_lf = (
        daily_lf.sort(["sid", "day"])
        .with_columns([
            (pl.col("close") / pl.col("close").shift(1).over("sid") - 1.0).alias("daily_ret")
        ])
        .with_columns([
            # pl.col("daily_ret").rolling_std(window_size=20, min_periods=5)
            #   .over("sid").fill_null(strategy="forward")
            #   .clip(lower_bound=0.005).alias("vol_20d"),
            (pl.col("close").shift(-1).over("sid") / pl.col("close") - 1.0).alias("fwd_ret_1"),
            (pl.col("close").shift(-2).over("sid") / pl.col("close") - 1.0).alias("fwd_ret_2"),
            (pl.col("close").shift(-3).over("sid") / pl.col("close") - 1.0).alias("fwd_ret_3")
        ])
    )
 
    # =========================================================================
    # downsample
    # =========================================================================
    if ds > 1:
        all_feat_lf = all_feat_lf.filter((pl.col("bar_idx") % ds) == 0)

    # =========================================================================
    # join 
    # =========================================================================
    curve_lf = (
        all_feat_lf
        .sort(["day", "sid", "bar_idx"])
        .group_by(["day", "sid"])
        .agg([
            pl.col("ofi_ratio").alias("daily_curve"),
            pl.len().alias("curve_len")  
        ])
        .filter(pl.col("curve_len") == bars_per_day) 
    )

    # =========================================================================
    # crossover concat 
    # =========================================================================

    shift_exprs = [
        pl.col("daily_curve").shift(i).over("sid").alias(f"lag_{i}") if i > 0 
        else pl.col("daily_curve").alias(f"lag_{i}") # i == 0 alias
        for i in reversed(range(config["cross_days"]))
    ]

    curve_lf = (
        curve_lf.sort(["sid", "day"])
        .with_columns(shift_exprs)
        .drop_nulls(subset=[f"lag_{i}" for i in range(config["cross_days"])]) 
    )

    panel_lf = curve_lf.join(
        daily_ret_lf.select(["day", "sid", "fwd_ret_1", "fwd_ret_2", "fwd_ret_3"]),
        on=["day", "sid"], how="inner"
    )
    return panel_lf
