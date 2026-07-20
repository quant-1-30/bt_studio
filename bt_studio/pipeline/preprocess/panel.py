import polars as pl


def build_fsm_panel(
    all_feat_lf: pl.LazyFrame,
    daily_lf: pl.LazyFrame,
    tune_config: dict,
    common_config: dict,
    is_train: bool = True,
) -> pl.DataFrame:
    
    join_how = "inner" if is_train else "left"
    ds = tune_config["downsample"]
    bars_per_day = 240 // ds  

    # =========================================================================
    # Type Align
    # =========================================================================
    def align_date_col(lf: pl.LazyFrame) -> pl.LazyFrame:
        schema = lf.collect_schema()
        if schema["day"] in [pl.Int32, pl.Int64]:
            lf = lf.with_columns(pl.col("day").cast(pl.String).str.to_date("%Y%m%d"))
        elif schema["day"] == pl.String:
            lf = lf.with_columns(pl.col("day").str.to_date("%Y%m%d"))
        elif schema["day"] == pl.Datetime:
            lf = lf.with_columns(pl.col("day").cast(pl.Date))
        return lf.with_columns([
            pl.col("sid").cast(pl.String).str.strip_chars(" \x00\t\n"),
            pl.col("day").cast(pl.Date),
        ])

    daily_lf = align_date_col(daily_lf)
    all_feat_lf = align_date_col(all_feat_lf)

    # =========================================================================
    # fut_ret normalize by mad
    # =========================================================================
    entry_idx = 240 - common_config["exclude_bars"]  
    entry_price_lf = all_feat_lf.filter(pl.col("bar_idx") == entry_idx).select(
        ["day", "sid", pl.col("close").alias("entry_price")]
    )

    stats_windows = common_config["stats_windows"]
    
    ret_exprs = [
        (pl.col("close").shift(-p).over("sid") / pl.col("entry_price") - 1.0).alias(f"raw_ret_{p}")
        for p in stats_windows
    ]

    daily_ret_lf = (
        daily_lf.join(entry_price_lf, on=["day", "sid"], how="left")
        .sort(["sid", "day"])
        .with_columns(ret_exprs)
    )
    
    # Step A: Median
    daily_ret_lf = daily_ret_lf.with_columns([
        pl.col(f"raw_ret_{p}").median().over("day").alias(f"median_{p}") 
        for p in stats_windows
    ])
    
    # Step B: MAD
    daily_ret_lf = daily_ret_lf.with_columns([
        (pl.col(f"raw_ret_{p}") - pl.col(f"median_{p}")).abs().median().over("day").alias(f"mad_{p}") 
        for p in stats_windows
    ])
    
    # Step C: Z-Score
    z_score_cols = [f"fwd_ret_{p}" for p in stats_windows]
    
    daily_ret_lf = daily_ret_lf.with_columns([
        ((pl.col(f"raw_ret_{p}") - pl.col(f"median_{p}")) / (1.4826 * pl.col(f"mad_{p}") + 1e-6)).alias(f"fwd_ret_{p}")
        for p in stats_windows
    ]).drop(
        [f"median_{p}" for p in stats_windows] + 
        [f"mad_{p}" for p in stats_windows] + 
        [f"raw_ret_{p}" for p in stats_windows] 
    )

    # =========================================================================
    # Skeleton Solve Suspending and Missing  
    # =========================================================================
    if ds > 1:
        all_feat_lf = all_feat_lf.filter((pl.col("bar_idx") % ds) == 0)

    calendar_lf = (
        daily_lf.select("day")
        .unique().sort("day")
        .with_row_index(name="trade_day_idx", offset=0)
        .with_columns(pl.col("trade_day_idx").cast(pl.Int32))
    )
    
    unique_sids_lf = daily_lf.select("sid").unique()
    skeleton_lf = unique_sids_lf.join(calendar_lf, how="cross")

    raw_curve_lf = (
        all_feat_lf.sort(["day", "sid", "bar_idx"])
        .group_by(["day", "sid"])
        .agg([
            pl.col("ofi_ratio").alias("daily_curve"),
            pl.col("ofi_ratio").count().alias("curve_len"),
        ])
        .filter(pl.col("curve_len") == bars_per_day)
        .select(["day", "sid", "daily_curve"])
    )

    curve_lf = skeleton_lf.join(raw_curve_lf, on=["day", "sid"], how="left")

    # =========================================================================
    # trading_days left join ---> shift 
    # =========================================================================
    actual_cross_days = max(1, tune_config.get("cross_days", 1))
    curve_lf = curve_lf.sort(["sid", "day"])

    shift_exprs = [
        pl.col("daily_curve").shift(i).over("sid").alias(f"lag_{i}")
        for i in reversed(range(actual_cross_days))
    ]

    curve_lf = curve_lf.with_columns(shift_exprs).drop(["daily_curve", "trade_day_idx"]) # lag_0

    # =========================================================================
    # final join
    # =========================================================================
    panel_lf = curve_lf.join(
        daily_ret_lf.select(["day", "sid"] + z_score_cols),
        on=["day", "sid"],
        how=join_how,
    )
    return panel_lf
