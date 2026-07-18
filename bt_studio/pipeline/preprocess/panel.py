import polars as pl


def align_skeleton(tick_lf: pl.LazyFrame) -> pl.LazyFrame:
    processed_lf = tick_lf.with_columns(
        (pl.col("tick") * 1000)
        .cast(pl.Int64)
        .cast(pl.Datetime("ms"))
        .alias("tick_dt")
    ).with_columns(
        pl.col("tick_dt").dt.date().alias("day")
    )

    processed_lf = processed_lf.with_columns(
        pl.when(
            pl.col("tick_dt").dt.hour().median().over(["day"]) < 8 # 9 -15 median > 8
        )  
        .then(pl.col("tick_dt").dt.offset_by("8h"))
        .otherwise(pl.col("tick_dt"))
        .alias("tick_dt")
    )

    processed_lf = processed_lf.with_columns(
        [
            (
                pl.col("tick_dt").dt.hour().cast(pl.Int32) * 60
                + pl.col("tick_dt").dt.minute().cast(pl.Int32)
            ).alias("to_minutes"),
            pl.col("tick_dt").dt.date().alias("day"),
        ]
    ).filter(
        ((pl.col("to_minutes") >= 9 * 60 + 30) & (pl.col("to_minutes") < 11 * 60 + 30)) |
        ((pl.col("to_minutes") >= 13 * 60) & (pl.col("to_minutes") < 15 * 60))
    )

    # timestamp to minute_idx
    processed_lf = processed_lf.with_columns(
        minute_idx=pl.when(pl.col("to_minutes") < 11 * 60 + 30)
        .then(pl.col("to_minutes") - (9 * 60 + 30))
        .otherwise((pl.col("to_minutes") - (13 * 60)) + 120)
        .cast(pl.Int32)
    )

    # ====================================================================
    # 240 skeleton
    # ====================================================================
    unique_pairs = processed_lf.select(["sid", "day"]).unique()

    skeleton_lf = unique_pairs.with_columns(
        pl.int_ranges(0, 240, dtype=pl.Int32).alias("minute_idx")
    ).explode("minute_idx")

    # ====================================================================
    # padding
    # ====================================================================
    padded_lf = (
        skeleton_lf.join(
            processed_lf, on=["day", "sid", "minute_idx"], how="left"
        )
        .sort(["day", "sid", "minute_idx"])
        # ensure padding in sid
        .with_columns(
            [
                pl.col("close")
                .forward_fill() 
                .over(["day", "sid"])
                .alias("close"),
            ]
        )
        .with_columns(
            [
                pl.col("close")
                .backward_fill()
                .over(["day", "sid"])
                .alias("close")
            ]
        )
        .with_columns(
            [
                pl.col("open").fill_null(pl.col("close")),
                pl.col("high").fill_null(pl.col("close")),
                pl.col("low").fill_null(pl.col("close")),
                pl.col("amount").fill_null(0.0),
                pl.col("volume").fill_null(0.0),
            ]
        )
        .drop(["to_minutes", "tick_dt", "tick"])
        # .rename({"tick_dt": "tick"})
    )
    return padded_lf


def build_fsm_panel(
    all_feat_lf: pl.LazyFrame,
    daily_lf: pl.LazyFrame,
    tune_config: dict,
    common_config: dict,
    is_train: bool = True,
) -> pl.DataFrame:
    # =========================================================================
    # config
    # =========================================================================
    join_how = "inner" if is_train else "left"

    ds = tune_config["downsample"]
    m = tune_config["motif_minutes"] // ds
    bars_per_day = 240 // ds  

    # =========================================================================
    # cast type
    # =========================================================================
    def align_date_col(lf: pl.LazyFrame) -> pl.LazyFrame:
        schema = lf.collect_schema()
        if schema["day"] in [pl.Int32, pl.Int64]:
            lf = lf.with_columns(
                pl.col("day").cast(pl.String).str.to_date("%Y%m%d")
            )
        elif schema["day"] == pl.String:
            lf = lf.with_columns(pl.col("day").str.to_date("%Y%m%d"))
        elif schema["day"] == pl.Datetime:
            lf = lf.with_columns(pl.col("day").cast(pl.Date))
        return lf.with_columns(
            [
                pl.col("sid").cast(pl.String).str.strip_chars(" \x00\t\n"),
                pl.col("day").cast(pl.Date),
            ]
        )

    daily_lf = align_date_col(daily_lf)
    all_feat_lf = align_date_col(all_feat_lf)

    # =========================================================================
    # 14:50 -> 230 -> fut_ret
    # =========================================================================
    entry_idx = 240 - common_config["exclude_bars"]  
    entry_price_lf = all_feat_lf.filter(pl.col("bar_idx") == entry_idx).select(
        ["day", "sid", pl.col("close").alias("entry_price")]
    )

    daily_ret_lf = (
        daily_lf.join(entry_price_lf, on=["day", "sid"], how="left")
        .sort(["sid", "day"]).with_columns(
            [
                (
                    pl.col("close").shift(-1).over("sid")
                    / pl.col("entry_price")
                    - 1.0
                ).alias("fwd_ret_1"),
                (
                    pl.col("close").shift(-2).over("sid")
                    / pl.col("entry_price")
                    - 1.0
                ).alias("fwd_ret_2"),
                (
                    pl.col("close").shift(-3).over("sid")
                    / pl.col("entry_price")
                    - 1.0
                ).alias("fwd_ret_3"),
            ]
        )
        .with_columns(
            pl.col("fwd_ret_1").fill_null(
                (pl.col("close").shift(-1).over("sid") / pl.col("close") - 1.0)
            )
        )
    )

    # =========================================================================
    # downsample
    # =========================================================================
    if ds > 1:
        all_feat_lf = all_feat_lf.filter((pl.col("bar_idx") % ds) == 0)

    # =========================================================================
    # Trading Calendar Index to Filter Suspend
    # =========================================================================
    calendar_lf = (
        daily_lf.select("day")
        .unique()
        .sort("day")
        .with_row_index(name="trade_day_idx", offset=0) # C++ autoincrement
        .with_columns(pl.col("trade_day_idx").cast(pl.Int32))
    )

    # =========================================================================
    # packing list == keep intraday dim
    # =========================================================================
    curve_lf = (
        all_feat_lf.sort(["day", "sid", "bar_idx"])
        .group_by(["day", "sid"])
        .agg(
            [
                pl.col("ofi_ratio").alias("daily_curve"),
                pl.col("ofi_ratio").count().alias("curve_len"),
            ]
        )
        .filter(pl.col("curve_len") == bars_per_day)  
    )

    curve_lf = curve_lf.join(calendar_lf, on="day", how="left")

    # =========================================================================
    # Crossover sort by stock and date
    # =========================================================================
    actual_cross_days = max(1, tune_config.get("cross_days", 1)) # ensure range(1) ---> 0
    
    curve_lf = curve_lf.sort(["sid", "day"])

    shift_exprs = [
        pl.col("daily_curve").shift(i).over("sid").alias(f"lag_{i}") 
        if i > 0 else pl.col("daily_curve").alias(f"lag_{i}")
        for i in reversed(range(actual_cross_days))
    ]

    curve_lf = (
        curve_lf
        .with_columns(
            (pl.col("trade_day_idx") - pl.col("trade_day_idx").shift(1).over("sid"))
            .fill_null(1)
            .alias("day_diff_from_last")
        )
        .filter( # better than pl.when
            pl.col("day_diff_from_last")
            # min_periods=1 avoid filter within window 
            .rolling_max(window_size=actual_cross_days, min_periods=1)
            .over("sid") == 1
        )
        .with_columns(shift_exprs)
        .drop(["trade_day_idx", "day_diff_from_last"])
        .drop_nulls(subset=[f"lag_{i}" for i in range(actual_cross_days)])
    )

    # =========================================================================
    # final join
    # =========================================================================
    panel_lf = curve_lf.join(
        daily_ret_lf.select(
            ["day", "sid", "fwd_ret_1", "fwd_ret_2", "fwd_ret_3"]
        ),
        on=["day", "sid"],
        how=join_how,
    )

    return panel_lf
