import polars as pl


# def align_skeleton(tick_lf: pl.LazyFrame) -> pl.LazyFrame:
#     # ====================================================================
#     # Eager Mode avoid Lazy Optimize Bug
#     # ====================================================================
#     df = tick_lf.collect(engine="streaming") 
#     if df.height == 0:
#         return df.lazy()

#     # TimeStamp to  Date
#     df = df.with_columns(
#         (pl.col("tick") * 1000).cast(pl.Int64).cast(pl.Datetime("ms")).alias("tick_dt")
#     )

#     # ====================================================================
#     #  UTC + 8 Hour ---> Aisa Shanghai
#     # ====================================================================
#     median_hour = df.select(pl.col("tick_dt").dt.hour().median()).item()
#     if median_hour is not None and median_hour < 8:
#         df = df.with_columns(tick_dt = pl.col("tick_dt").dt.offset_by("8h"))

#     # default i8 ---> Int32
#     df = df.with_columns([
#         (
#             pl.col("tick_dt").dt.hour().cast(pl.Int32) * 60 + 
#             pl.col("tick_dt").dt.minute().cast(pl.Int32)
#         ).alias("to_minutes"),
#         pl.col("tick_dt").dt.date().alias("day")
#     ])

#     # filter by A trading 
#     df = df.filter(
#         ((pl.col("to_minutes") >= 9 * 60 + 30) & (pl.col("to_minutes") < 11 * 60 + 30)) |
#         ((pl.col("to_minutes") >= 13 * 60) & (pl.col("to_minutes") < 15 * 60))
#     )

#     if df.height == 0:
#         return df.lazy()

#     # minute ---> 0-239 index
#     df = df.with_columns(
#         minute_idx = pl.when(pl.col("to_minutes") < 11 * 60 + 30)
#                        .then(pl.col("to_minutes") - (9 * 60 + 30))  
#                        .otherwise((pl.col("to_minutes") - (13 * 60)) + 120)
#                        .cast(pl.Int32)
#     ).filter((pl.col("minute_idx") >= 0) & (pl.col("minute_idx") < 240))

#     # ====================================================================
#     # explode replace Cross Join
#     # ====================================================================
#     unique_pairs = df.select(["sid", "day"]).unique()
#     skeleton_df = (
#         unique_pairs
#         .with_columns(pl.int_ranges(0, 240, dtype=pl.Int32).alias("minute_idx"))
#         .explode("minute_idx")
#     )
    
#     # join skeleton_df with df to fill missing minute_idx, then forward/backward fill close, and fill other columns with default values
#     padded_df = (
#         skeleton_df
#         .join(df, on=["day", "sid", "minute_idx"], how="left")
#         .sort(["day", "sid", "minute_idx"])
#         .with_columns([
#             pl.col("close").forward_fill().backward_fill().over(["day", "sid"]),
#         ])
#         .with_columns([
#             pl.col("open").fill_null(pl.col("close")),
#             pl.col("high").fill_null(pl.col("close")),
#             pl.col("low").fill_null(pl.col("close")),
#             pl.col("amount").fill_null(0.0),
#             pl.col("volume").fill_null(0.0)
#         ])
#         .drop(["tick", "to_minutes"])
#         .rename({"tick_dt": "tick"})
#     )
#     return padded_df.lazy()


# def build_fsm_panel(
#     all_feat_lf: pl.LazyFrame, 
#     daily_lf: pl.LazyFrame, 
#     tune_config: dict, 
#     common_config: dict,
#     is_train: bool = True
#     ) -> pl.DataFrame:
#     # =========================================================================
#     # config 
#     # =========================================================================
#     # train inner / oss left 
#     join_how = "inner" if is_train else "left"

#     ds = tune_config["downsample"]
#     bars_per_day = 240 // ds

#     # =========================================================================
#     # schema align with aligned_lf
#     # =========================================================================
#     daily_schema = daily_lf.collect_schema()

#     if daily_schema["day"] in [pl.Int32, pl.Int64]:
#         daily_lf = daily_lf.with_columns(pl.col("day").cast(pl.String).str.to_date("%Y%m%d"))
#     elif daily_schema["day"] == pl.String:
#         daily_lf = daily_lf.with_columns(pl.col("day").str.to_date("%Y%m%d"))
#     elif daily_schema["day"] == pl.Datetime:
#         daily_lf = daily_lf.with_columns(pl.col("day").cast(pl.Date))

#     daily_lf = daily_lf.with_columns([
#         pl.col("sid").cast(pl.String).str.strip_chars(" \x00\t\n"), 
#         pl.col("day").cast(pl.Date)
#     ])
    
#     # =========================================================================
#     # schema align with aligned_lf
#     # =========================================================================
#     feat_schema = all_feat_lf.collect_schema()
    
#     if feat_schema["day"] in [pl.Int32, pl.Int64]:
#         all_feat_lf = all_feat_lf.with_columns(pl.col("day").cast(pl.String).str.to_date("%Y%m%d"))
#     elif feat_schema["day"] == pl.String:
#         all_feat_lf = all_feat_lf.with_columns(pl.col("day").str.to_date("%Y%m%d"))
#     elif feat_schema["day"] == pl.Datetime:
#         all_feat_lf = all_feat_lf.with_columns(pl.col("day").cast(pl.Date))

#     all_feat_lf = all_feat_lf.with_columns([
#         pl.col("sid").cast(pl.String).str.strip_chars(" \x00\t\n"), 
#         pl.col("day").cast(pl.Date)
#     ])

#     # =========================================================================
#     # daily_ret and vol
#     # =========================================================================
#     entry_idx = 240 - common_config["exclude_bars"] # e.g 14:50 ---> 230 
#     entry_price_lf = (
#         all_feat_lf
#         .filter(pl.col("bar_idx") == entry_idx ) 
#         .select(["day", "sid", pl.col("close").alias("entry_price")])
#     )
    
#     daily_ret_lf = (
#         daily_lf.join(entry_price_lf, on=["day", "sid"], how="left")
#         .sort(["sid", "day"])
#         # next close / today 14:50 - 1 
#         .with_columns([
#             (pl.col("close").shift(-1).over("sid") / pl.col("entry_price") - 1.0).alias("fwd_ret_1"),
#             (pl.col("close").shift(-2).over("sid") / pl.col("entry_price") - 1.0).alias("fwd_ret_2"),
#             (pl.col("close").shift(-3).over("sid") / pl.col("entry_price") - 1.0).alias("fwd_ret_3")
#         ])
#         #  not found today 14:50 ---> today close
#         .with_columns(
#             pl.col("fwd_ret_1").fill_null((pl.col("close").shift(-1).over("sid") / pl.col("close") - 1.0))
#         )
#     )
#     # =========================================================================
#     # downsample
#     # =========================================================================
#     if ds > 1:
#         all_feat_lf = all_feat_lf.filter((pl.col("bar_idx") % ds) == 0)

#     # =========================================================================
#     # join 
#     # =========================================================================
#     curve_lf = (
#         all_feat_lf
#         .sort(["day", "sid", "bar_idx"])
#         .group_by(["day", "sid"])
#         .agg([
#             pl.col("ofi_ratio").alias("daily_curve"),
#             pl.len().alias("curve_len")  
#         ])
#         .filter(pl.col("curve_len") == bars_per_day) 
#     )

#     # =========================================================================
#     # crossover concat 
#     # =========================================================================

#     shift_exprs = [
#         pl.col("daily_curve").shift(i).over("sid").alias(f"lag_{i}") if i > 0 
#         else pl.col("daily_curve").alias(f"lag_{i}") # i == 0 alias
#         for i in reversed(range(tune_config["cross_days"]))
#     ]

#     curve_lf = (
#         curve_lf.sort(["sid", "day"])
#         .with_columns(shift_exprs)
#         .drop_nulls(subset=[f"lag_{i}" for i in range(tune_config["cross_days"])]) 
#     )

#     panel_lf = curve_lf.join(
#         daily_ret_lf.select(["day", "sid", "fwd_ret_1", "fwd_ret_2", "fwd_ret_3"]),
#         on=["day", "sid"], 
#         how=join_how
#     )
#     return panel_lf


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
        .drop(["to_minutes"])
        # .drop(["to_minutes", "tick_dt"])
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
    min_required_bars = max(2 * m - 2, 5)

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
    # packing list
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
        .filter(pl.col("curve_len") >= min_required_bars)
    )

    # =========================================================================
    # Crossover sort by stock and date
    # =========================================================================
    cross_days = tune_config["cross_days"]

    curve_lf = curve_lf.sort(["sid", "day"])

    shift_exprs = []
    for i in reversed(range(cross_days)):
        if i > 0:
            expr = (
                pl.col("daily_curve")
                .sort_by("day")
                .shift(i)
                .over("sid")
                .alias(f"lag_{i}")
            )
        else:
            expr = pl.col("daily_curve").alias(f"lag_{i}")
        shift_exprs.append(expr)

    curve_lf = curve_lf.with_columns(shift_exprs).drop_nulls(
        subset=[f"lag_{i}" for i in range(cross_days)]
    )

    # =========================================================================
    # join
    # =========================================================================
    panel_lf = curve_lf.join(
        daily_ret_lf.select(
            ["day", "sid", "fwd_ret_1", "fwd_ret_2", "fwd_ret_3"]
        ),
        on=["day", "sid"],
        how=join_how,
    )

    return panel_lf
