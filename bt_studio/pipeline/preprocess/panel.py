import polars as pl
from bt_studio.pipeline.indicators import regime_indicator


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


def build_static_panel(
    all_feat_lf: pl.LazyFrame,
    daily_lf: pl.LazyFrame,
    common_config: dict,
    is_train: bool = True,
) -> pl.LazyFrame:
    """
    [FIX P1-P2] Build the tune_config-INDEPENDENT part of the panel.
    This includes: regime_indicator, calendar, T/T+1 forward returns, gap, z-scores.
    None of these depend on downsample/motif_minutes/threshold_r, so they should
    be computed ONCE and reused across all HPO trials.

    Returns a LazyFrame with columns: day, sid, z_gap, raw_*, fwd_z_*, intra_*
    """
    join_how = "inner" if is_train else "left"
    eps = common_config["eps"]

    # =========================================================================
    # daily and all
    # =========================================================================
    daily_lf = align_date_col(daily_lf).sort(["sid", "day"])
    all_feat_lf = align_date_col(all_feat_lf).sort(["sid", "day", "bar_idx"])

    daily_lf = regime_indicator(daily_lf, common_config.get("regime_filter", {}))

    # =========================================================================
    # calendar
    # =========================================================================
    calendar_lf = (
        daily_lf.select("day").unique().sort("day")
        .with_columns(pl.col("day").shift(-1).alias("t1_day"))
        .drop_nulls() # abandon last day
    )

    # =========================================================================
    # T Gap
    # =========================================================================
    entry_idx = 240 - common_config["exclude_bars"]

    t_base_lf = (
        all_feat_lf.filter(pl.col("bar_idx") == entry_idx)
        .select(["day", "sid", pl.col("close").alias("entry_price")])
        .join(daily_lf.select(["day", "sid", pl.col("close").alias("t_close")]), on=["day", "sid"], how="left")
        .join(calendar_lf, on="day", how="inner")
    )

    # =========================================================================
    # T+1
    # =========================================================================
    targets_rets = common_config["T1_rets"]
    bar_idx_to_name = {max(0, minutes - 1): name for name, minutes in targets_rets.items()}

    t1_exprs = [pl.col("open").filter(pl.col("bar_idx") == 0).first().alias("t1_open_price")]
    for b_idx, name in bar_idx_to_name.items():
        t1_exprs.append(pl.col("close").filter(pl.col("bar_idx") == b_idx).first().alias(f"t1_price_{name}"))

    target_bar_indices = [0] + list(bar_idx_to_name.keys())
    t1_lf = (
        all_feat_lf.filter(pl.col("bar_idx").is_in(target_bar_indices))
        .group_by(["day", "sid"])
        .agg(t1_exprs)
        .rename({"day": "t1_day"})
    )

    # =========================================================================
    # T and T+1 ---> Gap / PnL / Momeum
    # =========================================================================
    target_lf = t_base_lf.join(t1_lf, on=["t1_day", "sid"], how="left")
    target_names = list(targets_rets.keys())

    # Gap
    target_lf = target_lf.with_columns([
        pl.when(pl.col("t_close") > eps)
        .then(pl.col("t1_open_price") / pl.col("t_close") - 1.0)
        .otherwise(None)
        .alias("raw_gap")
    ] + [
        # PnL T 14:50 ---> T+1
        pl.when(pl.col("entry_price") > eps)
        .then(pl.col(f"t1_price_{name}") / pl.col("entry_price") - 1.0)
        .otherwise(None)
        .alias(f"raw_{name}")
        for name in target_names
    ] + [
        # T +1 Momeum
        pl.when(pl.col("t1_open_price") > eps)
        .then(pl.col(f"t1_price_{name}") / pl.col("t1_open_price") - 1.0)
        .otherwise(None)
        .alias(f"intra_{name}")
        for name in target_names
    ])

    # =========================================================================
    # Robust Z-Score
    # =========================================================================
    cols_to_zscore = ["gap"] + target_names
    for name in cols_to_zscore:
        z_col_name = "z_gap" if name == "gap" else f"fwd_z_{name}"

        target_lf = target_lf.with_columns([
            pl.col(f"raw_{name}").median().over("day").alias(f"med_{name}")
        ]).with_columns([
            (pl.col(f"raw_{name}") - pl.col(f"med_{name}")).abs().median().over("day").alias(f"mad_{name}")
        ]).with_columns([
            ((pl.col(f"raw_{name}") - pl.col(f"med_{name}")) /
             (1.4826 * pl.when(pl.col(f"mad_{name}") < eps).then(eps).otherwise(pl.col(f"mad_{name}"))))
            .clip(-3.0, 3.0).alias(z_col_name)
        ]).drop([f"med_{name}", f"mad_{name}"])

    output_cols = ["z_gap"] + [f"raw_{n}" for n in target_names] + [f"fwd_z_{n}" for n in target_names] + [f"intra_{n}" for n in target_names]
    target_lf = target_lf.select(["day", "sid"] + output_cols)

    return target_lf.select(["day", "sid"] + output_cols)


def extract_curves_from_panel(
    all_feat_lf: pl.LazyFrame,
    daily_lf: pl.LazyFrame,
    tune_config: dict,
    common_config: dict,
) -> pl.LazyFrame:
    """
    [FIX P1-P2] Build the tune_config-DEPENDENT part of the panel.
    This does downsample + curve group_by + regime masking.
    Called per-trial with the trial's tune_config.

    Returns a LazyFrame with columns: day, sid, lag_0
    """
    ds = tune_config["downsample"]
    bars_per_day = 240 // ds

    daily_lf = align_date_col(daily_lf).sort(["sid", "day"])
    all_feat_lf = align_date_col(all_feat_lf).sort(["sid", "day", "bar_idx"])
    daily_lf = regime_indicator(daily_lf, common_config.get("regime_filter", {}))

    # =========================================================================
    # Downsample and Masking
    # =========================================================================
    if ds > 1:
        all_feat_lf = all_feat_lf.filter((pl.col("bar_idx") % ds) == 0)

    # =========================================================================
    # CRITICAL: sort_by("bar_idx") ensures intraday time ordering inside list
    # Polars group_by is multi-threaded hash aggregation that does NOT guarantee
    # element order inside the aggregated list. Without sort_by, 09:30 data could
    # end up after 14:00, turning the OFI curve into shuffled white noise.
    # =========================================================================
    curve_lf = (
        all_feat_lf.group_by(["day", "sid"])
        .agg([
            pl.col("ofi_ratio").sort_by("bar_idx").alias("daily_curve"),
            pl.col("ofi_ratio").count().alias("curve_len"),
        ])
        .filter(pl.col("curve_len") == bars_per_day)
    )

    curve_lf = curve_lf.join(
        daily_lf.select(["day", "sid", "regime_signal"]), on=["day", "sid"], how="left"
    ).with_columns([
        pl.when(pl.col("regime_signal") == 1)
        .then(pl.col("daily_curve"))
        .otherwise(None)
        .alias("lag_0")
    ]).drop(["regime_signal", "daily_curve", "curve_len"])

    return curve_lf


def build_fsm_panel(
    all_feat_lf: pl.LazyFrame,
    daily_lf: pl.LazyFrame,
    tune_config: dict,
    common_config: dict,
    is_train: bool = True,
) -> pl.LazyFrame:
    """
    Backward-compatible wrapper: builds full panel in one call.
    For HPO, prefer pre-computing build_static_panel once and calling
    extract_curves_from_panel per trial (see tune_train.py).
    """
    static_lf = build_static_panel(all_feat_lf, daily_lf, common_config, is_train)
    curve_lf = extract_curves_from_panel(all_feat_lf, daily_lf, tune_config, common_config)
    return curve_lf.join(static_lf, on=["day", "sid"], how="inner" if is_train else "left")
