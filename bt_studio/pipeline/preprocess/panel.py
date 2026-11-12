
#! /usr/bin/env python3

from typing import List, Optional
import polars as pl
from bt_studio.constant import TRADING_BARS_PER_DAY
from bt_studio.utils.math_func import entry_bar_idx
from bt_studio.utils.months import align_date_col, resolve_oos_paths_with_warmup


def build_static_panel(
    train_feat_lf: pl.LazyFrame,
    daily_lf: pl.LazyFrame,
    common_config: dict,
    is_train: bool = True,
) -> pl.LazyFrame:
    """Build the tune_config-independent static panel (calendar and T+1 forward returns).

    Returns a LazyFrame with columns: day, sid, raw_*, intra_*
    (raw_{h} = T entry -> T+1 exit return; intra_{h} = T+1 Open -> T+1 exit return).
    """
    eps = common_config["eps"]
    entry_idx = entry_bar_idx(common_config)
    targets_rets = common_config["T1_rets"]
    target_names = list(targets_rets.keys())

    # 1. Align and deduplicate daily and intraday frames
    daily_lf = align_date_col(daily_lf).unique(subset=["sid", "day"], keep="last")
    train_feat_lf = align_date_col(train_feat_lf).sort(["sid", "day", "bar_idx"])

    # 2. Trading calendar mapping (T -> T+1)
    calendar_lf = (
        daily_lf.select("day").unique().sort("day")
        .with_columns(pl.col("day").shift(-1).alias("t1_day"))
    )

    # 3. Single-pass intraday anchor extraction & consecutive trading day shift over sid
    anchors_lf = (
        train_feat_lf.group_by(["day", "sid"])
        .agg([
            pl.col("close").filter(pl.col("bar_idx") == entry_idx).first().alias("bar_entry_price"),
            pl.col("volume").filter(pl.col("bar_idx") <= entry_idx).sum().alias("entry_vol_sum"),
            pl.col("open").filter(pl.col("bar_idx") == 0).first().alias("_day_open"),
            *[
                pl.col("close").filter(pl.col("bar_idx") == (int(minutes) - 1)).first().alias(f"{name}_close")
                for name, minutes in targets_rets.items()
            ],
        ])
        .join(calendar_lf, on="day", how="left")
        .join(daily_lf.select(["day", "sid", pl.col("close").alias("t_close")]), on=["day", "sid"], how="left")
        .sort(["sid", "day"])
        .with_columns([
            (pl.col("day").shift(-1).over("sid") == pl.col("t1_day")).fill_null(False).alias("has_t1"),
            pl.when(pl.col("entry_vol_sum") > 0)
            .then(pl.coalesce(["bar_entry_price", "t_close"]))
            .otherwise(None)
            .alias("entry_price"),
            pl.col("_day_open").shift(-1).over("sid").alias("t1_open_price"),
            *[
                pl.col(f"{name}_close").shift(-1).over("sid").alias(f"t1_price_{name}")
                for name in target_names
            ],
        ])
    )

    if is_train:
        anchors_lf = anchors_lf.filter(pl.col("has_t1"))

    # 4. Forward holding returns (raw_*) and T+1 intraday returns (intra_*)
    target_lf = anchors_lf.with_columns([
        pl.when(pl.col("has_t1") & (pl.col("entry_price") > eps))
        .then(pl.col(f"t1_price_{name}") / pl.col("entry_price") - 1.0)
        .otherwise(None)
        .alias(f"raw_{name}")
        for name in target_names
    ] + [
        pl.when(pl.col("has_t1") & (pl.col("t1_open_price") > eps))
        .then(pl.col(f"t1_price_{name}") / pl.col("t1_open_price") - 1.0)
        .otherwise(None)
        .alias(f"intra_{name}")
        for name in target_names
    ])

    output_cols = [f"raw_{n}" for n in target_names] + [f"intra_{n}" for n in target_names]
    return target_lf.select(["day", "sid"] + output_cols)


def extract_curves_from_panel(
    train_feat_lf: pl.LazyFrame,
    tune_config: dict,
    common_config: dict,
) -> pl.LazyFrame:
    """Build the tune_config-dependent curve column (lag_0) via downsampling and causal forward fill."""
    ds = tune_config["downsample"]
    total_bars = int(common_config.get("bars_per_day", TRADING_BARS_PER_DAY))
    bars_per_day = total_bars // ds
    feature_col = common_config.get("feature_col", "ofi_ratio")
    min_valid_ratio = float(common_config.get("min_curve_valid_ratio", 0.90))

    if ds > 1:
        train_feat_lf = train_feat_lf.filter((pl.col("bar_idx") % ds) == 0)
    train_feat_lf = align_date_col(train_feat_lf.select(["day", "sid", "bar_idx", feature_col]))

    return (
        train_feat_lf.group_by(["day", "sid"])
        .agg([
            pl.col(feature_col).count().alias("curve_len"),
            pl.col(feature_col).sort_by("bar_idx").forward_fill().alias("lag_0"),
        ])
        .filter(
            (pl.col("lag_0").list.len() == bars_per_day)
            & (pl.col("curve_len") >= max(1, int(round(bars_per_day * min_valid_ratio))))
        )
        .drop("curve_len")
    )


def build_fsm_panel(
    train_feat_lf: pl.LazyFrame,
    daily_lf: pl.LazyFrame,
    tune_config: dict,
    common_config: dict,
    is_train: bool = True,
) -> pl.LazyFrame:
    """Build full FSM panel by joining downsampled curves with the static target panel."""
    static_lf = build_static_panel(train_feat_lf, daily_lf, common_config, is_train)
    curve_lf = extract_curves_from_panel(train_feat_lf, tune_config, common_config)
    return curve_lf.join(static_lf, on=["day", "sid"], how="inner" if is_train else "left")


def build_oos_panel(
    train_paths: List[str],
    pre_oos_yms: List[int],
    dret_path: str,
    tune_config: dict,
    common_config: dict,
) -> Optional[pl.LazyFrame]:
    """Concatenate OOS feature parquets with macro warmup months and build the OOS evaluation panel."""
    ranking_window = common_config.get("ranking_window", 5)
    oos_with_warmup_paths = resolve_oos_paths_with_warmup(
        all_paths=train_paths,
        oos_yms=pre_oos_yms,
        ranking_window=ranking_window,
    )
    if not oos_with_warmup_paths:
        return None

    return build_fsm_panel(
        pl.concat([pl.scan_parquet(p) for p in oos_with_warmup_paths]),
        pl.scan_parquet(dret_path),
        tune_config,
        common_config,
        is_train=False,
    )
