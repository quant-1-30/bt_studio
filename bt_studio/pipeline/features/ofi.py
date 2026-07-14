import polars as pl


def build_ofi(aligned_lf: pl.LazyFrame, common_config: dict) -> pl.LazyFrame:
    # downsample_m = common_config.get("downsample_m", 1)

    ofi_expr = (
        (pl.col("close") * 2 - pl.col("high") - pl.col("low")) / 
        (pl.col("high") - pl.col("low") + 1e-8)
    ) * pl.col("amount")
    
    feat_lf = (
        aligned_lf
        .with_columns([
            ofi_expr.alias("raw_ofi")
        ])
        .with_columns([
            pl.col("raw_ofi").cum_sum().over(["day", "sid"]).alias("cum_ofi"),
            pl.col("amount").cum_sum().over(["day", "sid"]).alias("cum_amount"),
            # ((pl.col("minute_idx") - 1) // downsample_m).cast(pl.Int32).alias("bar_idx")
            pl.col("minute_idx").alias("bar_idx")
        ])
        .with_columns([
            (pl.col("cum_ofi") / (pl.col("cum_amount") + 1e-8)).alias("ofi_ratio")
        ])
        .group_by(["day", "sid", "bar_idx"])
        .agg([
            pl.col("raw_ofi").sum().alias("agg_ofi"),
            pl.col("amount").sum().alias("agg_amount"),
            pl.col("close").last().alias("close"),
            pl.col("ofi_ratio").last().alias("ofi_ratio") 
        ])
        .sort(["day", "sid", "bar_idx"])
    )
    return feat_lf


# def build_impact_ofi(aligned_lf: pl.LazyFrame, common_config: dict) -> pl.LazyFrame:
#     return (
#         aligned_lf.sort(["day", "sid", "minute_idx"])
#         .with_columns([
#             pl.col("minute_idx").alias("bar_idx"),
#             pl.col("close").shift(1).over(["day", "sid"]).alias("prev_close")
#         ])
#         .with_columns(pl.col("prev_close").fill_null(pl.col("open")))
#         .with_columns([
#             (pl.col("close") - pl.col("prev_close")).sign().alias("direction"),
#             pl.col("amount").sqrt().alias("sqrt_amount")
#         ])
#         .with_columns([
#             (pl.col("direction") * pl.col("sqrt_amount")).alias("impact_step")
#         ])
#         .with_columns([
#             (
#                 pl.col("impact_step").cum_sum().over(["day", "sid"]) / 
#                 (pl.col("sqrt_amount").sum().over(["day", "sid"]) + 1e-8)
#             ).alias("daily_curve")
#         ])
#         .group_by(["day", "sid", "bar_idx"])
#         .agg([pl.col("daily_curve").last(), pl.col("close").last()])
#     )


# def build_liquidity_ofi(aligned_lf: pl.LazyFrame, common_config: dict) -> pl.LazyFrame:
#     """行为金融学：流动性吸收与羊群耗竭曲线"""
#     return (
#         aligned_lf.sort(["day", "sid", "minute_idx"])
#         .with_columns([
#             pl.col("minute_idx").alias("bar_idx"),
#             pl.col("close").shift(1).over(["day", "sid"]).alias("prev_close")
#         ])
#         .with_columns(pl.col("prev_close").fill_null(pl.col("open")))
#         .with_columns([
#             # 1. 真实日内波动幅度 (True Range percentage)
#             (
#                 pl.max_horizontal(pl.col("high"), pl.col("prev_close")) - 
#                 pl.min_horizontal(pl.col("low"), pl.col("prev_close"))
#             ) / (pl.col("prev_close") + 1e-8).alias("true_range_pct"),
#             # 2. 资金偏向：收盘价高于开盘价视为主动买盘承接
#             (pl.col("close") - pl.col("open")).sign().alias("absorption_dir")
#         ])
#         .with_columns([
#             # 3. 行为金融学核弹：高斯平滑惩罚 (Gaussian Volatility Penalty)
#             # 假设波动率超过 1% (0.01) 时，exp(-100 * 0.01) = 0.36，权重衰减；
#             # 若波动率为 0.1%，exp(-0.1) = 0.90，权重极大保留！
#             (
#                 pl.col("amount").log() * 
#                 (-100.0 * pl.col("true_range_pct")).exp() * 
#                 pl.col("absorption_dir")
#             ).alias("absorption_step")
#         ])
#         .with_columns([
#             # 4. 积分成 1D 曲线
#             pl.col("absorption_step").cum_sum().over(["day", "sid"]).alias("daily_curve")
#         ])
#         .group_by(["day", "sid", "bar_idx"])
#         .agg([pl.col("daily_curve").last(), pl.col("close").last()])
#     )


# def build_divergence_ofi(aligned_lf: pl.LazyFrame, common_config: dict) -> pl.LazyFrame:
#     """信息论：量价背离与动量点火曲线"""
#     return (
#         aligned_lf.sort(["day", "sid", "minute_idx"])
#         .with_columns([
#             pl.col("minute_idx").alias("bar_idx"),
#             # 1. 价格走势的累计收益率 (Price Trajectory)
#             (pl.col("close") / pl.col("open").first().over(["day", "sid"]) - 1.0).alias("cum_ret")
#         ])
#         .with_columns([
#             # 2. 对价格和成交量分别在当天进行横向 Z-Score 归一化 (提取相对强弱)
#             ((pl.col("cum_ret") - pl.col("cum_ret").mean().over(["day", "sid"])) / 
#              (pl.col("cum_ret").std().over(["day", "sid"]) + 1e-8)).alias("z_price"),
             
#             ((pl.col("amount") - pl.col("amount").mean().over(["day", "sid"])) / 
#              (pl.col("amount").std().over(["day", "sid"]) + 1e-8)).alias("z_volume")
#         ])
#         .with_columns([
#             # 3. 1D 坍缩：量价背离动量
#             # 价格 Z-score 与 成交量 Z-score 的乘积。
#             # 价量齐升为正，价跌量缩为正（健康趋势）；价升量缩为负（诱多背离）
#             (pl.col("z_price") * pl.col("z_volume")).alias("divergence_step")
#         ])
#         .with_columns([
#             # 4. 积分成曲线
#             pl.col("divergence_step").cum_sum().over(["day", "sid"]).alias("daily_curve")
#         ])
#         .group_by(["day", "sid", "bar_idx"])
#         .agg([pl.col("daily_curve").last(), pl.col("close").last()])
#     )

