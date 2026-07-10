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


# def build_ofi(aligned_lf: pl.LazyFrame) -> pl.LazyFrame:
#     """
#     高能、高信噪比的微观量价特征生成器 (100% 兼容原有接口)
#     """
#     # =========================================================================
#     # 💡 特征 1 重构：量价动量强度 (Price-Volume Power, 替代脆弱的 ADL OFI)
#     # 逻辑：价格波动的方向乘以成交量的对数。只有放量配合的趋势才被视为强形态，过滤无量空涨。
#     # =========================================================================
#     pv_power_expr = (
#         pl.col("close").pct_change().sign().fill_null(0) * 
#         (pl.col("amount") + 1.0).log()
#     )
    
#     # =========================================================================
#     # 💡 特征 2 重构：日内稳健波动率收缩 (Normalized High-Low Realized Volatility)
#     # 逻辑：使用高低价差对数，并去除绝对价格影响，用于捕捉主力拉升前夜的“波动率极度收窄”
#     # =========================================================================
#     volatility_expr = (
#         (pl.col("high") - pl.col("low")) / (pl.col("close") + 1e-8)
#     )

#     feat_lf = (
#         aligned_lf
#         .with_columns([
#             pl.col("minute_idx").alias("bar_idx"),
#             pv_power_expr.alias("raw_pv_power"),
#             volatility_expr.alias("raw_vol")
#         ])
#         .group_by(["day", "sid", "bar_idx"])
#         .agg([
#             pl.col("raw_pv_power").sum().alias("agg_pv"),
#             pl.col("raw_vol").sum().alias("volatility"), # 💡 保持原Volatility列名，零感替换
#             pl.col("amount").sum().alias("agg_amount"),
#             pl.col("close").last().alias("close") 
#         ])
#         .sort(["day", "sid", "bar_idx"])
#         .with_columns([
#             # 💡 将量价动力进行日内累积积分，形成完美的、具备物理趋势的连续曲线
#             pl.col("agg_pv").cum_sum().over(["day", "sid"]).alias("ofi_ratio") # 💡 保持原OFI列名
#         ])
#     )
#     return feat_lf

