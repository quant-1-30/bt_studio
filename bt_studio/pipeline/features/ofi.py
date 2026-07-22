import polars as pl
import numpy as np


# Cross-Sectional Demean Only (preserves time-series shape for STUMPY/DTW)
def demean_expr(col_name: str) -> pl.Expr:
    median = pl.col(col_name).median().over(["day", "minute_idx"])
    return pl.col(col_name) - median


# MAD Normalize (used for correlation/weight calculation only)
def robust_zscore_expr(col_name: str) -> pl.Expr:
    median = pl.col(col_name).median().over(["day", "minute_idx"])
    mad = (pl.col(col_name) - median).abs().median().over(["day", "minute_idx"])
    robust_scale = 1.4826 * mad + 1e-6
    return (pl.col(col_name) - median) / robust_scale


def build_ofi(aligned_lf: pl.LazyFrame, common_config: dict) -> pl.LazyFrame:
    eps = common_config["eps"]
    min_w = common_config["min_factor_weight"]   
    
    sorted_lf = aligned_lf.sort(["day", "sid", "minute_idx"])
    
    step1_lf = (
        aligned_lf.sort(["day", "sid", "minute_idx"])
        .with_columns([
            # (pl.col("close") - pl.col("close").over(["day", "sid"]).shift(1)).alias("close_diff"), 
            (pl.col("close") - pl.col("close").shift(1).over(["day", "sid"])).alias("close_diff"),
            ((pl.col("high") - pl.col("low")) / (pl.col("close") + eps)).alias("pct")
        ])
        .with_columns([
            pl.col("close_diff").fill_null(
                pl.col("close") - pl.col("open") # # if alpha else pl.lit(0.0)
            ) 
        ])
        .with_columns([
            pl.col("close_diff").sign().cast(pl.Int8).alias("raw_dir")
        ])
        .with_columns([
            (
                pl.when(pl.col("raw_dir") != 0)
                .then(pl.col("raw_dir"))
                .otherwise(None)
            )
            .forward_fill().over(["day", "sid"]) 
            .fill_null(0) 
            .alias("direction")
        ])
    )

    
    step2_lf = (
        step1_lf
        .with_columns([
            # Signed Amount (SA)
            (pl.col("direction") * pl.col("amount")).alias("sa_step"),
            
            # Impact (IMP) 
            (pl.col("direction") * pl.col("amount").sqrt()).alias("impact_step"),
            pl.col("amount").sqrt().alias("sqrt_amount"),
            
            # Liquidity (LIQ) 
            (pl.col("direction") * pl.col("amount") / (pl.col("pct") + eps)).alias("liquidity_step"),
            (pl.col("amount") / (pl.col("pct") + eps)).alias("liquidity_scale")
        ])
    )
    
    step3_lf = (
        step2_lf
        .with_columns([
            pl.col("sa_step").cum_sum().over(["day", "sid"]).alias("cum_sa"),
            pl.col("amount").cum_sum().over(["day", "sid"]).alias("cum_amount"),
            
            pl.col("impact_step").cum_sum().over(["day", "sid"]).alias("cum_impact"),
            pl.col("sqrt_amount").cum_sum().over(["day", "sid"]).alias("cum_sqrt_amount"),
            
            pl.col("liquidity_step").cum_sum().over(["day", "sid"]).alias("cum_liquidity"),
            pl.col("liquidity_scale").cum_sum().over(["day", "sid"]).alias("cum_liquidity_scale")
        ])
    )
    
    # 5. scale to [-1, 1]
    step4_lf = (
        step3_lf
        .with_columns([
            (pl.col("cum_sa") / (pl.col("cum_amount") + eps)).alias("sa_ratio"),
            (pl.col("cum_impact") / (pl.col("cum_sqrt_amount") + eps)).alias("impact_ratio"),
            (pl.col("cum_liquidity") / (pl.col("cum_liquidity_scale") + eps)).alias("liquidity_ratio")
        ])
    )
    
    # MDP
    # z-scored ratios: 仅用于相关性/权重计算（数学上需要标准化）
    # demeaned ratios: 用于信号合成（保留时序几何形状，避免时变 MAD 扭曲）
    step5_lf = (
        step4_lf
        .with_columns([
            robust_zscore_expr("sa_ratio").alias("sa_z_tmp"),
            robust_zscore_expr("impact_ratio").alias("imp_z_tmp"),
            robust_zscore_expr("liquidity_ratio").alias("liq_z_tmp"),
            demean_expr("sa_ratio").alias("sa_demean"),
            demean_expr("impact_ratio").alias("imp_demean"),
            demean_expr("liquidity_ratio").alias("liq_demean")
        ])
        .with_columns([
            # E(X)=0, E(Y)=0,Cov(X,Y) = E(XY)
            (pl.col("sa_z_tmp") * pl.col("imp_z_tmp")).mean().over(["day", "minute_idx"]).alias("rho_sa_imp"),
            (pl.col("sa_z_tmp") * pl.col("liq_z_tmp")).mean().over(["day", "minute_idx"]).alias("rho_sa_liq"),
            (pl.col("imp_z_tmp") * pl.col("liq_z_tmp")).mean().over(["day", "minute_idx"]).alias("rho_imp_liq")
        ])
        .with_columns([
            (
                1.0 - pl.col("rho_imp_liq")**2 - pl.col("rho_sa_imp") - pl.col("rho_sa_liq") + 
                pl.col("rho_sa_imp") * pl.col("rho_imp_liq") + pl.col("rho_sa_liq") * pl.col("rho_imp_liq")
            ).alias("w_sa_raw"),
            
            (
                1.0 - pl.col("rho_sa_liq")**2 - pl.col("rho_sa_imp") - pl.col("rho_imp_liq") + 
                pl.col("rho_sa_imp") * pl.col("rho_sa_liq") + pl.col("rho_sa_liq") * pl.col("rho_imp_liq")
            ).alias("w_imp_raw"),
            
            (
                1.0 - pl.col("rho_sa_imp")**2 - pl.col("rho_sa_liq") - pl.col("rho_imp_liq") + 
                pl.col("rho_sa_imp") * pl.col("rho_sa_liq") + pl.col("rho_sa_imp") * pl.col("rho_imp_liq")
            ).alias("w_liq_raw")
        ])
        .with_columns([
            pl.max_horizontal(pl.col("w_sa_raw"), min_w).alias("w_sa_pos"),
            pl.max_horizontal(pl.col("w_imp_raw"), min_w).alias("w_imp_pos"),
            pl.max_horizontal(pl.col("w_liq_raw"), min_w).alias("w_liq_pos")
        ])
        .with_columns([
            (pl.col("w_sa_pos") + pl.col("w_imp_pos") + pl.col("w_liq_pos")).alias("w_sum")
        ])
        .with_columns([
            (pl.col("w_sa_pos") / pl.col("w_sum")).alias("w_sa"),
            (pl.col("w_imp_pos") / pl.col("w_sum")).alias("w_imp"),
            (pl.col("w_liq_pos") / pl.col("w_sum")).alias("w_liq")
        ])
        .with_columns([
            # 使用 demeaned ratios 合成信号（保留时序形状，不除以时变 MAD）
            (
                pl.col("w_sa") * pl.col("sa_demean") + 
                pl.col("w_imp") * pl.col("imp_demean") + 
                pl.col("w_liq") * pl.col("liq_demean")
            ).alias("raw_score")
        ])
        .select(["day", "sid", "minute_idx","open", "close", "raw_score"])
    )

    final_lf = (
        step5_lf
        .with_columns([
            # 仅去均值（去市场共移），不除以时变 MAD，不做 tanh 压缩
            # STUMPY 内部会对子序列做 z-normalization，无需预先压缩形状
            demean_expr("raw_score").alias("ofi_ratio")
        ])
        .rename({"minute_idx": "bar_idx"})
        .select(["day", "sid", "bar_idx", "open", "close", "ofi_ratio"])
        .sort(["day", "sid", "bar_idx"])
    )
    return final_lf