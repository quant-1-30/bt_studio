import polars as pl
import numpy as np


def find_pareto_front(df_results: pl.DataFrame, common_config: dict) -> pl.DataFrame:
    """
        帕累托支配核心定义
            a. 在所有指标上都不比对方差
            b. 必须至少有一个指标严格优于对方
    """
    valid_df = df_results.drop_nulls(subset=["metrics_score"])
    if valid_df.height == 0: 
        return pl.DataFrame()
        
    # Complexity
    valid_df = valid_df.with_columns([
        (
            pl.max_horizontal(pl.col("config/cross_days"), 1) * # cross_day 0 ---> complexity 0
            ((pl.col("config/motif_minutes") / pl.col("config/downsample")) * common_config["dtw_window_frac"]) * 
            (1.0 - pl.col("config/threshold_r"))
        ).alias("complexity")
    ])
    
    scores = valid_df["metrics_score"].to_numpy()
    complexities = valid_df["complexity"].to_numpy()
    
    # broadcast ---> row control by column
    S_diff = scores[None, :] - scores[:, None] # eg [[10, 20]] - [[10], [20]]
    C_diff = complexities[None, :] - complexities[:, None] 
    
    dominates = (S_diff >= 0) & (C_diff <= 0) & ((S_diff > 0) | (C_diff < 0))
    is_dominated = dominates.any(axis=1)
    
    return valid_df.filter(~is_dominated)


def select_best_model_from_pareto(pareto_df: pl.DataFrame) -> dict: # Shift-and-Divide Utility
    if pareto_df.height == 0: 
        return None
    
    min_score = pareto_df["metrics_score"].min()
    
    return (
        # transfer to positive space 
        pareto_df.with_columns(
            ((pl.col("metrics_score") - min_score + 1.0) / pl.col("complexity")).alias("efficiency")
        )
        .sort("efficiency", descending=True)
        .row(0, named=True)
    )
