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
            ((pl.col("config/motif_minutes") / pl.col("config/downsample")) * common_config["dtw_window_frac"]) * 
            (1.0 - pl.col("config/threshold_r"))
        ).alias("complexity")
    ])
    
    # scores = valid_df["metrics_score"].to_numpy()
    # complexities = valid_df["complexity"].to_numpy()
    min_score = valid_df["metrics_score"].min()
    valid_df = valid_df.with_columns(
        ((pl.col("metrics_score") - min_score + 1.0) / pl.col("complexity")).alias("efficiency")
    )
    
    efficiency = valid_df["efficiency"].to_numpy()
    density = valid_df["valid_sample_ratio"].to_numpy() 
    autocorr = valid_df["autocorr"].to_numpy()          
    
    # row control by column ---> colj - rowi
    Eff_diff = eff[None, :] - eff[:, None] 
    Den_diff = density[None, :] - density[:, None]
    Auto_diff = autocorr[None, :] - autocorr[:, None]
    
    dominates = (
        (Eff_diff >= 0) & (Den_diff >= 0) & (Auto_diff >= 0) & 
        ((Eff_diff > 0) | (Den_diff > 0) | (Auto_diff > 0))
    )
    
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


def select_best_model_from_pareto(pareto_df: pl.DataFrame) -> dict | None:
    if pareto_df.height == 0: 
        return None
    
    # =========================================================================
    # avoid score stuck in zero or negative
    # =========================================================================
    utility_lf = (
        pareto_df.lazy()
        .with_columns(
            (
                pl.col("efficiency") * 
                (pl.col("autocorr") + 1.1) * # [-1, 1] ---> [0.1, 2.1]
                (pl.col("valid_sample_ratio") + 0.1) # [0, 1] ---> [0.1, 1.1]
            ).alias("final_utility")
        )
        .sort("final_utility", descending=True)
        .limit(1) # heap
    )
    
    final_best_df = utility_lf.collect()
    if final_best_df.height == 0:
        return None
        
    return final_best_df.to_dicts()[0] # .row(n, named=True) heavy ops
