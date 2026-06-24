import math
import numpy as np
import polars as pl
import scipy.stats as stats
from dtaidistance import dtw


class BayesianOnlineFSM:
    def __init__(self, prior_matrix):
        # Postperior Dirichlet
        # np.ones((num_macro_states, num_micro_bins))
        self.prior_matrix = prior_matrix if isinstance(prior_matrix, np.ndarray) else np.array(prior_matrix) 
        self.num_micro_bins = self.prior_matrix.shape[1]

    def update_posterior(self, macro_state, realized_ret, gpd_edges):
        real_bin = np.digitize(realized_ret, gpd_edges)
        real_bin = min(max(real_bin, 0), self.num_micro_bins - 1) # to avoid surpass
        self.prior_matrix[macro_state, real_bin] += 1.0

    def predict_expected_return(self, macro_state, gpd_centers):
        row_counts = self.prior_matrix[macro_state, :]
        probs = row_counts / np.sum(row_counts)
        return float(np.sum(probs * gpd_centers))

    def get_matrix(self):
        return self.prior_matrix.tolist()


class MotifFSMModel:
    def __init__(self, config: dict):
        """
        初始化模型配置，剔除外部字典依赖
        """
        # 自动计算序列长度 (m) 和 距离阈值 (threshold_d)
        if "ndays" in config:
            config["m"] = int(config["ndays"] * (240 // config["downsample"]))
        if "threshold_r" in config:
            # 根据皮尔逊相关系数(r)转换为欧氏距离(d)
            config["threshold_d"] = math.sqrt(2 * config["m"] * (1 - config["threshold_r"]))
            
        self.config = config
        self.dtw_window = max(1, int(config.get("m", 48) * config.get("dtw_window_frac", 0.1)))
        self.num_bins = config.get("num_bins", 5) # 默认分为 5 个收益率桶
        
    def build_stumpy_array(self, panel_df: pl.DataFrame) -> np.ndarray:
        """
        直接基于 Panel Data 提取 curve，并用 NaN 阻断拼接成 1D 数组
        """
        m = self.config["m"]
        nan_buffer = np.full(m, np.nan)
        padded_series = []
        
        # 将序列直接转为 numpy 并拼接
        curves = panel_df["curve"].to_list()
        for curve in curves:
            curve_arr = np.nan_to_num(np.array(curve, dtype=np.float64), nan=0.0)
            padded_series.append(curve_arr)
            padded_series.append(nan_buffer)
            
        return np.concatenate(padded_series) if padded_series else np.array([])

    def fit_and_evaluate(self, panel_df: pl.DataFrame, motif: np.ndarray, stats_windows: list) -> dict:
        """
        训练 FSM 状态矩阵并评估 OOS 表现
        """
        # =================================================================
        # 1. 动态生成 宏观状态 (Macro States) & 横截面收益分箱 (Return Bins)
        # =================================================================
        # 1.1 计算 Macro State: 统计横截面全市场每日 OFI 净流向并做三分位划分
        daily_macro = (
            panel_df.select(
                pl.col("day"),
                pl.col("curve").list.sum().alias("sid_ofi_sum")
            )
            .group_by("day")
            .agg(pl.col("sid_ofi_sum").mean().alias("daily_ofi_mean"))
        )
        
        p33 = daily_macro["daily_ofi_mean"].quantile(1/3)
        p67 = daily_macro["daily_ofi_mean"].quantile(2/3)
        
        daily_macro = daily_macro.with_columns(
            pl.when(pl.col("daily_ofi_mean") <= p33).then(0)
            .when(pl.col("daily_ofi_mean") <= p67).then(1)
            .otherwise(2).cast(pl.Int32).alias("macro_state")
        )
        
        panel_df = panel_df.join(daily_macro.select(["day", "macro_state"]), on="day", how="left")
        
        # 1.2 计算 Return Bins: 零基等分横截面分箱
        if "fwd_ret_1" in panel_df.columns:
            panel_df = (
                panel_df.with_columns([
                    pl.col("fwd_ret_1").rank(method="ordinal").over("day").alias("ret_rank"),
                    pl.col("fwd_ret_1").count().over("day").alias("ret_count")
                ])
                .with_columns([
                    (((pl.col("ret_rank") - 1) / pl.col("ret_count") * self.num_bins).cast(pl.Int32))
                    .clip(0, self.num_bins - 1).alias("ret_bin")
                ])
                .drop(["ret_rank", "ret_count"]) # 清理辅助列
            )

        # =================================================================
        # 2. 向量化 DTW 距离计算与触发器过滤
        # =================================================================
        curves = np.stack(panel_df["curve"].to_numpy())
        z_curves = robust_z_normalize(curves)
        z_motif = robust_z_normalize(motif)
        
        # 使用 dtaidistance 的 C 语言底层多线程加速 (放弃 list comprehension)
        distances = dtw.distance_matrix_fast(
            np.ascontiguousarray(z_curves, dtype=np.float64), 
            np.ascontiguousarray(np.array([z_motif]), dtype=np.float64), 
            window=self.dtw_window, 
            max_dist=self.config["threshold_d"],
            block=((0, len(z_curves)), (0, 1))
        ).flatten()
        
        panel_df = panel_df.with_columns(pl.Series("distance", distances))
        triggers = panel_df.filter(pl.col("distance") < self.config["threshold_d"])
        
        if triggers.height < 10:
            return {"status": "failed", "reason": "触发过少 < 10", "metrics_score": 0.0}

        # =================================================================
        # 3. 统计触发次数并构建 FSM 先验转移矩阵
        # =================================================================
        # 拉普拉斯平滑，起步填充 1.0
        fsm_prior_matrix = np.ones((3, self.num_bins), dtype=np.float64) 
        
        trigger_macros = triggers["macro_state"].to_numpy()
        trigger_bins = triggers["ret_bin"].to_numpy()
        
        for m_state, bin_idx in zip(trigger_macros, trigger_bins):
            if m_state is not None and bin_idx is not None:
                fsm_prior_matrix[m_state, bin_idx] += 1.0

        # =================================================================
        # 4. 统计检验 (KS & MW-U Test)
        # =================================================================
        ks_results = {}
        any_window_passed_soft = False 
        any_window_passed_hard = False 
        base_score = 100.0

        for fw in stats_windows:
            col_name = f"fwd_ret_{fw}"
            if col_name not in triggers.columns: continue
                
            cond_rets = triggers[col_name].drop_nulls().to_numpy()
            uncond_rets = panel_df[col_name].drop_nulls().to_numpy()
            
            if len(cond_rets) < 5 or np.std(cond_rets) < 1e-8:
                continue
                
            ks_stat, ks_pval = stats.ks_2samp(cond_rets, uncond_rets)
            cond_mean, uncond_mean = np.mean(cond_rets), np.mean(uncond_rets) 
            
            alt = 'greater' if cond_mean > uncond_mean else 'less'
            try:
                u_stat, u_pval = stats.mannwhitneyu(cond_rets, uncond_rets, alternative=alt)
            except ValueError:
                continue

            # 业务评分规则
            score = 0.0
            if u_pval <= 0.05:
                score = base_score
                any_window_passed_soft = True
                any_window_passed_hard = True
            elif u_pval <= 0.15:
                score = base_score * 0.1 
                any_window_passed_soft = True

            # 长度惩罚项
            penalty = 1.0 if self.config["m"] >= self.config.get("penalty_m", 24) else 0.5
            score *= penalty

            ks_results[f"T+{fw}"] = {
                "cond_mean": float(cond_mean),
                "uncond_mean": float(uncond_mean),
                "ks_pval": float(ks_pval),
                "u_pval": float(u_pval),
                "score": float(score)
            }

        if not any_window_passed_soft:
            return {
                "status": "failed", 
                "reason": "未达显著性软门槛(p>0.15)", 
                "metrics_score": 0.0,
                "passed_strict_alpha": False
            }

        raw_score = max([v["score"] for v in ks_results.values()]) if ks_results else 0.0

        return {
            "status": "success",
            "passed_strict_alpha": any_window_passed_hard, 
            "fsm_trigger_count": len(triggers), 
            "ks_results": ks_results,
            "fsm_prior_matrix": fsm_prior_matrix.tolist(), # 直接可 JSON 序列化输出
            "learned_motif": motif.tolist(),
            "metrics_score": raw_score   
        }


def trainable(config: dict, hf_dfs: dict, daily_ret: pl.DataFrame, macro_dict: dict, run_params: dict):
    """
    Ray 调参的无缝入口
    """
    model = MotifFSMModel(config, macro_dict)
    
    # Stage 1: STUMPY 提取形态
    padded_arr = model.build_stumpy_array(hf_dfs)
    if len(padded_arr) < config["m"]:
        return {"status": "failed", "metrics_score": 0.0}
        
    tsc, tsc_v = get_atsc(padded_arr, config)
    if tsc_v is None or len(tsc_v) == 0:
        return {"status": "failed", "metrics_score": 0.0}
        
    # Stage 2: 提取个股快照面板 (Lazy 拼接)
    panel_df = build_panel_from_chunk(hf_dfs, daily_ret, config, run_params["signal_type"])
    if len(panel_df) == 0:
        return {"metrics_score": 0.0}
        
    # Stage 3: 训练与状态机生成
    res = model.fit_and_evaluate(panel_df, tsc_v, run_params["stats_window"])
    
    # 🌟 垃圾回收，防内存爆炸
    del padded_arr, panel_df
    gc.collect()
    
    return {
        "metrics_score": res.get("metrics_score", 0.0),
        "learned_motif": res.get("learned_motif", []).tolist() if "learned_motif" in res else [],
        "fsm_prior_matrix": res.get("fsm_prior_matrix", np.array([])).tolist() if "fsm_prior_matrix" in res else []
    }
