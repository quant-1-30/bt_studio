import math
import numpy as np
import polars as pl

from bt_studio.pipeline.fsm.astc import *


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
    def __init__(self, config: dict, macro_dict: dict, gpd_dict: dict, quantiles: list):
        config["m"] = int(config["ndays"] * (240 // config["downsample"]))
        config["threshold_d"] = math.sqrt(2 * config["m"] * (1 - config["threshold_r"]))
        self.config = config 
        self.dtw_window = max(1, int(config["m"] * config["dtw_window_frac"]))
        
        self.macro_dict = macro_dict 
        self.gpd_dict = gpd_dict 
        self.quantiles = quantiles

    def validate(self, train_panel_df: pl.DataFrame, learned_motif, gpd_dict: dict, stats_window: list):
        result = evaluate_and_build_fsm(
            train_panel_df, learned_motif, self.config, 
            self.macro_dict, gpd_dict, self.quantiles, stats_window
        )
        return result

    def fit(self, train_panel_df: pl.DataFrame, tsc_v: np.ndarray, stats_window: list):
        """Ray Tune train FSM"""
        learned_motif = tsc_v[-1]

        if len(train_panel_df) == 0:
            return {"status": "failed", "reason": "无有效训练快照", "metrics_score": -np.inf}

        res = evaluate_and_build_fsm(
            train_panel_df, learned_motif, self.config, 
            self.macro_dict, self.gpd_dict, self.quantiles, stats_window
        )
        return res

    def __call__(self, oos_panel_df: pl.DataFrame, fsm_prior_matrix: np.ndarray, learned_motif: np.ndarray, top_k: int = 10):
        """
        Chronological Simulation
        """
        fsm = BayesianOnlineFSM(fsm_prior_matrix)
            
        scored_records =[]
        pending_pre_top_k =[] 
        trading_days = oos_panel_df["day"].unique().sort().to_list()
        z_motif_c = np.ascontiguousarray(robust_z_normalize(learned_motif), dtype=np.float64)

        for curr_date in trading_days:
            today_df = oos_panel_df.filter(pl.col("day") == curr_date)
            
            # =======================================================
            # 1. Today 14:55 to update Yesterday Macro State
            # =======================================================
            if pending_pre_top_k:
                today_rets = {row["sid"]: row["daily_ret"] for row in today_df.iter_rows(named=True)}
                edges, _ = self.gpd_dict.get(curr_date, (None, None))

                if edges is not None:
                    for trigger in pending_pre_top_k:
                        sid = trigger["sid"]
                        macro_yesterday = trigger["macro_state"]
                        
                        realized_ret = today_rets.get(sid, 0.0) 
                        fsm.update_posterior(macro_yesterday, realized_ret, edges)
            
            # =======================================================
            # 2. Signal Generation Today 14:55
            # =======================================================
            macro_state = self.macro_dict.get(curr_date, 1)
            _, centers = self.gpd_dict.get(curr_date, (None, None))
            today_candidates =[]
            
            for row in today_df.iter_rows(named=True):
                z_today = np.ascontiguousarray(robust_z_normalize(row["curve"]), dtype=np.float64)
                dist = dtw.distance_fast(z_today, z_motif_c, window=self.dtw_window, max_dist=self.config["threshold_d"])
                
                if dist < self.config["threshold_d"]:
                    score_z = fsm.predict_expected_return(macro_state, centers) if centers is not None else 0.0
                    score_ret = score_z * row["daily_vol"]
 
                    today_candidates.append({
                        "day": curr_date,
                        "sid": row["sid"],
                        "distance": float(dist),
                        "score": float(score_ret),
                        "macro_state": macro_state
                    })
                    # atr 
                    
            # =======================================================
            # 3. Cross-Sectional Rank & Top-K
            # =======================================================
            if today_candidates:
                today_candidates.sort(key=lambda x: x["score"], reverse=True)
                top_candidates = today_candidates[:top_k]
                scored_records.extend(top_candidates)
                
                pending_pre_top_k =[
                    {"sid": c["sid"], "macro_state": c["macro_state"]} 
                    for c in top_candidates
                ]
            else:
                pending_pre_top_k =[]
                
        return pl.DataFrame(scored_records)


def trainable(config: dict, hf_dfs: dict, daily_ret: pl.DataFrame, macro_dict: dict, gpd_dict: dict, run_params: dict): 
    
    config["m"] = int(config["ndays"] * (240 // config["downsample"]))
    config["threshold_d"] = math.sqrt(2 * config["m"] * (1 - config["threshold_r"]))

    # =========================================================
    # 🌟 stage 1 detect sample stumpy
    # =========================================================
    padded_arr = build_stumpy_from_chunk(hf_dfs, config, run_params["signal_type"])
    
    if len(padded_arr) < config["m"]:
        return {"status": "failed", "reason": "降采样后数据不足", "metrics_score": 0.0} # -np.inf

    tsc, tsc_v = get_atsc(padded_arr, config)
    
    if tsc_v is None or len(tsc_v) == 0:
        return {"status": "failed", "reason": "未找到有效 Motif", "metrics_score": 0.0}

    # =========================================================
    # 🌟 stage 2 process universe panel 
    # =========================================================
    
    panel_df = build_panel_from_chunk(hf_dfs, daily_ret, config, run_params["signal_type"])
    
    if len(panel_df) == 0:
        # return {"metrics_score": -np.inf}
        return {"metrics_score": 0.0}
    
    # =========================================================
    # 🌟 stage 3 calculate gpd
    # =========================================================
    
    model = MotifFSMModel(config, macro_dict, gpd_dict, run_params["quantiles"])
    res = model.fit(panel_df, tsc_v, run_params["stats_window"])

    del padded_arr, panel_df
    gc.collect()
    
    # if res.get("status") == "success":
    return {
        "metrics_score": res["metrics_score"],
        "learned_motif": res.get("learned_motif", []),
        "fsm_prior_matrix": res.get("fsm_prior_matrix",[])
    }
    # else:
    #     return {"metrics_score": -np.inf}
