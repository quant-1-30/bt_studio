# ==============================================================================
# C++  OpenMP Ray CPU
# ==============================================================================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'

try:
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  

import gc
import math
import ray
import pickle
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pandas as pd
from ray import tune
from typing import Union, List
from workflow.astc import *
from workflow.preprocess import *
from workflow.util import consume_time


# annual 8% vol ---> daily vol 0.5% vol * sqrt(252)
THRESVOL = 0.005 


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
        config["m"] = int(config["ndays"] * np.floor(240 / config["downsample"]))
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

    def fit(self, padded_array: np.ndarray, train_panel_df: pl.DataFrame, stats_window: list):
        """Ray Tune train FSM"""

        if len(padded_array) < self.config["m"]:
            return {"status": "failed", "reason": "降采样后数据不足", "metrics_score": -np.inf}
            
        tsc, tsc_v = get_atsc(padded_array, self.config)
        if tsc_v is None or len(tsc_v) == 0:
            return {"status": "failed", "reason": "未找到有效 Motif", "metrics_score": -np.inf}

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


def trainable(config, chunks_meta, chunks_ref, daily_ret, macro_dict, gpd_dict, quantiles, samples, train_end, stats_window, signal_type): 
    
    config["m"] = int(config["ndays"] * np.floor(4 * 60 / config["downsample"]))
    config["threshold_d"] = math.sqrt(2 * config["m"] * (1 - config["threshold_r"]))

    # =========================================================
    # 🌟 stage 1 detect sample stumpy
    # =========================================================
    padded_arr = build_stumpy_from_chunk(chunks_ref, samples, config, signal_type)
    
    if len(padded_arr) < config["m"]:
        return {"metrics_score": -np.inf}

    tsc, tsc_v = get_atsc(padded_arr, config)
    
    if tsc_v is None or len(tsc_v) == 0:
        return {"metrics_score": -np.inf} 

    # =========================================================
    # 🌟 stage 2 process universe panel 
    # =========================================================
    
    panel_df = build_panel_from_chunk(chunks_meta, chunks_ref, daily_ret, config, signal_type)
    
    if len(panel_df) == 0:
        return {"metrics_score": -np.inf}
    
    # =========================================================
    # 🌟 stage 3 calculate gpd
    # =========================================================

    train_panel = panel_df.filter(pl.col("day") <= train_end)
    
    model = MotifFSMModel(config, macro_dict, gpd_dict, quantiles)
    res = model.fit(padded_arr, train_panel, stats_window)

    del padded_arr, panel_df, train_panel
    gc.collect()
    
    if res.get("status") == "success":
        return {
            "metrics_score": res["metrics_score"],
            "learned_motif": res.get("learned_motif", []),
            "fsm_prior_matrix": res.get("fsm_prior_matrix",[])
        }
    else:
        return {"metrics_score": -np.inf}


@consume_time
def run_wfo_pipeline(start_year, 
                     end_year, 
                     benchmark, 
                     market, 
                     max_concurrency=1, 
                     num_samples=200, 
                     quantiles=[0.1, 0.3, 0.7, 0.9],
                     loopback=504,
                     freq_month=6, 
                     stats_window=[5, 10, 20], 
                     signal_type="vwap"):

    # ========================================================= Preprocess =========================================================
     
    md_api, universe, samples = prepare_universe(start_year * 10000, end_year * 10000, market)
    
    global_start_date = (start_year - 2) * 10000 + 101
    global_end_date = end_year * 10000 + 1231
    
    global_daily_ret, global_macro_dict = prepare_daily(
        md_api, universe, benchmark, global_start_date, global_end_date, stats_window, THRESVOL
    )
    
    # GPD not dependant on hpo
    # "loopback": tune.choice([21, 63, 126, 252]),
    # "gpd_quantiles": tune.choice([[0.10, 0.30, 0.70, 0.90], [0.20, 0.40, 0.60, 0.80], [0.05, 0.25, 0.75, 0.95]]),
    # "gpd_freq_month": tune.choice([3, 6]), # update frequency 
    global_gpd_dict = build_rolling_gpd(
        global_daily_ret, quantiles=quantiles, loopback=loopback, freq_month=freq_month
    )
    print("global state calculate finish")

    last_valid_state = {
        "config": None,
        "motif": None,
        "fsm_matrix": None,
        "valid_year": None 
    }
 
    for _year in range(start_year, end_year):
        
        train_start = (_year - 1) * 10000 + 101 
        train_end = (_year - 1) * 10000 + 1231 

        oos_start = _year * 10000 + 101
        oos_end = _year * 10000 + 1231

        print(f"\n{'='*50}\n🚀 WFO: 训练期 [{train_start}-{train_end}] -> 推理期[{_year}]\n{'='*50}")

        train_chunks_meta, train_chunks_ref = prepare_ray_chunks(
            md_api, universe, train_start, train_end, signal_type
        ) 

        retune = True

        # ================================================================ Evaluation  ======================================================  

        if last_valid_state["config"] is not None:
            print(f" test motif is valid on {train_start}-{train_end}")

            config = last_valid_state["config"]
            
            train_panel = build_panel_from_chunk(train_chunks_meta, train_chunks_ref, global_daily_ret, config, signal_type)
            train_eval = train_panel.filter(pl.col("day") >= train_start)
            
            valid_model = MotifFSMModel(config, global_macro_dict, global_gpd_dict, quantiles)
            eval_res = valid_model.validate(train_eval, last_valid_state["motif"], global_gpd_dict, stats_window)
            
            if eval_res.get("status") == "success" and eval_res["metrics_score"] > 0:
                print(f"✅ 模型依然坚挺 (Score: {eval_res['metrics_score']:.2f})，跳过 Ray Tune")
                retune = False
            else:
                print("❌ 模型发生 Alpha 衰减，启动超参重搜！")
            
        # ================================================================ Ray Tune ======================================================  

        if retune:
            print(f" Ray tune on train_df")

            search_space = {
                "downsample": tune.choice([15, 20, 30, 60]), 
                "ndays": tune.choice([1, 2, 3]), 

                # dtw / linalg_norm
                "threshold_r": tune.uniform(0.70, 0.95),
                "dtw_window_frac": tune.choice([0.05, 0.10, 0.15, 0.20]), 

                # metrics
                "penalty_m": tune.choice([20, 30, 40]), 

            }

            # --- 配置 ASHA 算法 (早停) 避免score -np.inf ---
            # metric: 优化目标, mode:最大化 / grace_period: 至少跑多久才开始判断 / reduction_factor: 每轮淘汰比例
            asha_scheduler = tune.schedulers.ASHAScheduler( 
                metric="metrics_score", # target
                mode="max", 
                grace_period=1, # util when to criterior
                reduction_factor=4 # 
            )
            
            wrapped_trainable = tune.with_resources( # auto put in plasma and get
                tune.with_parameters(
                    trainable,
                    chunks_meta=train_chunks_meta,
                    chunks_ref=train_chunks_ref,
                    daily_ret=global_daily_ret,
                    macro_dict=global_macro_dict,
                    gpd_dict=global_gpd_dict,
                    quantiles=quantiles,
                    samples=samples,
                    train_end=train_end,
                    stats_window=stats_window,
                    signal_type=signal_type 
                ),
                resources={"cpu": 1, "gpu": 0} 
            )
            tuner = tune.Tuner( 
                wrapped_trainable,
                param_space=search_space,   
                tune_config=tune.TuneConfig(
                    # search_alg = HyperOptSearch(),
                    num_samples=num_samples,            
                    max_concurrent_trials=max_concurrency,    
                    scheduler=asha_scheduler
                ),
                run_config=tune.RunConfig(
                    name="my_strategy_hpo",
                    storage_path="/tmp/ray_tune_results",
                ),
            )
            results = tuner.fit() 
            # search_alg.save("./my-checkpoint.pkl")
            # search_alg.restore("./my-checkpoint.pkl")
            best_trial = results.get_best_result("metrics_score", "max")
            
            if best_trial.metrics.get("metrics_score", -np.inf) == -np.inf:
                print(f"⚠️ {_year-1} Skip to next year")
                last_valid_state = {"config": None, "motif": None, "fsm_matrix": None, "valid_year": None}
                continue
                
            last_valid_state["config"] = best_trial.config
            last_valid_state["motif"] = np.array(best_trial.metrics["learned_motif"])
            last_valid_state["fsm_matrix"] = np.array(best_trial.metrics["fsm_prior_matrix"])
            last_valid_state["valid_year"] = _year - 1

            # save to model repo
            model_checkpoint = {
                "last_valid_state": last_valid_state,
                "train_start": train_start,
                "train_end": train_end
            }

            model_dir = f"checkpoint/fsm"
            os.makedirs(model_dir, exist_ok=True)
            with open(f"{model_dir}/model_{_year}.pkl", "wb") as f:
                pickle.dump(model_checkpoint, f)
            print(f"{model_dir}/model_{_year}.pkl already saved")
        
        # Ray Plasma Store Release or OOM  
        ray.internal.free(train_chunks_ref)
        del train_chunks_meta, train_chunks_ref
        gc.collect()

        # ========================================================= OSS Predict ========================================================= 
        print(f"📈 开始执行 {_year} 年的 OOS ")

        current_config = last_valid_state["config"]
        current_motif = last_valid_state["motif"]
        current_fsm = last_valid_state["fsm_matrix"]

        oos_chunks_meta, oos_chunks_ref = prepare_ray_chunks(
            md_api, universe, oos_start, oos_end, signal_type
        )
        
        oos_panel = build_panel_from_chunk(oos_chunks_meta, oos_chunks_ref, global_daily_ret, current_config, signal_type)
        
        if len(oos_panel) == 0:
            print(f"⚠️ OOS 阶段数据提取为空")
            continue
            
        oos_panel = oos_panel.filter(pl.col("day") >= oos_start)
        infer_model = MotifFSMModel(current_config, global_macro_dict, global_gpd_dict, quantiles)
        scored_df = infer_model(oos_panel, current_fsm, current_motif, top_k=10)
            
        if len(scored_df) > 0:
            output_path = f"data/fsm"
            os.makedirs(output_path, exist_ok=True)
            scored_df.write_parquet(f"{output_path}/scores_{_year}.parquet")
            print(f"✅ {_year} 年产出高质量信号: {len(scored_df)}")
        else:
            print(f"⚠️ {_year} 年未产出任何符合阈值的交易信号")
            
        # Release Memory
        ray.internal.free(oos_chunks_ref)
        del oos_chunks_meta, oos_chunks_ref, oos_panel
        gc.collect()


if __name__ == "__main__":

    env_vars = { 
      "OMP_NUM_THREADS": "1",
      "MKL_NUM_THREADS": "1",
      "OPENBLAS_NUM_THREADS": "1",
      "VECLIB_MAXIMUM_THREADS": "1",
      "NUMEXPR_NUM_THREADS": "1"
    }
    ray.init(address="auto", 
             namespace="backtest", 
             runtime_env={"env_vars": env_vars}, # 
             ignore_reinit_error=True,
    )

    start_date=2010
    end_date=2026
    benchmark=b"000001"
    market = "6"
    max_concurrency = 10 
    run_wfo_pipeline(start_date, end_date, benchmark, market, max_concurrency)

    # try:
    #     run_wfo_pipeline(start_date, end_date, benchmark, market)
    # except KeyboardInterrupt:
    #     print("\n⏹️ 任务被手动终止")
    # finally:
    #     print("清理集群资源...")
    #     ray.shutdown()
