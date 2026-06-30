
import os
import gc
import pickle
import joblib
import multiprocessing
import numpy as np
import polars as pl
from datetime import datetime
from dotenv import load_dotenv

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass 

# ==============================================================================
# C++ / OpenMP 线程锁死 (防止 Ray Worker 与 Polars/NumPy C引擎死锁)
# ==============================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'

import ray
import mlflow
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from ray.air.integrations.mlflow import MLflowLoggerCallback
# from prefect import flow, task, get_run_logger

from bt_studio.pipeline.features import build_ofi
from bt_studio.pipeline.patterns import build_fsm_panel, evaluate_and_build_fsm, discover_fsm_pattern
from bt_studio.pipeline.preprocess import prepare_macro, prepare_tick, universe_sample
from bt_studio.pipeline.inference import FSMPredictor


BASE_DIR = "/Users/hengxinliu/startup/bt_studio/result/fsm"
MODEL_DIR = f"{BASE_DIR}/models"
load_dotenv()

def get_latest_ckpt(target_year: int) -> str:
    if not os.path.exists(MODEL_DIR): return None
    valid_models = []
    for f in os.listdir(MODEL_DIR):
        if f.startswith("model_") and f.endswith(".pkl"):
            try:
                y = int(f.replace("model_", "").replace(".pkl", ""))
                if y <= target_year:
                    valid_models.append((y, os.path.join(MODEL_DIR, f)))
            except ValueError:
                continue
                
    if not valid_models: return None
    valid_models.sort(key=lambda x: x[0], reverse=True)
    return valid_models[0][1]

# ==============================================================================
# Node 1 Macro and Universe
# ==============================================================================
# @task(name="Node_Prepare_Daily_Universe") 
def node_prepare_daily_universe(exp_config: dict):
    rq = exp_config["run_params"]
    os.makedirs(BASE_DIR, exist_ok=True)
    
    universe_lf, daily_lf = prepare_macro(
        start_date=rq["start_date"], 
        end_date=rq["end_date"], 
        benchmark=rq["benchmark"].encode()
    )
    # sample universe 
    filtered_uni_lf = universe_sample(universe_lf, daily_lf, exceed=120, topk=0.80)
    filtered_uni_df = filtered_uni_lf.collect()
    
    valid_sids = filtered_uni_df["sid"].unique().to_list()

    dret_path = f"{BASE_DIR}/global_daily.parquet"
    filtered_uni_df.write_parquet(dret_path)
    return {"dret_path": dret_path, "universe_sids": valid_sids}


# ==============================================================================
# Node 2: Minute and Ofi
# ==============================================================================
# @task(name="Node_Extract_HF_Data") 
def node_extract_hf_data(year: int, sids: list[bytes], exp_config: dict):
    
    def _fetch_and_build(start_d: int, end_d: int, out_path: str):
        snapshot_dict = prepare_tick(start_date=start_d, end_date=end_d, sids=sids)
        
        lfs = []
        for sid, lf in snapshot_dict.items():
            lfs.append(build_ofi(lf))
            
        if lfs:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            pl.concat(lfs).collect().write_parquet(out_path)
            return [out_path]
        return []

    train_paths = _fetch_and_build((year - 1) * 10000 + 101, (year - 1) * 10000 + 1231, f"{BASE_DIR}/train/hf_{year-1}.parquet")
    oos_paths = _fetch_and_build(year * 10000 + 101, year * 10000 + 1231, f"{BASE_DIR}/oos/hf_{year}.parquet")
    return {"train_paths": train_paths, "oos_paths": oos_paths}


# ==============================================================================
# Node 3: OOS Decay
# ==============================================================================
# @task(name="Node_Check_Decay") 
def node_check_decay(year: int, dret_path: str, oos_paths: list, exp_config: dict):
    prev_model_path = get_latest_ckpt(year - 1)
    if not prev_model_path:
        return True 
        
    with open(prev_model_path, "rb") as f: 
        model_ckpt = pickle.load(f)

    aligned_lfs = [pl.scan_parquet(p) for p in oos_paths if os.path.exists(p)]
    if not aligned_lfs: return True

    # Panel Data
    oos_panel_df = build_fsm_panel(aligned_lfs, pl.scan_parquet(dret_path), model_ckpt["config"]).collect()
    cross_days = int(model_ckpt["config"]["cross_days"])
    curves_2d = np.hstack([np.vstack(oos_panel_df[f"lag_{i}"].to_list()) for i in reversed(range(cross_days))])
    
    rp = exp_config["run_params"]
    
    eval_res = evaluate_and_build_fsm(
        panel_df=oos_panel_df, curves_2d=curves_2d, motif=model_ckpt["motif"],
        config=model_ckpt["config"], stats_windows=rp["stats_windows"], alternative=rp["alternative"]
    )
   
    if eval_res.get("status") == "success" and eval_res["metrics_score"] > 0:
        # get_run_logger().info(f"✅ 历史 Motif 依然显著 (P-val: {eval_res['u_pval']:.4f})")
        return False 
    return True

# ==============================================================================
# Node 4: Ray Tune 
# ==============================================================================
def trainable_fsm_worker(config, hf_paths, dret_path, prior_config):
    """Ray 内部 Worker 函数"""
    aligned_lfs = [pl.scan_parquet(p) for p in hf_paths]
    panel_lf = build_fsm_panel(aligned_lfs, pl.scan_parquet(dret_path), config)
    
    result = discover_fsm_pattern(config, panel_lf, prior_config)
    if result["status"] == "success":
        tune.report({
            "metrics_score": result["metrics_score"], "u_pval": result["u_pval"],
            "learned_motif": result["learned_motif"], "fsm_network": result["fsm_network"]
        })
    else:
        tune.report({"metrics_score": 0.0, "u_pval": 1.0})


# @task(name="Node_Tune") 
def node_tune(year: int, dret_path: str, train_paths: list, exp_config: dict):
    ray.init(address="auto", ignore_reinit_error=True)

    rp, sb = exp_config["run_params"], exp_config["search_bounds"]
    
    search_space = {
        "downsample": tune.choice(sb["downsample"]), 
        "cross_days": tune.choice(sb["cross_days"]), 
        "motif_minutes": tune.choice(sb["motif_minutes"]), 
        "threshold_r": tune.uniform(*sb["threshold_r"]),
        "dtw_window_frac": tune.uniform(*sb["dtw_window_frac"]), 
        "z_abs_bound": tune.uniform(*sb["z_abs_bound"])
    }

    search_alg = OptunaSearch()
    prev_ckpt = get_latest_ckpt(year - 1)
    if prev_ckpt:
        prior_cfg = pickle.load(open(prev_ckpt, "rb"))["config"]
        search_alg = OptunaSearch(points_to_evaluate=[{k: prior_cfg[k] for k in search_space if k in prior_cfg}])

    wrapped_trainable = tune.with_resources(
        tune.with_parameters(trainable_fsm_worker, hf_paths=train_paths, dret_path=dret_path, prior_config=rp),
        resources={"cpu": 1, "gpu": 0} 
    )

    tuner = tune.Tuner(
        wrapped_trainable, param_space=search_space,   
        tune_config=tune.TuneConfig(
            metric="metrics_score", mode="max", search_alg=search_alg, num_samples=sb["num_trials"],            
            scheduler=tune.schedulers.ASHAScheduler(grace_period=sb["grace_period"], reduction_factor=sb["reduction_factor"]),
            max_concurrent_trials=sb["max_concurrent_trials"]
        ),
        run_config=tune.RunConfig(name=f"fsm_hpo_{year}", storage_path="/tmp/ray_results")
    )
    
    best_trial = tuner.fit().get_best_result("metrics_score", "max")
    if best_trial.metrics.get("metrics_score", 0.0) <= 0.0:
        return False 
 
    mlflow.set_experiment("FSM_Production_Models")
    with mlflow.start_run(run_name=f"FSM_{year}_v3"):
        mlflow.log_params(best_trial.config)
        mlflow.log_metric("train_score", best_trial.metrics["metrics_score"])
        
        model_ckpt = {
            "config": best_trial.config, 
            "motif": np.array(best_trial.metrics.get("learned_motif", [])), 
            "fsm_network": best_trial.metrics.get("fsm_network", {}),
            "valid_year": year
        }
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(f"{MODEL_DIR}/model_{year}.pkl", "wb") as f: pickle.dump(model_ckpt, f)
    
    gc.collect()
    return True

# ==============================================================================
# Node 5: OOS 
# ==============================================================================
# @task(name="Node_OOS_Inference") 
def node_oos_inference(year: int, dret_path: str, oos_paths: list, exp_config: dict):
    final_model_path = get_latest_ckpt(year)
    if not final_model_path: return
        
    with open(final_model_path, "rb") as f: model_ckpt = pickle.load(f)
    
    aligned_lfs = [pl.scan_parquet(p) for p in oos_paths if os.path.exists(p)]
    if not aligned_lfs: return
    
    oos_panel_lf = build_fsm_panel(aligned_lfs, pl.scan_parquet(dret_path), model_ckpt["config"])
    
    scored_df = FSMPredictor(model_ckpt).predict(oos_panel_lf)
    
    if scored_df.height > 0:
        out_path = f"{BASE_DIR}/scores/scores_{year}.parquet"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        scored_df.write_parquet(out_path)
        # get_run_logger().info(f"✅ {year} 年 OOS 生成打分: {scored_df.height} 条")


# ==============================================================================
# DAG (Walk-Forward)
# ==============================================================================
# @flow(name="WFO_FSM_Pipeline")
def wfo_pipeline(exp_config):
    # logger = get_run_logger()
    
    # Node 1
    global_data = node_prepare_daily_universe(exp_config)

    for y in range(2004, 2011):
        # logger.info(f"========== 🚀 {y} 年 Walk-Forward ==========")
        
        # Node 2
        paths = node_extract_hf_data(y, global_data["universe_sids"], exp_config)
        
        # Node 3
        if node_check_decay(y, global_data["dret_path"], paths["oos_paths"], exp_config):
            # logger.info(f"🔄 启动 {y-1} 年数据 Ray Tune 调优...")
            
            # Node 4
            if not node_tune(y, global_data["dret_path"], paths["train_paths"], exp_config): 
                continue
        
        # Node 5
        node_oos_inference(y, global_data["dret_path"], paths["oos_paths"], exp_config)


if __name__ == "__main__":


    exp_config = {
        "run_params": {
            "start_date": 20040101, "end_date": 20111231, "benchmark": "1A0001", 
            "vol_window": 20 , # used for vol in fut_ret_fw
            "top_k_ratio": 0.25, # used for sample
            "stats_windows": [1,2,3], # T+1 ---> T+3
            "alternative": "greater" # suited for a stock 
        },

        "search_bounds": {
            "downsample": [1, 3, 5], "cross_days": [1, 2, 3], "motif_minutes": [30, 60, 120, 240, 360], 
            "threshold_r": [0.70, 0.90], "dtw_window_frac": [0.05, 0.20], "z_abs_bound": [0.8, 1.5],
            "num_trials": 100, "grace_period": 5, "reduction_factor": 4, "max_concurrent_trials": 8
        }
    }

    wfo_pipeline(exp_config)
