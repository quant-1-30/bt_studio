import os
import multiprocessing

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass 

# ==============================================================================
# C++  OpenMP Ray CPU
# ==============================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'

# gRPC spawn
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0' 
# os.environ['GRPC_VERBOSITY'] = 'DEBUG'

import ray
import joblib
import mlflow
import numpy as np
import polars as pl
from datetime import datetime
from dotenv import load_dotenv

from ray import tune
from ray.tune.search.optuna import OptunaSearch

from ray.air.integrations.mlflow import MLflowLoggerCallback

from prefect import flow, task, get_run_logger

from bt_studio.pipeline.fsm.preprocess import prepare_macro, prepare_chunks, process_to_residuals, build_rolling_gpd
from bt_studio.pipeline.fsm.tune import trainable, MotifFSMModel
from bt_studio.pipeline.fsm.astc import build_panel_from_chunk


BASE_DIR = "/Users/hengxinliu/startup/bt_studio/result/fsm"
MODEL_DIR = f"{BASE_DIR}/models"

load_dotenv()


def get_latest_ckpt(target_year: int) -> str:

    if not os.path.exists(MODEL_DIR):
        return None
    
    valid_models = []
    for f in os.listdir(MODEL_DIR):
        if f.startswith("model_") and f.endswith(".pkl"):
            try:
                y = int(f.replace("model_", "").replace(".pkl", ""))
                if y <= target_year:
                    valid_models.append((y, os.path.join(MODEL_DIR, f)))
            except ValueError:
                continue
                
    if not valid_models:
        return None
        
    valid_models.sort(key=lambda x: x[0], reverse=True)
    latest_year, latest_path = valid_models[0]
    print(f"🔍 Dynamic Lookup (Target <= {target_year}): Found model from year {latest_year}")
    return latest_path


# ========================================================
# macro data
# ========================================================
@task(name="Get_Macro_Data", cache_key_fn=None) 
def get_macro(run_config: dict):
    
    rq = run_config["run_params"]
    # out_path = "/data/fsm/global_daily.parquet"
    os.makedirs(BASE_DIR, exist_ok=True)
    universe, global_macro_dict, global_daily_ret = prepare_macro(
        rq["start_date"], rq["end_date"], rq["benchmark"], rq["stats_window"], loopback=rq["loopback"])

    dret_path = f"{BASE_DIR}/global_dret.parquet"
    global_daily_ret.write_parquet(dret_path)
    macro_path = f"{BASE_DIR}/macro_dict.bz2" 
    with open(macro_path, "wb") as f:
        joblib.dump({
                "universe": universe,
                "macro_dict": global_macro_dict,
        }, f, compress=3)
        
    return {"dret_path": dret_path, "macro_path": macro_path}
# ========================================================
# gpb ret
# ========================================================
@task(name="Compute_Global_gpd") 
def compute_global_gpd(dret_path:str, config: dict):
    
    rp = config["run_params"]
    global_daily_ret = pl.read_parquet(dret_path)
    
    gpd_dict = build_rolling_gpd(
        global_daily_ret, quantiles=rp["quantiles"], loopback=rp["loopback"], freq_month=rp["freq_month"]
    )
    
    gpd_path = f"{BASE_DIR}/gpd_macro.bz2"
    with open(gpd_path, "wb") as f:
        joblib.dump(gpd_dict, f, compress=3)
    return gpd_path


@task(name="Extract_hf_data") 
def extract_hf_data(year: int, macro_path: str, config: dict):
    
    rp = config["run_params"]
    macro_path = macro_path  #xcom ---> str
    macro_data = joblib.load(macro_path)
    universe = macro_data["universe"]
    
    # Train
    train_start = (year - 1) * 10000 + 101
    train_end = (year - 1) * 10000 + 1231
    
    chunk_df = prepare_chunks(universe, train_start, train_end, adj=1)
    hf_dfs = process_to_residuals(chunk_df, rp["signal_type"])
    
    train_hf_path = f"{BASE_DIR}/train/hf_dfs_{year-1}.bz2"
    os.makedirs(os.path.dirname(train_hf_path), exist_ok=True)
    with open(train_hf_path, "wb") as f:
        joblib.dump(hf_dfs, f, compress=3)
    
    # Valid and OOS
    oos_start = year * 10000 + 101
    oos_end = year * 10000 + 1231
    oos_chunk = prepare_chunks(universe, oos_start, oos_end, adj=1)
    oos_hf_dfs = process_to_residuals(oos_chunk, rp["signal_type"])
    
    oos_hf_path = f"{BASE_DIR}/oos/hf_dfs_{year}.bz2"
    os.makedirs(os.path.dirname(oos_hf_path), exist_ok=True)
    with open(oos_hf_path, "wb") as f:
        joblib.dump(oos_hf_dfs, f, compress=3)
    return {"train": train_hf_path, "oos": oos_hf_path}


@task(name="Check_decay_node") 
def check_decay_node(year: int, g_paths: dict, gpd_path: str, oos_path: str, exp_config: dict):
    
    prev_model_path = get_latest_ckpt(year - 1)
    if not prev_model_path:
        print("⚠️ 无历史模型冷启动搜索")
        return f"WFO_Loop_{year}.run_ray_tune"
    # macro_path = g_paths["macro_path"]
    macor_data = joblib.load(g_paths["macro_path"]) # auto mmap
    
    # dret_path = g_paths["dret_path"]
    global_daily_ret = joblib.load(g_paths["dret_path"])
        
    gpd_dict = joblib.load(gpd_path)
    
    # oos_path = hf_paths["oos"]
    hf_dfs = joblib.load(oos_path)
    with open(prev_model_path, "rb") as f: 
        model_config = pickle.load()

    rp = exp_config["run_params"]
    valid_panel = build_panel_from_chunk(hf_dfs, global_daily_ret, model_config["config"], rp["signal_type"])
    
    valid_model = MotifFSMModel(model_config["config"], macro_data["macro_dict"], gpd_dict, rp["quantiles"])
    eval_res = valid_model.validate(valid_panel, model_config["motif"], gpd_dict, rp["stats_window"])
   
    if eval_res.get("status") == "success" and eval_res["metrics_score"] > 0:
        print(f"✅ 模型表现优异 (Score: {eval_res['metrics_score']:.2f})")
        return False
    else:
        print(f"❌ Alpha 衰减，重新搜索！")
        return True


@task(name="Tune_Node") 
def tune_node(year: int, g_paths: dict, gpd_path: str, hf_train_path: str, exp_config: dict):

    ray.init(address="auto", ignore_reinit_error=True)
    # =================================================================
    # Default Prior To Historical Prior 
    # =================================================================
    prev_model_path = get_latest_ckpt(year - 1)

    # load Config 
    rp = exp_config["run_params"]
    search_bounds = exp_config["search_bounds"]

    # load Ray Train
    macro_data = joblib.load(g_paths["macro_path"])
    daily_ret = pl.read_parquet(g_paths["dret_path"])
    gpd_dict = joblib.load(gpd_path)
    hf_dfs = joblib.load(hf_train_path)

    if prev_model_path:
        with open(prev_model_path, "rb") as f: 
            prior_config = pickle.load(f)["config"]
        prior_point = {k: prior_config[k] for k in search_bounds.keys() if k in prior_config}
        print(f"prior : {prior_point}")
        search_alg = OptunaSearch(points_to_evaluate=[prior_point])
    else:
        search_alg = OptunaSearch() 
    
    search_space = {
        "downsample": tune.choice(search_bounds["downsample"]), 
        "ndays": tune.choice(search_bounds["ndays"]), 
        # dtw / linalg_norm
        "threshold_r": tune.uniform(search_bounds["threshold_r"][0], search_bounds["threshold_r"][1]),
        "dtw_window_frac": tune.choice(search_bounds["dtw_window_frac"]), 
        # metrics
        "penalty_m": tune.choice(search_bounds["penalty_m"])
    }

    # ==============================================================
    # MLflow Recorder
    # ==============================================================
    mlflow_callback = MLflowLoggerCallback(
        tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
        experiment_name=f"FSM_TUNING_{year}",
        save_artifact=False # auto save avoid occupy hardware
    )
   # --- 配置 ASHA 算法 (早停) 避免score -np.inf ---
    asha_scheduler = tune.schedulers.ASHAScheduler( 
        grace_period=search_bounds["grace_period"], 
        reduction_factor=search_bounds["reduction_factor"]  
    )

    wrapped_trainable = tune.with_resources(
        tune.with_parameters(
            trainable, 
            hf_dfs=hf_dfs, 
            daily_ret=daily_ret,
            macro_dict=macro_data["macro_dict"],
            gpd_dict=gpd_dict,
            run_params=rp
        ),
        resources={"cpu": 1, "gpu": 0} 
    )

    tuner = tune.Tuner( 
        wrapped_trainable,
        param_space=search_space,   
        tune_config=tune.TuneConfig(
            # metric: 优化目标, mode:最大化 
            metric="metrics_score",
            mode="max", 
            search_alg=search_alg, 
            num_samples=search_bounds["num_trials"],            
            scheduler=asha_scheduler,
            max_concurrent_trials=search_bounds["max_concurrent_trials"]
        ),
        run_config=tune.RunConfig(
            name=f"fsm_hpo_{year}",
            storage_path="/tmp/ray_tune_results",
            callbacks=[mlflow_callback] 
        ),
    )
    results = tuner.fit() 
    # =========================================================
    # 🌟 Ray Error -> Airflow Skip
    # =========================================================
    best_trial = results.get_best_result("metrics_score", "max")
    score = best_trial.metrics.get("metrics_score", 0.0)
    is_strictly_significant = best_trial.metrics.get("passed_strict_alpha", False)
        
    if score <= 0.0 or not is_strictly_significant:
        print(f"⚠️ {year-1} 年无收敛参数 (Score: {score}, Sig: {is_strictly_significant})")
        return False 
 
    # ==============================================================
    # MLflow Model Registry
    # ==============================================================
    mlflow.set_experiment("FSM_Production_Models")
    with mlflow.start_run(run_name=f"FSM_{year}_v1"):
        mlflow.log_params(best_trial.config)
        mlflow.log_metric("train_score", best_trial.metrics["metrics_score"])
        
        model_checkpoint = {
            "config": best_trial.config,
            "motif": np.array(best_trial.metrics["learned_motif"]),
            "fsm_matrix": np.array(best_trial.metrics["fsm_prior_matrix"]),
            "valid_year": year
        }
        os.makedirs(MODEL_DIR, exist_ok=True)
        local_path = f"{MODEL_DIR}/model_{year}.pkl"
        with open(local_path, "wb") as f: 
            pickle.dump(model_checkpoint, f)
        mlflow.log_artifact(local_path, artifact_path="best_tuned_model")
        mlflow.set_tag("stage", "production")

    gc.collect()
    return True


@task(name="Out_of_Sample_Inference") 
def oos_node(year: int, g_paths: dict, gpd_path: str, oos_path: str, exp_config: dict):
    
    final_model_path = get_latest_ckpt(year)
    if not final_model_path:
        print(f"⚠️ {year} 年无可用模型，跳过 OOS。")
        return
    print(f"🚀 {year} 年 OOS 推理使用模型: {final_model_path}")
    
    macro_data = joblib.load(g_paths["macro_path"])
    gpd_dict = joblib.load(gpd_path)
    global_daily_ret = pl.read_parquet(g_paths["dret_path"])
    hf_dfs = joblib.load(oos_path)
    
    with open(final_model_path, "rb") as f: 
        model_config = pickle.load(f)
        
    rp = exp_config["run_params"] 
    oos_panel = build_panel_from_chunk(hf_dfs, global_daily_ret, model_config["config"], rp["signal_type"])
    
    if len(oos_panel) == 0: return
    
    infer_model = MotifFSMModel(model_config["config"], macro_data["macro_dict"], gpd_dict, rp["quantiles"])
    scored_df = infer_model.predict(oos_panel, model_config["fsm_matrix"], model_config["motif"], top_k=10)
        
    if len(scored_df) > 0:
        output_path = f"{BASE_DIR}/scores"
        os.makedirs(output_path, exist_ok=True)
        scored_df.write_parquet(f"{output_path}/scores_{year}.parquet")
        print(f"✅ {year} 年 OOS 生成打分: {len(scored_df)} 条记录")

# =========================================================
# DAG
# =========================================================
@flow(name="WFO_FSM_Pipeline")
def wfo_pipeline():
    logger = get_run_logger()
    
    exp_config = {
        "run_params": {
            "start_date": 20040101, 
            "end_date": 20111231, 
            "benchmark": "1A0001", 
            "quantiles": [0.1, 0.3, 0.7, 0.9],
            "loopback": 504,
            "freq_month": 6, 
            "stats_window": [3, 4, 5], 
            "signal_type": "vwap"
        },
        "search_bounds": {
            "downsample": [15, 20, 30, 60], 
            "ndays": [1, 2, 3], 
            "threshold_r": [0.60, 0.95],
            "dtw_window_frac": [0.05, 0.10, 0.15, 0.20], 
            "penalty_m": [20, 30, 40],

            "num_trials": 200, 
            "grace_period": 5,
            "reduction_factor": 4,

            "max_concurrent_trials": 8
        }
    }

    global_paths = get_macro(exp_config)
    gpd_global_path = compute_global_gpd(global_paths["dret_path"], exp_config)

    # Walk-Forward
    for y in range(2004, 2011):
        logger.info(f"========== 🚀 开始执行 {y} 年 Walk-Forward ==========")
        
        paths = extract_hf_data(y, global_paths["macro_path"], exp_config)
        needs_retune = check_decay_node(y, global_paths, gpd_global_path, paths["oos"], exp_config)
        
        if needs_retune:
            logger.info(f"🔄 启动 {y-1} 年数据 Ray Tune 调优...")
            tune_success = tune_node(y, global_paths, gpd_global_path, paths["train"], exp_config)
            if not tune_success:
                logger.warning(f"⏩ {y} 年无可用模型，跳过 OOS 推理。")
                continue
        else:
            logger.info(f"♻️ 复用前代模型")
        
        oos_node(y, global_paths, gpd_global_path, paths["oos"], exp_config)


if __name__ == "__main__":

    # standalone not worker
    # wfo_pipeline.serve(
    #     name="FSM-WFO-Production",
    #     # image="bt_studio_image:v1",        
    #     # schedule={"cron": "0 18 * * *"},    
    #     # parameters={                        
    #     #     "exp_config": { ... }
    #     # }
    # )
    
    # Production Work Pool
    # wfo_pipeline.from_source(
    #     source="/Users/hengxinliu/startup/bt_studio", # 你的项目绝对路径（或者 Git 地址）
    #     entrypoint="bt_studio/pipeline/dags/fsm.py:wfo_pipeline" # 必须是：文件名:函数名
    # ).deploy(
    #     name="FSM-WFO-Production",
    #     work_pool_name="compute-node-pool", 
    # )

    # prefect worker start --pool compute-node-pool

    # prefect deployment run "WFO_FSM_Pipeline/FSM-WFO-Production"

    wfo_pipeline() # for test
