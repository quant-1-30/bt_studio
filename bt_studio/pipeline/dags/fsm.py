import os
import joblib
import shutil
import polars as pl
from datetime import datetime
from airflow.sdk import dag, task, task_group
from airflow.task.trigger_rule import TriggerRule
from airflow import DAG

from bt_studio.pipeline.fsm.preprocess import *

BASE_DIR = "/Users/hengxinliu/startup/bt_studio/result/fsm"
MODEL_DIR = f"{BASE_DIR}/models"


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


@dag(dag_id="fsm_wfo_pipeline_v3", start_date=datetime(2023, 1, 1), schedule=None, catchup=False)
def wfo_pipeline(start_year: int, end_year: int):

    # ========================================================
    # load yaml
    # ========================================================
    @task
    def load_experiment_config():
        """
        """
        return {
            "run_params": {
                "start_date": 20050101, 
                "end_date": 20260101, 
                "benchmark": "1A0001", 
                "quantiles": [0.1, 0.3, 0.7, 0.9],
                "loopback": 504,
                "freq_month": 6, 
                "stats_window": [5, 10, 20], 
                "signal_type": "vwap",

                # num_asset
                "num_samples": 0.1,
                # tune trials
                "num_trials": 200 
            },
            "search_bounds": {
                "downsample": [15, 20, 30, 60], 
                "ndays": [1, 2, 3], 

                # dtw / linalg_norm
                "threshold_r": [0.70, 0.95],
                "dtw_window_frac": [0.05, 0.10, 0.15, 0.20], 

                # metrics
                "penalty_m": [20, 30, 40],

            },
        }

    exp_config = load_experiment_config()

    # ========================================================
    # macro data
    # ========================================================
    @task(multiple_outputs=True)  # <--- 增加这个参数
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
    @task
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

    exp_config = load_experiment_config()
    
    global_paths = get_macro(exp_config)

    gpd_global_path = compute_global_gpd(global_paths["dret_path"], exp_config)

    # ========================================================
    # WFO TaskGroup
    # ========================================================
    def build_wfo_year_group(year: int):
        
        @task_group(group_id=f"WFO_Loop_{year}")
        def wfo_year():

            @task(multiple_outputs=True)  
            def extract_hf_data(macro_path: str, config: dict):
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
            
            @task.branch
            def check_decay_node(g_paths: dict, gpd_path: str, oos_path: str):
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

                rp = config["run_params"]
                valid_panel = build_panel_from_chunk(hf_dfs, global_daily_ret, model_config["config"], rp["signal_type"])
                
                valid_model = MotifFSMModel(model_config["config"], macro_data["macro_dict"], gpd_dict, rp["quantiles"])
                eval_res = valid_model.validate(valid_panel, model_config["motif"], gpd_dict, rp["stats_window"])
       
                if eval_res.get("status") == "success" and eval_res["metrics_score"] > 0:
                    print(f"✅ 模型表现优异 (Score: {eval_res['metrics_score']:.2f})")
                    return f"WFO_Loop_{year}.reuse_model"
                else:
                    print(f"❌ Alpha 衰减，重新搜索！")
                    return f"WFO_Loop_{year}.run_ray_tune"

            @task(task_id="run_ray_tune")
            def tune_node(g_paths: dict, gpd_path: str, hf_train_path: str, config: dict):
                
                # =================================================================
                # Default Prior To Historical Prior 
                # =================================================================
                prev_model_path = get_latest_ckpt(year - 1)

                if prev_model_path:
                    with open(prev_model_path, "rb") as f: 
                        prior_config = pickle.load(f)["config"]

                    prior_point = {k: prior_config[k] for k in search_bounds.keys() if k in prior_config}
                    search_alg = HyperOptSearch(points_to_evaluate=[prior_point])
                    print(f"🧠 继承历史先验: {prior_point}")
                else:
                    search_alg = HyperOptSearch() # 全局无偏探索
                    print("🌍 无历史先验，执行全局贝叶斯探索")

                # avoid Airflow XCom Crash
                search_space = {
                    "downsample": tune.choice(search_bounds["downsample"]), 
                    "ndays": tune.choice(search_bounds["m"]), 

                    # dtw / linalg_norm
                    "threshold_r": tune.uniform(search_bounds["threshold_d"][0], search_bounds["threshold_d"][1]),
                    "dtw_window_frac": tune.choice([0.05, 0.10, 0.15, 0.20]), 

                    # metrics
                    "penalty_m": tune.choice(search_bounds["penalty_m"]), 
                }

                # load Ray Train
                macro_data = joblib.load(g_paths["macro_path"])
                daily_ret = pl.read_parquet(g_paths["dret_path"])
                gpd_dict = joblib.load(gpd_path)
                # hf_dfs = joblib.load(hf_paths["train"])
                hf_dfs = joblib.load(hf_train_path)
                
                rp = config["run_params"]

                wrapped_trainable = tune.with_resources(
                    tune.with_parameters(
                        fsm_trainable, 
                        hf_dfs=hf_dfs, 
                        daily_ret=daily_ret,
                        macro_dict=macro_data["macro_dict"],
                        gpd_dict=gpd_dict,
                        config=rp
                    ),
                    resources={"cpu": 2, "gpu": 0} 
                )
                
                tuner = tune.Tuner( 
                    wrapped_trainable,
                    param_space=search_space,   
                    tune_config=tune.TuneConfig(
                        search_alg=search_alg, 
                        num_samples=rp["num_trials"],            
                        scheduler=asha_scheduler
                    ),
                    run_config=tune.RunConfig(
                        name=f"fsm_hpo_{year}",
                        storage_path="/tmp/ray_tune_results",
                    ),
                )
                results = tuner.fit() 
                best_trial = results.get_best_result("metrics_score", "max")
            
                if best_trial.metrics.get("metrics_score", -np.inf) == -np.inf:
                    print(f"⚠️ {year-1} 年重搜失败")
                    return "Failed"

                model_checkpoint = {
                    "config": best_trial.config,
                    "motif": np.array(best_trial.metrics["learned_motif"]),
                    "fsm_matrix": np.array(best_trial.metrics["fsm_prior_matrix"]),
                    "valid_year": year
                }

                os.makedirs(MODEL_DIR, exist_ok=True)
                with open(f"{MODEL_DIR}/model_{year}.pkl", "wb") as f: 
                    pickle.dump(model_checkpoint, f)
                gc.collect()
                return "Tuned"

            @task(task_id="reuse_model")
            def reuse_node():
                """DAG PlaceHold"""
                return "Reused"

            @task(trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
            def oos_node(g_paths: dict, gpd_path: str, oos_path: str, config: dict, tune_node_res, reuse_node_res):
                # latest model
                final_model_path = get_latest_model_path(year)
                if not final_model_path:
                    print(f"⚠️ {year} 年无可用模型，跳过 OOS。")
                    return

                print(f"🚀 {year} 年 OOS 推理使用模型: {final_model_path}")
                
                macro_data = joblib.load(g_paths["macro_path"])
                gpd_dict = joblib.load(gpd_path)
                global_daily_ret = pl.read_parquet(g_paths["dret_path"])
                # hf_dfs = joblib.load(hf_paths["oos"])
                hf_dfs = joblib.load(oos_path)
                
                with open(final_model_path, "rb") as f: 
                    model_config = pickle.load(f)

                rp = config["run_params"] 
                oos_panel = build_panel_from_chunk(hf_dfs, global_daily_ret, model_config["config"], rp["signal_type"])
                
                if len(oos_panel) == 0: return
                
                infer_model = MotifFSMModel(model_config["config"], macro_data["macro_dict"], gpd_dict, rp["quantiles"])
                scored_df = infer_model.predict(oos_panel, model_config["fsm_matrix"], model_config["motif"], top_k=10)
                    
                if len(scored_df) > 0:
                    output_path = f"{BASE_DIR}/scores"
                    os.makedirs(output_path, exist_ok=True)
                    scored_df.write_parquet(f"{output_path}/scores_{year}.parquet")
                    print(f"✅ {year} 年 OOS 生成打分: {len(scored_df)} 条记录")

            # --- DAG ---
            paths = extract_hf_data(global_paths["macro_path"], exp_config)
            branch = check_decay_node(global_paths, gpd_global_path, paths["oos"])
            
            t_tune = tune_node(global_paths, gpd_global_path, paths["train"], exp_config)
            t_reuse = reuse_node()
            
            branch >> [t_tune, t_reuse]
            oos_node(global_paths, gpd_global_path, paths["oos"], exp_config, t_tune, t_reuse)

        return wfo_year()

    # ========================================================
    # WFO
    # ========================================================
    prev_group = None
    for y in range(2008, 2027):
        curr_group = build_wfo_year_group(y)
        if prev_group is not None:
            prev_group >> curr_group
        prev_group = curr_group


dag = wfo_pipeline(2006, 2026)


# if __name__ == "__main__":

#     dag.test()
