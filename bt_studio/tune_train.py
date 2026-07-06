
import os

# ==============================================================================
# C++ / OpenMP Ray Worker Polars/NumPy CEngine DeadLock Prevention
# ==============================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"
os.environ["RAYON_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'

os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5001"

import multiprocessing
import numpy as np
import polars as pl
from datetime import datetime
from dotenv import load_dotenv

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass 

import ray
import optuna
import mlflow
import pickle
import shutil
import gc
import calendar
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from ray.air.integrations.mlflow import MLflowLoggerCallback
# from prefect import flow, task, get_run_logger

from bt_studio.pipeline.preprocess import prepare_macro, prepare_tick, universe_sample, build_fsm_panel
from bt_studio.pipeline.features import build_ofi
from bt_studio.pipeline.patterns import evaluate_and_build_fsm, discover_fsm_pattern
from bt_studio.pipeline.inference import FSMPredictor
from bt_studio.pipeline.metrics import find_pareto_front, select_best_model_from_pareto

BASE_DIR = "/Users/hengxinliu/startup/bt_studio/result/fsm"
MODEL_DIR = f"{BASE_DIR}/models"
FEATURE_DIR = f"{BASE_DIR}/features"  
SCORE_DIR = f"{BASE_DIR}/scores"     

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(SCORE_DIR, exist_ok=True)


# ==============================================================================
# Node 1 Macro and Universe
# ==============================================================================

# @task(name="Node_Prepare_Macro")
def node_prepare_macro(exp_config: dict, warm=10000):
    common_config = exp_config["run_params"]
    os.makedirs(BASE_DIR, exist_ok=True)
    dret_path = f"{BASE_DIR}/global_daily.parquet"

    if os.path.exists(dret_path):
        print(f"✅ [Cache Hit] Daily Cache: {dret_path}")
    else: 
        print(f"🚀 [Cache Miss] Generating Daily Universe...")
        universe_lf, daily_lf = prepare_macro(
            start_date=common_config["start_date"] - warm, 
            end_date=common_config["end_date"], 
            benchmark=common_config["benchmark"].encode()
        )
        filtered_uni_lf = universe_sample(universe_lf, daily_lf, exceed=120, topk=common_config["top_k_ratio"])
        filtered_uni_lf.sink_parquet(dret_path) # engine=streaming  
        
    final_scan_lf = pl.scan_parquet(dret_path) # lazy operation than read_parquet() for small parquet files 

    dynamic_sids_by_year = dict(
        final_scan_lf.select([
            # pl.col("day").dt.year().alias("year"),
            # pl.col("day").from_epoch(time_unit="s").dt.year().alias("year"),
            pl.col("day").cast(pl.String).str.to_date("%Y%m%d").dt.year().alias("year"),
            pl.col("sid")
        ])
        .group_by("year")
        .agg(pl.col("sid").unique())
        .collect(engine="streaming")  
        .iter_rows() # yield (year, [sids]) 
    )
    return {"dret_path": dret_path, "universe_sids": dynamic_sids_by_year}


# ==============================================================================
# Node 2: Minute and Ofi
# ==============================================================================

# @task(name="Node_Extract_feature") 
def node_extract_feature_monthly(ymonths: list[int], sids: list[bytes], exp_config: dict) -> list[str]:
    """
    - ymonths: [200912, 201001, 201002, ...]
    """
    if not ymonths:
        return []

    paths = []
    missing_ymonths = []
    for ym in ymonths:
        out_path = f"{FEATURE_DIR}/hf_{ym}.parquet"
        if os.path.exists(out_path):
            paths.append(out_path)
        else:
            missing_ymonths.append(ym)
            
    if not missing_ymonths:
        return sorted(paths)

    missing_ymonths = sorted(missing_ymonths)

    # calculate rpc intervals 
    min_ym = missing_ymonths[0]
    start_d = min_ym * 100 + 1
    
    max_ym = missing_ymonths[-1]
    max_year, max_month = max_ym // 100, max_ym % 100
    _, last_day = calendar.monthrange(max_year, max_month)
    end_d = max_ym * 100 + last_day

    print(f"📥 [gRPC Batch] Fetching giant tick data from {start_d} to {end_d} for {len(missing_ymonths)} months...")
    
    snapshot_dict = prepare_tick(start_date=start_d, end_date=end_d, sids=sids)
    
    eager_dfs = []
    print("🧮 Calculating features per security (Eager Evaluation)...")
    for sid_bytes, lf in snapshot_dict.items():
        processed_lf = build_ofi(lf)
        
        # avoid offload of lazyframe
        processed_df = processed_lf.collect()
        
        if processed_df.height > 0:
            eager_dfs.append(processed_df)
        
    if not eager_dfs:
        print("⚠️ [Warning] No data found in this period.")
        return sorted(paths)

    print("🥞 Merging securities into a unified DataFrame...")
    big_df = pl.concat(eager_dfs)
    # big_df = big_df.with_columns((pl.col("day") // 100).alias("month_id"))
    big_df = big_df.with_columns(
    (pl.col("day").dt.year() * 100 + pl.col("day").dt.month()).alias("month_id")
    )
    
    print("💾 Writing partitioned monthly parquets to disk...")
    for ym in missing_ymonths:
        out_path = f"{FEATURE_DIR}/hf_{ym}.parquet"
        
        month_df = big_df.filter(pl.col("month_id") == ym).drop("month_id")
        # month_lf = big_lf.filter(pl.col("month_id") == ym).drop("month_id")
        # month_lf.sink_parquet(out_path) # not supported with window func
        # month_lf.collect(streaming=False).write_parquet(out_path)
        
        if month_df.height > 0:
            month_df.write_parquet(out_path)
            paths.append(out_path)
            print(f"✅ Saved feature: {out_path}")
        else:
            print(f"ℹ️ Month {ym} has no record data, skipping save.")

    return sorted(list(set(paths)))


# ==============================================================================
# Node 3: OOS Decay
# ==============================================================================

# @task(name="Node_Check_Decay") 
def node_check_decay_monthly(
    prev_model_id: int, 
    prev_oos_months: list[int], 
    dret_path: str, 
    prev_oos_paths: list[str], 
    common_config: dict
) -> bool:
    """
    - prev_model_id: e.g. 201007
    - prev_oos_months: e.g. [201007, ..., 201012])
    """
    prev_model_path = f"{MODEL_DIR}/model_{prev_model_id}.pkl"
    if not os.path.exists(prev_model_path):
        print(f"NotFound{prev_model_id} or First force to retune...")
        return True
        
    with open(prev_model_path, "rb") as f:
        model_ckpt = pickle.load(f)
        
    aligned_lfs = [pl.scan_parquet(p) for p in prev_oos_paths if os.path.exists(p)]
    if not aligned_lfs:
        return True 
        
    panel_lf = build_fsm_panel(pl.concat(aligned_lfs), pl.scan_parquet(dret_path), model_ckpt["config"])
    panel_df = panel_lf.collect(streaming=True)
    
    curves_2d = prepare_curves(panel_df, model_ckpt["config"], common_config)
    
    result = evaluate_and_build_fsm_md(
        panel_df, curves_2d, model_ckpt["motif"], model_ckpt["config"], common_config
    )
    
    if result.get("status") == "success" and result["metrics_score"] > 0.0:
        print(f" {prev_model_id} on ({prev_oos_months[0]}-{prev_oos_months[-1]}) effective and (P-val: {result['u_pval']:.4f})")
        return False
        
    print(f"🔄 last {prev_model_id} decay (P-val: {result.get('u_pval', 1.0):.4f}) trigger retune")
    return True


# ==============================================================================
# Node 4: Ray Tune 
# ==============================================================================

def trainable_fsm_worker(config, hf_pa, dret_pa, common_config):
    # ray.tune auto ray.get from ptr to Arrow ---> Polars DataFrame
    hf_lf = pl.from_arrow(hf_pa).clone().lazy() # 
    dret_lf = pl.from_arrow(dret_pa).clone().lazy()

    panel_lf = build_fsm_panel(hf_lf, dret_lf, config)
    result = discover_fsm_pattern(panel_lf, config, common_config)

    if result["status"] == "success":
        tune.report({
            "metrics_score": result["metrics_score"], "u_pval": result["u_pval"],
            "learned_motif": result["learned_motif"], "fsm_network": result["fsm_network"]
        })
    else:
        print(f"\n[Worker Filtered] Config: {config} -> Reason: {result.get('reason', 'Unknown')}\n")
        tune.report({"metrics_score": 0.0, "u_pval": 1.0})

    del panel_lf, hf_lf, dret_lf
    gc.collect()


# @task(name="Node_Tune") 
def node_tune_monthly(model_id: int, dret_path: str, train_paths: list[str], exp_config: dict, prev_model_id:str) -> bool:
    common_config, sb = exp_config["run_params"], exp_config["search_bounds"]

    # =========================================================================
    # read_parquet and put arrow into Ray Plasma
    # =========================================================================

    hf_dfs = [pl.read_parquet(p) for p in train_paths if os.path.exists(p)] 
    if not hf_dfs: 
        return False
    
    hf_pa = pl.concat(hf_dfs).to_arrow()
    dret_pa = pl.read_parquet(dret_path, columns=["day", "sid", "close"]).to_arrow()

    hf_ref = ray.put(hf_pa)
    dret_ref = ray.put(dret_pa)
    
    # =========================================================================
    # Ray search and Opt algo
    # =========================================================================

    search_space = {
        "downsample": tune.choice(sb["downsample"]), 
        "cross_days": tune.choice(sb["cross_days"]), 
        "motif_minutes": tune.choice(sb["motif_minutes"]), 
        "threshold_r": tune.uniform(*sb["threshold_r"]),
        "dtw_window_frac": tune.uniform(*sb["dtw_window_frac"]), 
    }

    points_to_evaluate = None
    if os.path.exists(f"{MODEL_DIR}/model_{prev_model_id}.pkl"):
        print(f"NotFound{prev_model_id} or First force to retune...")
        with open(prev_model_path, "rb") as f:
            prior_cfg = pickle.load(f)["config"]
            points_to_evaluate=[{k: prior_cfg[k] for k in search_space if k in prior_cfg}]

    # multithread / sample optimize
    optuna_sampler = optuna.samplers.TPESampler(n_startup_trials=10, multivariate=True) #
    search_alg = OptunaSearch(
        sampler=optuna_sampler,
        points_to_evaluate=points_to_evaluate
    )

    # =========================================================================
    # Ray Tune and Initialize MLflowLoggerCallback
    # =========================================================================
    wrapped_trainable = tune.with_resources(
        tune.with_parameters(
            trainable_fsm_worker,
            hf_pa=hf_ref,      
            dret_pa=dret_ref,   
            common_config=common_config),
        resources={"cpu": 1, "gpu": 0} 
    )

    mlflow_callback = MLflowLoggerCallback(
        tracking_uri=mlflow.get_tracking_uri(), # default 5000 and set by env
        experiment_name="FSM_Production_Models",
        save_artifact=True 
    )

    tuner = tune.Tuner(
        wrapped_trainable, param_space=search_space,   
        tune_config=tune.TuneConfig(
            metric="metrics_score", 
            mode="max", 
            search_alg=search_alg, 
            num_samples=sb["num_trials"],            
            scheduler=tune.schedulers.ASHAScheduler(grace_period=sb["grace_period"], reduction_factor=sb["reduction_factor"]),
            max_concurrent_trials=sb["max_concurrent_trials"]
        ),
        run_config=tune.RunConfig(
            name=f"fsm_hpo_{model_id}", 
            storage_path="/tmp/ray_results",
            callbacks=[mlflow_callback]
            )  
        )
    
    # =========================================================================
    # Ray Tune Fit and Optuna.importance
    # =========================================================================

    results = tuner.fit()
    df_results = results.get_dataframe()
    best_trial_result = results.get_best_result("metrics_score", "max")
    
    if best_trial_result.metrics.get("metrics_score", 0.0) <= 0.0:
        return False

    # 1. fANOVA 
    is_plateau = validate_parameter_plateau_fanova(
        df_results, best_trial_result.config, best_trial_result.metrics["metrics_score"]
    )
    if not is_plateau:
        print(f"❌ {model_id} isolated")
        return False

    # 2. pareto front
    pareto_front_df = find_pareto_front(df_results)
    best_model_dict = select_best_model_from_pareto(pareto_front_df)
    
    if not best_model_dict: 
        return False
        
    print(f"✅ {model_id} Score: {best_model_dict['metrics_score']:.1f}")
    
    # =========================================================================
    # Model Save
    # =========================================================================
    best_config = {k.replace("config/", ""): v for k, v in best_model_dict.items() if k.startswith("config/")}
    model_ckpt = {
        "config": best_config, 
        "motif": np.array(best_model_dict.get("learned_motif", [])), 
        "fsm_network": best_model_dict.get("fsm_network", {}),
        "valid_month": model_id 
    }
    
    pkl_path = f"{MODEL_DIR}/model_{model_id}.pkl"
    with open(pkl_path, "wb") as f: 
        pickle.dump(model_ckpt, f)
    
    # mlflow.set_experiment("FSM_Production_Models")
    # with mlflow.start_run(run_name=f"FSM_{year}_v3"):
    #     mlflow.log_params(best_trial.config)
    #     mlflow.log_metric("train_score", best_trial.metrics["metrics_score"])

    # # =========================================================================
    # # Callback ---> MLflow Run ID
    # # =========================================================================
    # best_trial_id = best_trial_result.metrics.get("trial_id") 
    # best_run_id = None
    
    # for trial_obj, run_id in mlflow_callback._trial_runs.items():
    #     if trial_obj.trial_id == best_trial_id:
    #         best_run_id = run_id
    #         break

    # # =========================================================================
    # # MlflowClient Async .pkl upload Trial Artifacts 
    # # =========================================================================
    # if best_run_id:
    #     try:
    #         print(f"Link {pkl_path} to MLflow Run: {best_run_id}")
    #         client = mlflow.tracking.MlflowClient()
    #         client.log_artifact(best_run_id, pkl_path) # C Client and avoid start_run lock conflict
    #         print("upload Trial to Artifacts ")
    #     except Exception as e:
    #         print(f"upload failure: {e}")

    return True


# ==============================================================================
# Node 5: OOS 
# ==============================================================================

# @task(name="Node_OOS_Inference") 
def node_oos_inference_monthly(model_id: int, oos_months: list[int], dret_path: str, oos_paths: list[str], exp_config: dict):
    model_path = f"{MODEL_DIR}/model_{model_id}.pkl"
    if not os.path.exists(model_path):
        print(f"⚠️ {model_id} NotFound and Skip OOS Inference")
        return
        
    with open(model_path, "rb") as f:
        model_ckpt = pickle.load(f)
        
    aligned_lfs = [pl.scan_parquet(p) for p in oos_paths if os.path.exists(p)]
    if not aligned_lfs: 
        return
    
    # fsm predict
    panel_lf = build_fsm_panel(pl.concat(aligned_lfs), pl.scan_parquet(dret_path), model_ckpt["config"])
    scored_df = FSMPredictor(model_ckpt).predict(panel_lf)
    
    if scored_df.height > 0:
        out_path = f"{SCORE_DIR}/scores_{model_id}.parquet"
        scored_df.write_parquet(out_path)
        print(f"✅ {model_id} {oos_months[0]} between {oos_months[-1]} Oss: {scored_df.height} ")


# ==============================================================================
# DAG (Walk-Forward)
# ==============================================================================

# @flow(name="WFO_FSM_Pipeline")
def wfo_pipeline(exp_config):
    # avoid parallel thread and system hangup
    runtime_env = {
        "env_vars": {
            "POLARS_MAX_THREADS": "1",
            "RAYON_NUM_THREADS": "1",       
            "NUMBA_NUM_THREADS": "1",       
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1"
        },
        # "working_dir": os.path.dirname(os.path.abspath(__file__)) 
    }
    ray.init(num_cpus=6, runtime_env=runtime_env, ignore_reinit_error=True)

    global_data = node_prepare_macro(exp_config)
    
    daily_df = pl.read_parquet(global_data["dret_path"], columns=["day"])
    all_months = (
        # daily_df.select((pl.col("day").dt.year() * 100 + pl.col("day").dt.month()).alias("month_id"))
        daily_df.select((pl.col("day") // 100).cast(pl.Int32).alias("month_id"))
        .unique().sort("month_id")["month_id"].to_list()
    )
    
    TRAIN_WINDOW = exp_config["run_params"]["train_window"]     
    STEP = exp_config["run_params"]["update_freq"]          
    
    last_available_model_id = None

    for idx in range(TRAIN_WINDOW, len(all_months), STEP):
        oos_months = all_months[idx : min(idx + STEP, len(all_months))]
        train_months = all_months[idx - TRAIN_WINDOW : idx]            
        
        model_id = oos_months[0]           
        
        sids_train = []
        for m in train_months:
            sids_train.extend(global_data["universe_sids"].get(m // 100, []))
        sids_train = list(set(sids_train))
        
        sids_oos = list(set(global_data["universe_sids"].get(model_id // 100, [])))
        
        if not sids_train or not sids_oos: 
            continue
            
        print(f"\n==========================================================================")
        print(f"📅 WFO  | OOS : {model_id} | Train Sids: {len(sids_train)} | OOS Sids: {len(sids_oos)}")
        print(f"==========================================================================\n")
        
        train_paths = node_extract_feature_monthly(train_months, sids_train, exp_config)
        oos_paths = node_extract_feature_monthly(oos_months, sids_oos, exp_config)

        if last_available_model_id is None:
            is_decayed = True
            print("🚀 First run (Cold Start), forcing HPO Training...")
        else:
            prev_oos_months = train_months[-STEP:] 
            prev_oos_paths = node_extract_feature_monthly(prev_oos_months, sids_train, exp_config)
            
            is_decayed = node_check_decay_monthly(
                last_available_model_id, prev_oos_months, global_data["dret_path"], prev_oos_paths, exp_config["run_params"]
            )
        
        if is_decayed:
            print(f"🔄 Model decayed or First Run. Tuning Model for {model_id}...")
            success = node_tune_monthly(model_id, global_data["dret_path"], train_paths, exp_config, last_available_model_id)
            if not success:
                print(f"⚠️ {model_id} HPO Failed. Skipping this window.")
                continue
            last_available_model_id = model_id 
        else:
            print(f"🌲 Model {last_available_model_id} remains effective. Inheriting to {model_id}")
            shutil.copy(f"{MODEL_DIR}/model_{last_available_model_id}.pkl", f"{MODEL_DIR}/model_{model_id}.pkl")
            last_available_model_id = model_id 
            
        node_oos_inference_monthly(model_id, oos_months, global_data["dret_path"], oos_paths, exp_config)



if __name__ == "__main__":

    load_dotenv()

    exp_config = {
        "run_params": {
            "start_date": 20100101, "end_date": 20201231, "benchmark": "1A0001",
            "train_window": 12,
            "update_freq": 6, 
            "top_k_ratio": 0.25, # used for sample
            "exclude_bars": 10, # exclude last 10 bars means 14:50
            "edge_ratio": 0.25, # ratio of macro state edge bins 
            "alternative": "greater", # stats 
            "stats_windows": [1,2,3], # T+1 ---> T+3 Fut Ret
            "prior_weight": 0.3 # 0.3 + 0.7 
        },

        "search_bounds": {
            "downsample": [3, 5, 10], # downsample for DTW
            "cross_days": [2, 3, 4], # concat cross_days of lagged curves to 2D array for DTW 
            "motif_minutes": [30, 60, 120, 240], # used from motif length intraday
            "threshold_r": [0.7, 0.90], 
            "dtw_window_frac": [0.05, 0.10], # used for DTW offset 
            "grace_period": 5, "reduction_factor": 4, 
            "num_trials": 100, 
            "max_concurrent_trials": 4
        }
    }

    wfo_pipeline(exp_config)
