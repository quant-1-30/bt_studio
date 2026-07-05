
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
import gc
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from ray.air.integrations.mlflow import MLflowLoggerCallback
# from prefect import flow, task, get_run_logger

from bt_studio.pipeline.preprocess import prepare_macro, prepare_tick, universe_sample, build_fsm_panel
from bt_studio.pipeline.features import build_ofi
from bt_studio.pipeline.patterns import evaluate_and_build_fsm, discover_fsm_pattern
from bt_studio.pipeline.inference import FSMPredictor
from bt_studio.utils.common import get_latest_ckpt

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
    common_config = exp_config["common_params"]
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
        filtered_uni_lf = universe_sample(universe_lf, daily_lf, exceed=120, topk=rq["top_k_ratio"])
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
def node_extract_feature(year: int, sids: list[bytes], exp_config: dict):
    
    def _fetch_and_build(target_year: int, out_path: str):
        out_path = f"{BASE_DIR}/features/hf_{target_year}.parquet"
        
        if os.path.exists(out_path):
            print(f"✅ [Cache Hit] Feature: {out_path}")
            return out_path

        # rpc
        start_d = (year - 1) * 10000 + 101
        end_d = (year - 1) * 10000 + 1231 
        snapshot_dict = prepare_tick(start_date=start_d, end_date=end_d, sids=sids)
        
        lfs = []
        for sid, lf in snapshot_dict.items():
            lfs.append(build_ofi(lf))
            
        if lfs:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            # pl.concat(lfs).collect().write_parquet(out_path)
            pl.concat(lfs).sink_parquet(out_path)
            return out_path
        return None

    train_path = _fetch_and_build(year, f"{BASE_DIR}/train/hf_{year-1}.parquet")
    oos_path = _fetch_and_build(year, f"{BASE_DIR}/oos/hf_{year}.parquet")
    return {
        "train_paths": [train_path] if train_path else [], 
        "oos_paths": [oos_path] if oos_path else []
    }


def node_extract_feature_timer(year_months: list[int], sids: list[bytes], exp_config: dict) -> list[str]:
    """
        - months: [201001, 201002, ...]
    """
    paths = []
    for ym in year_months:
        out_path = f"{FEATURE_DIR}/hf_{ym}.parquet"
        
        if os.path.exists(out_path):
            paths.append(out_path)
            continue
            
        print(f"📥 [I/O]  y{m} feature and calculate from rpc tick ...")
        
        # ym ---> YYYYMMDD start / end
        start_d = m * 100 + 1
        end_d = m * 100 + 31 
        snapshot_dict = prepare_tick(start_date=start_d, end_date=end_d, sids=sids)
        
        lfs = []
        for sid_bytes, lf in snapshot_dict.items():
            # OFI + Volatility
            lfs.append(build_ofi_vol(lf))
            
        if lfs:
            # pl.concat(lfs).collect(streaming=True).write_parquet(out_path)
            pl.concat(lfs).sink_parquet(out_path)
            paths.append(out_path)
            print(f"💾 [I/O] {ym} save to: {out_path}")
    return paths


# ==============================================================================
# Node 3: OOS Decay
# ==============================================================================

def node_check_decay(
    prev_model_id: int, 
    prev_oos_months: list[int], 
    dret_path: str, 
    prev_oos_paths: list[str], 
    exp_config: dict
) -> bool:
    """
    - prev_model_id: 201007
    - prev_oos_months: [201007, ..., 201012]
    """
    prev_model_path = f"{MODEL_DIR}/model_{prev_model_id}.pkl"
    if not os.path.exists(prev_model_path):
        print(f"🔄 未检测到历史模型 {prev_model_id}，判定为首期运行，强制启动重训...")
        return True
        
    with open(prev_model_path, "rb") as f:
        model_ckpt = pickle.load(f)
        
    aligned_lfs = [pl.scan_parquet(p) for p in prev_oos_paths if os.path.exists(p)]
    if not aligned_lfs:
        return True 
        
    # FSM Matrix and Extract Feature and Evaluate
    common_config = exp_config["run_params"]

    panel_lf = build_fsm_panel(pl.concat(aligned_lfs), pl.scan_parquet(dret_path), model_ckpt["config"]).collect(engine="streaming")
    panel_df = panel_lf.collect(streaming=True)
    curves_2d = prepare_curves(oss_panel_df, common_config, model_ckpt["config"])
    
    # avoid stumpy and pre motif scan 2D DTW
    result = evaluate_and_build_fsm_md(
        panel_df, curves_md, model_ckpt["motif"], model_ckpt["config"], common_config
    )
    
    if result.get("status") == "success" and result["metrics_score"] > 0.0:
        print(f" {prev_model_id} effective on ({prev_oos_months[0]}-{prev_oos_months[-1]}) and (P-val: {result['u_pval']:.4f})")
        return False 
        
    print(f" {prev_model_id} decay and (P-val: {result.get('u_pval', 1.0):.4f}) retune")
    return True


# ==============================================================================
# Node 4: Ray Tune 
# ==============================================================================

def trainable_fsm_worker(config, hf_pa, dret_pa, common_config):
    # ray.tune auto ray.get from ptr to Arrow ---> Polars DataFrame
    hf_lf = pl.from_arrow(hf_pa).clone().lazy() # 
    dret_lf = pl.from_arrow(dret_pa).clone().lazy()

    panel_lf = build_fsm_panel(hf_lf, dret_lf, config)
    result = discover_fsm_pattern(config, panel_lf, common_config)

    if result["status"] == "success":
        tune.report({
            "metrics_score": result["metrics_score"], "u_pval": result["u_pval"],
            "learned_motif": result["learned_motif"], "fsm_network": result["fsm_network"]
        })
    else:
        print(f"\n[Worker Filtered] Config: {config} -> Reason: {result.get('reason', 'Unknown')}\n")
        tune.report({"metrics_score": 0.0, "u_pval": 1.0})

    # enforce recycle
    del panel_lf, hf_lf, dret_lf
    gc.collect()

# @task(name="Node_Tune") 
def node_tune(year: int, dret_path: str, train_paths: list, exp_config: dict):
    common_config, sb = exp_config["run_params"], exp_config["search_bounds"]

    # =========================================================================
    # main_thread read_parquet and put arrow into Ray Plasma
    # =========================================================================
    hf_dfs = [pl.read_parquet(p) for p in train_paths]
    hf_pa = pl.concat(hf_dfs).to_arrow() 
    dret_pa = pl.read_parquet(dret_path).to_arrow()
    
    # put into Ray Plasma
    hf_ref = ray.put(hf_pa)
    dret_ref = ray.put(dret_pa)
    
    # prev_ckpt = get_latest_ckpt(year - 1, MODEL_DIR)

    # =========================================================================
    # ray tune and search
    # =========================================================================
    
    search_space = {
        "downsample": tune.choice(sb["downsample"]), 
        "cross_days": tune.choice(sb["cross_days"]), 
        "motif_minutes": tune.choice(sb["motif_minutes"]), 
        "threshold_r": tune.uniform(*sb["threshold_r"]),
        "dtw_window_frac": tune.uniform(*sb["dtw_window_frac"]), 
    }


    points_to_evaluate = None
    if prev_ckpt:
        prior_cfg = pickle.load(open(prev_ckpt, "rb"))["config"]
        points_to_evaluate=[{k: prior_cfg[k] for k in search_space if k in prior_cfg}]

    # multithread / sample optimize
    optuna_sampler = optuna.samplers.TPESampler(n_startup_trials=10, multivariate=True) #
    search_alg = OptunaSearch(
        sampler=optuna_sampler,
        points_to_evaluate=points_to_evaluate
    )
    
    wrapped_trainable = tune.with_resources(
        # tune.with_parameters(trainable_fsm_worker, hf_paths=train_paths, dret_path=dret_path, prior_config=rp),
        tune.with_parameters(
            trainable_fsm_worker,
            hf_pa=hf_ref,      
            dret_pa=dret_ref,   
            common_config=common_config),
        resources={"cpu": 1, "gpu": 0} 
    )

    # =========================================================================
    # Initialize MLflowLoggerCallback and Tracking all Trial in MLflow
    # =========================================================================
    mlflow_callback = MLflowLoggerCallback(
        tracking_uri=mlflow.get_tracking_uri(), # via env
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
            scheduler=tune.schedulers.ASHAScheduler(grace_period=sb["grace_period"], 
            reduction_factor=sb["reduction_factor"]),
            max_concurrent_trials=sb["max_concurrent_trials"]
        ),
        run_config=tune.RunConfig(
            name=f"fsm_hpo_{year}", 
            storage_path="/tmp/ray_results"),
            callbacks=[mlflow_callback]  
        )

    results = tuner.fit()
    best_trial = results.get_best_result("metrics_score", "max")
    df_results = results.get_dataframe() # all 100 trial
    
    if best_trial.metrics.get("metrics_score", 0.0) <= 0.0:
        return False 
 
    # mlflow.set_experiment("FSM_Production_Models")
    # with mlflow.start_run(run_name=f"FSM_{year}_v3"):
    #     mlflow.log_params(best_trial.config)
    #     mlflow.log_metric("train_score", best_trial.metrics["metrics_score"])
        
    model_ckpt = {
        "config": best_trial.config, 
        "motif": np.array(best_trial.metrics.get("learned_motif", [])), 
        "fsm_network": best_trial.metrics.get("fsm_network", {}),
        "valid_year": year
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(f"{MODEL_DIR}/model_{year}.pkl", "wb") as f: 
        pickle.dump(model_ckpt, f)
    
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

    gc.collect()
    return True

# ==============================================================================
# Node 5: OOS 
# ==============================================================================

# @task(name="Node_OOS_Inference") 
def node_oos_inference(year: int, dret_path: str, oos_paths: list, exp_config: dict):
    # final_model_path = get_latest_ckpt(year, MODEL_DIR)
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
        # get_run_logger().info(f"✅ {year} OOS Score: {scored_df.height}")


# ==============================================================================
# DAG (Walk-Forward)
# ==============================================================================

# @flow(name="WFO_FSM_Pipeline")
def wfo_pipeline(exp_config):
    # logger = get_run_logger()

    runtime_env = {
        "env_vars": {
            "POLARS_MAX_THREADS": "1",
            "RAYON_NUM_THREADS": "1",       # rayon Rust 
            "NUMBA_NUM_THREADS": "1",       # numba
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1"
        }
    }

    ray.init(num_cpus=6, runtime_env=runtime_env, ignore_reinit_error=True)

    # =====================================================================
    # Node 1 Macro State
    # =====================================================================
    global_data = node_prepare_macro(exp_config)

    start_year = exp_config["run_params"]["start_date"] // 10000
    end_year = exp_config["run_params"]["end_date"] // 10000

    for y in range(start_year, end_year + 1):
        # logger.info(f"==========  {y} year Walk-Forward ==========")

        # =====================================================================
        # Node 2 Select UniversePool between Train and Oss
        # =====================================================================
        sids_y_minus_1 = global_data["universe_sids"].get(y - 1, [])
        sids_y = global_data["universe_sids"].get(y, [])
        
        target_sids = list(set(sids_y_minus_1 + sids_y)) # for speed
        if not target_sids:
            continue

        # logger.info(f"========= {y} year Walk-Forward (Active Universe: {len(target_sids)} ==========") 
        
        # =====================================================================
        # Node 3 Feature Extraction
        # =====================================================================
        paths = node_extract_feature(y, target_sids, exp_config)
        
        # =====================================================================
        # Node 4 Test Decay of Previous Model
        # =====================================================================
        if node_check_decay(y, global_data["dret_path"], paths["oos_paths"], exp_config):
            # logger.info(f"🔄 启动 {y-1} Ray Tune ...")
            # =====================================================================
            # Node 5 Ray Tune for new Model
            # =====================================================================
            if not node_tune(y, global_data["dret_path"], paths["train_paths"], exp_config): 
                continue
        
        # =====================================================================
        # Node 6 Oss Inference
        # =====================================================================
        node_oos_inference(y, global_data["dret_path"], paths["oos_paths"], exp_config)


if __name__ == "__main__":

    load_dotenv()

    exp_config = {
        "common_params": {
            "start_date": 20100101, "end_date": 20201231, "benchmark": "1A0001", 
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
            "max_concurrent_trials": 6
        }
    }

    wfo_pipeline(exp_config)





# **情况 A：模型衰减了（OOS 失效），触发 Retune**
# * **动作**：彻底抛弃过去所有的包袱！Optuna 用最近 1 年（2010.07-2011.06）的数据挖出新 Motif。
# * **矩阵构建**：新 Motif 的 FSM 矩阵，**完完全全、仅仅基于这最近 1 年的数据生成**。没有任何先验（Prior）融合。这保证了模型具有极其敏锐的市场嗅觉，零滞后！

# **情况 B：模型没有衰减（OOS 有效），继续沿用老 Motif**
# * 老形态依然能赚钱，不需要启动 Optuna 重新搜索。
# * **但是矩阵需要更新吗？必须更新！**（因为形态虽然有效，但市场可能从牛转熊，状态转移概率变了）。
# * **动作（滚动计数）**：拿着这个老 Motif，在往前滑动的 1 年窗口（2010.07-2011.06）里重新扫描一遍全市场，**重新统计一遍频数（Count），直接生成新的 FSM 矩阵**。
# * **为什么不滞后？**：因为我们用的是“滑动窗口（Rolling Window）”而不是“扩张窗口（Expanding Window）”。随着时间推移，2010 年上半年的数据被自动移出了窗口，它的“牛市记忆”被物理清除了，所以绝对不会有滞后性。

# **💡 极简代码修正结论：**
# 在 `evaluate_and_build_fsm_md` 中，**彻底删掉 `prior_fsm_network` 的贝叶斯融合代码**！完全信任当前滑动窗口（最近 12 个月）统计出来的客观概率。相信数据的自然演替，才是对抗市场漂移最好的方式。
