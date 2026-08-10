import os
import multiprocessing
import numpy as np
import polars as pl
from dotenv import load_dotenv

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import gc
import re
import pickle
import shutil
import ray
import optuna
import mlflow
import calendar
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from ray.air.integrations.mlflow import MLflowLoggerCallback

from bt_studio.pipeline.preprocess import prepare_macro, prepare_tick, universe_sample, extract_curves_from_panel, build_static_panel, build_fsm_panel
from bt_studio.pipeline.features import build_ofi
from bt_studio.pipeline.patterns import prepare_curves, evaluate_and_build_fsm, discover_fsm_pattern
from bt_studio.pipeline.inference import FSMPredictor
from bt_studio.pipeline.metrics import validate_parameter_plateau_fanova, find_pareto_front, select_best_model_from_pareto
from bt_studio.pipeline.utils import consume_time

# ==============================================================================
# Node 1 Macro and Universe
# ==============================================================================

# @task(name="Node_Prepare_Macro")
def node_prepare_macro(common_config: dict, warm=10000):
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
        filtered_uni_lf = universe_sample(universe_lf, daily_lf, common_config)
        filtered_uni_lf.sink_parquet(dret_path) 
        
    final_scan_lf = pl.scan_parquet(dret_path) 

    # =========================================================================
    # Groupby Month ID avoid lookahead
    # =========================================================================
    dynamic_sids_by_month = dict(
        final_scan_lf.select([
            (pl.col("date").dt.year() * 100 + pl.col("date").dt.month()).alias("month_id"),
            pl.col("sid")
        ])
        .group_by("month_id")
        .agg(pl.col("sid").unique())
        .collect(engine="streaming")  
        .iter_rows() # (month_id, [sids])
    )
    return {"dret_path": dret_path, "universe_sids": dynamic_sids_by_month}


# ==============================================================================
# Node 2: Minute and Ofi
# ==============================================================================

# @task(name="Node_Extract_feature") 
def node_extract_feature_monthly(ymonths: list[int], universe_sids: dict, common_config: dict) -> list[str]:
    """
    - ymonths: [200912, 201001, ...]
    - universe_sids: dict(month_id -> list[sids])
    """
    if not ymonths: return []

    paths, missing_ymonths = [], []
    for ym in ymonths:
        out_path = f"{FEATURE_DIR}/hf_{ym}.parquet"
        if os.path.exists(out_path):
            paths.append(out_path)
        else:
            missing_ymonths.append(ym)
            
    if not missing_ymonths: return sorted(paths)
    missing_ymonths = sorted(missing_ymonths)

    # MDAPI Api 
    month_sids_list = [universe_sids.get(ym, []) for ym in missing_ymonths]
    union_sids = list(set().union(*month_sids_list))
    if not union_sids: return sorted(paths)

    min_ym, max_ym = missing_ymonths[0], missing_ymonths[-1]
    start_d = min_ym * 100 + 1
    max_year, max_month = max_ym // 100, max_ym % 100
    _, last_day = calendar.monthrange(max_year, max_month)
    end_d = max_ym * 100 + last_day

    print(f"📥 [gRPC Batch] Fetching tick data from {start_d} to {end_d}...")
    snapshot_dict = prepare_tick(start_date=start_d, end_date=end_d, sids=union_sids)
    
    # [FIX] Concat ALL sids BEFORE build_ofi (cross-sectional demean needs multiple sids)
    all_lazy_frames = list(snapshot_dict.values())
    if not all_lazy_frames:
        return sorted(paths)

    combined_lf = pl.concat(all_lazy_frames)
    ofi_df = build_ofi(combined_lf, common_config).collect()

    if ofi_df.height == 0:
        return sorted(paths)

    # [DEBUG] Verify ofi_ratio is not all zeros
    ofi_stats = ofi_df.select([
        pl.col("ofi_ratio").mean().alias("mean"),
        pl.col("ofi_ratio").std().alias("std"),
        (pl.col("ofi_ratio") != 0).sum().alias("nonzero_count"),
        pl.len().alias("total"),
    ]).row(0, named=True)
    print(f"[DEBUG build_ofi] ofi_ratio: mean={ofi_stats['mean']:.6f}, std={ofi_stats['std']:.6f}, nonzero={ofi_stats['nonzero_count']}/{ofi_stats['total']}")
    if ofi_stats["nonzero_count"] == 0 or ofi_stats["std"] <= common_config["eps"]:
        print(f"  ⚠️ [WARNING] ofi_ratio is ALL ZEROS! Cross-sectional demean may have failed.")

    ofi_df = ofi_df.with_columns((pl.col("day").dt.year() * 100 + pl.col("day").dt.month()).alias("month_id"))

    print("💾 Writing partitioned PIT monthly parquets to disk...")
    for ym in missing_ymonths:
        out_path = f"{FEATURE_DIR}/hf_{ym}.parquet"
        valid_sids_for_month = universe_sids.get(ym, [])
        
        if not valid_sids_for_month:
            print(f"{ym} no legal sids")
            continue
            
        month_df = ofi_df.filter(
            (pl.col("month_id") == ym) & 
            (pl.col("sid").is_in(valid_sids_for_month)) 
        ).drop("month_id")
        
        if month_df.height > 0:
            month_df.write_parquet(out_path) # sink_parquet(out_path) but not supported with window func
            paths.append(out_path)
            print(f"Saved Strict PIT feature: {out_path}")

    return sorted(list(set(paths)))


# ==============================================================================
# Node 3: OOS Decay
# ==============================================================================
# @task(name="Node_Check_Decay") 
def node_check_decay_monthly(
    prev_model_id: int, 
    prev_oos_yms: list[int], 
    dret_path: str, 
    prev_oos_paths: list[str], 
    common_config: dict
) -> bool:
    """
    - prev_model_id: e.g. 201007
    - prev_oos_yms: e.g. [201007, ..., 201012])
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
        
    panel_lf = build_fsm_panel(pl.concat(aligned_lfs), pl.scan_parquet(dret_path), model_ckpt["config"], common_config)
    panel_df = panel_lf.collect(streaming=True)

    curves_2d = prepare_curves(panel_df, model_ckpt["config"], common_config)
    result = evaluate_and_build_fsm(
        panel_df, curves_2d, model_ckpt["motif"], model_ckpt["config"], common_config
    )

    u_pval = result.get("u_pval", 1.0) 
    if result.get("status") == "success":
        print(f" {prev_model_id} on ({prev_oos_yms[0]}-{prev_oos_yms[-1]}) effective and (P-val: {u_pval:.4f})")
        return False
        
    print(f"🔄 last {prev_model_id} decay (P-val: {u_pval:.4f}) trigger retune")
    return True


# ==============================================================================
# Node 4a: Train Data Preparation accoicated with common_config
# ==============================================================================
def node_prepare_train_data(dret_path: str, train_paths: list[str], common_config: dict) -> dict | None:
    """
    - scan_parquet + collect + ray.put ---> Plasma
    - ObjectRef + lazy frame
    - [FIX P1-P2] Pre-compute static panel ONCE (targets/z-scores/gap),
      so HPO trials only pay for downsample+curve per-trial.
    """
    lazy_frames = [pl.scan_parquet(p) for p in train_paths if os.path.exists(p)]
    if not lazy_frames:
        return None

    hf_lazy = pl.concat(lazy_frames)
    dret_lazy = pl.scan_parquet(dret_path).select(["day", "sid", "close"])

    # [FIX P1-P2] Pre-compute static panel (targets + z-scores) ONCE.
    # This is tune_config-independent, so 500 trials no longer repeat it.
    try:
        static_lazy = build_static_panel(hf_lazy, dret_lazy, common_config, is_train=True)
    except Exception as e:
        print(f"💥 Static panel pre-compute failed: {str(e)}")

    try:
        hf_pa = hf_lazy.collect().to_arrow()
        dret_pa = dret_lazy.collect().to_arrow()
        static_pa = static_lazy.collect().to_arrow()
    except Exception as e:
        print(f"💥 OOM or Engine Error during Polars collect: {str(e)}")
        return None

    return {
        "hf_ref": ray.put(hf_pa),
        "dret_ref": ray.put(dret_pa),
        "static_ref": ray.put(static_pa),
        "hf_lazy": hf_lazy,
        "dret_lazy": dret_lazy,
    }


# ==============================================================================
# Node 4b: Ray Tune Trainable + HPO accoicated with tune_config
# ==============================================================================
def trainable_fsm_worker(config, hf_ref, dret_ref, static_ref, common_config):
    """Ray worker trial and reuse_actors=True"""
    hf_lf = pl.from_arrow(hf_ref).lazy()
    dret_lf = pl.from_arrow(dret_ref).lazy()
    static_lf = pl.from_arrow(static_ref).lazy()

    # [FIX P1-P2] Reuse pre-computed static panel; only pay for curves per-trial.
    # manual build_fsm_panel 
    curve_lf = extract_curves_from_panel(hf_lf, dret_lf, config, common_config)
    panel_lf = curve_lf.join(static_lf, on=["day", "sid"], how="inner")

    result = discover_fsm_pattern(panel_lf, config, common_config)

    print("Tune Reason :", result.get("reason", "Unknown")) 
    tune.report({
        "metrics_score": result["metrics_score"], 
        "u_pval": result.get("u_pval", 1.0), 
        "trigger_count": result.get("trigger_count", 0),
        "valid_sample_ratio": result.get("valid_sample_ratio", 0.0),
        "autocorr": result.get("autocorr", 0.0),
    })

    del panel_lf, curve_lf, static_lf, hf_lf, dret_lf
    gc.collect()


# @task(name="Node_Tune") 
def node_tune_monthly(prev_model_id: str, model_id: int, train_data: dict, exp_config: dict) -> bool:
    """
        node_prepare_train_data ---> tune_config
    """
    common_config, search_config = exp_config["common_params"], exp_config["search_bounds"]

    hf_ref, dret_ref, static_ref = train_data["hf_ref"], train_data["dret_ref"], train_data["static_ref"]
    hf_lazy, dret_lazy = train_data["hf_lazy"], train_data["dret_lazy"]

    # =========================================================================
    # Ray search and Opt algo
    # =========================================================================

    search_space = {
        "downsample": tune.choice(search_config["downsample"]), 
        "motif_minutes": tune.choice(search_config["motif_minutes"]), 
        "threshold_r": tune.uniform(*search_config["threshold_r"]),
    }

    points_to_evaluate = None
    prev_model_path = f"{MODEL_DIR}/model_{prev_model_id}.pkl"  
    if os.path.exists(prev_model_path):
        print(f"Prior model found, using as point to evaluate...")
        with open(prev_model_path, "rb") as f:
            prior_cfg = pickle.load(f)["config"]
            points_to_evaluate=[{k: prior_cfg[k] for k in search_space if k in prior_cfg}]

    # multithread / sample optimize
    optuna_sampler = optuna.samplers.TPESampler(
        n_startup_trials=common_config["n_startup_trials"], 
        multivariate=True 
    ) 

    search_alg = OptunaSearch(
        sampler=optuna_sampler,
        points_to_evaluate=points_to_evaluate
    )

    # =========================================================================
    # Ray Tune and Initialize MLflowLoggerCallback (default 5000)
    # =========================================================================
    wrapped_trainable = tune.with_resources(
        tune.with_parameters(
            trainable_fsm_worker,
            hf_ref=hf_ref,      
            dret_ref=dret_ref,   
            static_ref=static_ref,
            common_config=common_config),
        resources={"cpu": 1, "gpu": 0} 
    )

    mlflow_callback = MLflowLoggerCallback(
        tracking_uri=mlflow.get_tracking_uri(), 
        experiment_name="FSM_Production_Models",
        save_artifact=False  
    )

    tuner = tune.Tuner(
        wrapped_trainable, 
        param_space=search_space,   
        tune_config=tune.TuneConfig(
            metric="metrics_score", 
            mode="max", # bic_store direction 
            search_alg=search_alg, 
            num_samples=search_config["num_trials"],        
            # # used for Iterative Training not for One-shot / Single-step Computation    
            # scheduler=tune.schedulers.ASHAScheduler(grace_period=search_config["grace_period"], reduction_factor=search_config["grace_factor"]),
            max_concurrent_trials=common_config["num_workers"],
            # due to heavy numba / stumpy and reuse to increase cpu usage
            reuse_actors=True,
        ),
        run_config=tune.RunConfig(
            name=f"fsm_hpo_{model_id}", 
            storage_path=common_config["storage_path"],
            callbacks=[mlflow_callback]
            )  
        )
    
    # =========================================================================
    # Ray Tune Fit and Optuna.importance
    # =========================================================================
    results = tuner.fit() 
    # best_trial_result = results.get_best_result("metrics_score", "max")
    print("\n[Ray Tune] All Trials Completed. Analyzing Results...\n")

    # =========================================================================
    # filter by P-val and metrics_score
    # =========================================================================
    df_results = pl.from_pandas(results.get_dataframe()) # abandon dict fsm_matrix

    if df_results.height == 0:
        print(f"⚠️ [Failed] {model_id} df_results height 0")
        return False

    max_pval =0.05 # common_config.get("u_pval", 0.2)
    min_triggers = common_config.get("trigger", 30)
    
    stats_valid_trials = df_results.filter(
        (pl.col("u_pval") <= max_pval) & 
        (pl.col("trigger_count") >= min_triggers)
    )
    if stats_valid_trials.height == 0:
        print(f"⚠️ [Failed] {model_id} found P-val <= {max_pval} and reach bottom triggers")
        return False

    best_valid_row = stats_valid_trials.sort("metrics_score", descending=True).row(0, named=True)
    best_valid_score = best_valid_row["metrics_score"]
    best_valid_config = {k.replace("config/", ""): v for k, v in best_valid_row.items() if k.startswith("config/")}

    # fANOVA 
    is_plateau = validate_parameter_plateau_fanova(
        df_results, best_valid_config, best_valid_score
    )
    if not is_plateau:
        print(f"❌ {model_id} isolated")
        return False

    # pareto front
    pareto_front_df = find_pareto_front(stats_valid_trials, common_config)
    best_model_dict = select_best_model_from_pareto(pareto_front_df)
    
    if not best_model_dict:
        print(f"❌ NotFound best_model_dict for {model_id} from pareto front") 
        return False
        
    print(f"✅ {model_id} Score: {best_model_dict['metrics_score']:.1f}")
    
    # =========================================================================
    # Model Save
    # =========================================================================
    # Ray ResultGrid ---> Result
    best_trial_id = best_model_dict["trial_id"]
    best_ray_result = None

    for r in results:
        if r.metrics.get("trial_id") == best_trial_id:
            best_ray_result = r
            break

    if best_ray_result is None:
        raise KeyError(f"Failed to locate original Ray Result for trial_id: {best_trial_id}")
    
    # trial config
    best_config = {k.replace("config/", ""): v for k, v in best_model_dict.items() if k.startswith("config/")}
    if "m" not in best_config:
        best_config["m"] = int(best_config["motif_minutes"] // best_config["downsample"])
    if "threshold_d" not in best_config:
        best_config["threshold_d"] = float(np.sqrt(2 * best_config["m"] * (1.0 - best_config["threshold_r"])))

    # rerun avoid tune.report
    final_panel_lf = build_fsm_panel(hf_lazy, dret_lazy, best_config, common_config)
    final_result = discover_fsm_pattern(final_panel_lf, best_config, common_config)
    
    model_ckpt = {
        "config": best_config, 
        "motif": np.array(final_result.get("learned_motif", [])),       
        "fsm_matrix": final_result.get("fsm_matrix", {}),             
        "valid_month": model_id 
    }
    
    pkl_path = f"{MODEL_DIR}/model_{model_id}.pkl"
    with open(pkl_path, "wb") as f: 
        pickle.dump(model_ckpt, f)
    return True

# ==============================================================================
# Node 5: Update Fsm Matrix While Retain Motif
# ==============================================================================

def node_update_fsm_matrix(model_id: int, prev_model_id: int, dret_path: str, train_paths: list[str], common_config: dict) -> bool:
    prev_model_path = f"{MODEL_DIR}/model_{prev_model_id}.pkl"

    if not os.path.exists(prev_model_path):
        return False
        
    with open(prev_model_path, "rb") as f:
        pre_model_ckpt = pickle.load(f)

    prev_tune_config, prev_motif = pre_model_ckpt["config"], pre_model_ckpt["motif"]
    hf_lfs = [pl.scan_parquet(p) for p in train_paths if os.path.exists(p)]
    if not hf_lfs: return False
    
    panel_df = build_fsm_panel(pl.concat(hf_lfs), pl.scan_parquet(dret_path), prev_tune_config, common_config).collect(streaming=True)
    curves = prepare_curves(panel_df, prev_tune_config, common_config) 

    result = evaluate_and_build_fsm(panel_df, curves, prev_motif, prev_tune_config, common_config, skip_stats=True)
    if result["status"] != "success":
        print(f"⚠️ {model_id} matrix update failed due to ({result.get('reason')})")
        return False
        
    print(f"✅ {model_id} matrix has update by last 12month")
    new_ckpt = {
        "config": prev_tune_config, 
        "motif": prev_motif, 
        "fsm_matrix": result["fsm_matrix"], 
        "valid_month": model_id
    }
    with open(f"{MODEL_DIR}/model_{model_id}.pkl", "wb") as f: 
        pickle.dump(new_ckpt, f)
    return True

# ==============================================================================
# Node 6: OOS 
# ==============================================================================

# @task(name="Node_OOS_Inference") 
def node_oos_inference_monthly(
    model_id: int, 
    dret_path: str, 
    oos_months: list[int], 
    oos_paths: list[str], 
    warmup_paths: list[str], # add warm month
    common_config: dict
):
    model_path = f"{MODEL_DIR}/model_{model_id}.pkl"
    if not os.path.exists(model_path):
        print(f"⚠️ {model_id} NotFound and Skip OOS Inference")
        return
        
    with open(model_path, "rb") as f:
        model_ckpt = pickle.load(f)
        
    # ensure rolling_quantile avoid nan
    all_paths = warmup_paths + oos_paths
    if not all_paths:
        print(f"⚠️ {model_id} no avaiable warmup/oos feature and skip OOS")
        return
    aligned_lfs = [pl.scan_parquet(p) for p in all_paths if os.path.exists(p)]
    if not aligned_lfs: return
    
    panel_lf = build_fsm_panel(pl.concat(aligned_lfs), pl.scan_parquet(dret_path), model_ckpt["config"], common_config, is_train=False)
    scored_df = FSMPredictor(model_ckpt, common_config).predict(panel_lf)
    
    if scored_df.height > 0:
        # extract warm ym and filter
        match = re.search(r"hf_(\d{6})", warmup_paths[0])
        if match:
            warmup_ym = int(match.group(1)) # group(0) ---> hf201206 / group(1) --> \d{6}
        else:
            raise ValueError(f"Cannot parse YYYYMM: {warmup_paths[0]}")
        
        scored_df = scored_df.filter(
            (pl.col("day").dt.year() * 100 + pl.col("day").dt.month()) != warmup_ym
        )
        
        if scored_df.height > 0:
            out_path = f"{SCORE_DIR}/scores_{model_id}.parquet"
            scored_df.write_parquet(out_path)
            print(f"✅ {model_id} {oos_months[0]} between {oos_months[-1]} Oss: {scored_df.height} ")

# ==============================================================================
# DAG (Walk-Forward)
# ==============================================================================

# @flow(name="WFO_FSM_Pipeline")
@consume_time
def wfo_pipeline(exp_config):
    common_config = exp_config["common_params"]     

    # =========================================================================
    # Ray init: restricted only in worker
    # =========================================================================
    runtime_env = {
        "env_vars": {
            "POLARS_MAX_THREADS": "1",
            "RAYON_NUM_THREADS": "1",       
            "NUMBA_NUM_THREADS": "1",       
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            # "GRPC_ENABLE_FORK_SUPPORT": "0", # set in global
        },
        # "working_dir": os.path.dirname(os.path.abspath(__file__)) 
    }
    ray.init(num_cpus=common_config["num_workers"], runtime_env=runtime_env, ignore_reinit_error=True)

    # Setup Macro
    global_data = node_prepare_macro(common_config)
    daily_lazy = pl.scan_parquet(global_data["dret_path"]).select(["day"])
    month_id_expr = (pl.col("day") // 100).cast(pl.Int32).alias("month_id")

    yms = (
        daily_lazy
        .select(month_id_expr)
        .unique()                 
        .sort("month_id")          
        .collect()                 
        .get_column("month_id")    
        .to_list()
    )

    # Setup Walkforward
    TRAIN_WINDOW = common_config["train_window"]     
    STEP = common_config["oss_step"]     
    last_available_model_id = None

    for idx in range(TRAIN_WINDOW, len(yms), STEP):

        train_yms = yms[idx - TRAIN_WINDOW : idx]            
        oos_yms = yms[idx : min(idx + STEP, len(yms))]
        warmup_month = [train_yms[-1]] # used for oss cold start
        
        model_id = oos_yms[0]           
        
        print(f"\n===========================================================================")
        print(f"📅 WFO | OOS : {model_id} | Train Month: {train_yms[0]}-{train_yms[-1]} ")
        print(f"============================================================================\n")
        
        train_paths = node_extract_feature_monthly(train_yms, global_data["universe_sids"], common_config)
        oos_paths = node_extract_feature_monthly(oos_yms, global_data["universe_sids"], common_config)
        warmup_paths = node_extract_feature_monthly(warmup_month, global_data["universe_sids"], common_config)

        if last_available_model_id is None:
            is_decayed = True
            print("🚀 First run (Cold Start), forcing HPO Training...")
        else:
            prev_oos_yms = train_yms[-STEP:] 
            prev_oos_paths = node_extract_feature_monthly(prev_oos_yms, global_data["universe_sids"], common_config)
            
            is_decayed = node_check_decay_monthly(
                last_available_model_id, prev_oos_yms, global_data["dret_path"], prev_oos_paths, common_config
            )
        if is_decayed:
            print(f"🔄 Model decayed or First Run. Tuning Model for {model_id}...")

            train_data = node_prepare_train_data(global_data["dret_path"], train_paths, common_config)
            if train_data is None:
                print(f"⚠️ {model_id} HPO Failed: No valid training parquet files found.")
                continue

            success = node_tune_monthly(last_available_model_id, model_id, train_data, exp_config)
            if not success:
                print(f"⚠️ {model_id} HPO Failed. Skipping this window.")
                continue
            last_available_model_id = model_id 
        else:
            print(f"🌲 Model {last_available_model_id} remains effective. Inheriting to {model_id}")
            success = node_update_fsm_matrix(model_id, last_available_model_id, global_data["dret_path"], train_paths, common_config)
            if not success:
                print(f"Update Matrix Failure and Direct inherit pre model_ckpt...")
                shutil.copy(f"{MODEL_DIR}/model_{last_available_model_id}.pkl", f"{MODEL_DIR}/model_{model_id}.pkl")
 
            last_available_model_id = model_id

        node_oos_inference_monthly(model_id, global_data["dret_path"], oos_yms, oos_paths, warmup_paths, common_config)


if __name__ == "__main__":

    load_dotenv()

    exp_config = {
        "common_params": {
            "start_date": 20100101, "end_date": 20201231, "benchmark": "1A0001",

            # sample
            "top_k_ratio": 0.25, # used for sample
            "days_since_ipo": 120, # days since ipo

            # train / oss 
            "train_window": 12,
            "oss_step": 6,

            # indicator
            "regime_filter": {
                "ma_window": 20
            },

            # fut_ret T+1
            "T1_rets": {
                "open_5m":  5,   
                "open_15m": 15,  
                "open_30m": 30,  
            },
            "decay_minutes": 15, # minute calculate weight
            
            # ofi
            "eps": 1e-8,
            "min_factor_weight": 0.05,      

            # stumpy curves and overlap for stumpy  
            "exclude_bars": 10, # 14:50
            "dtw_window_frac": 0.1, 

            # macro ranking state and fut_ret rank state
            "ranking_window": 5, # rolling macro_state 
            "ranking_ratio": 0.25, # ranking

            "topk": 5, # candidate
            "trigger": 30, 

             # stats
            "alternative": "greater",
            "u_pval": 0.10, # 0.05 too strict and least

            # concurrency
            "num_workers": 12,
            "n_startup_trials": 40,

            # "fanova_min_samples": 20,     
            # "fanova_neighbor_ratio": 0.15,
            # "fanova_drop_iqr_limit": 1.5, # 1.5 IQR
            
            # "complexity_weights": {
            #     "offset_scale": 60.0,     
            #     "dtw_scale": 10.0,        
            #     "motif_scale": 30.0,      
            # },
            
            # "gap_z_lower": -2.0, # 2 MAD 
            # "gap_z_upper": 3.0,  # 3 MAD 

            "seed": 42, # for reproducibility 

            "storage_path": "/tmp/ray_results", # for Ray Tune
        },

        "search_bounds": {
            "downsample": [3, 4, 5], # downsample for DTW
            "motif_minutes": [30, 45, 60, 90], # used from motif length intraday
            "threshold_r": [0.65, 0.90], 
            "num_trials": 800, 
        }
    }

    # Setup Dir
    BASE_DIR = "/Users/hengxinliu/startup/bt_studio/result/fsm"
    MODEL_DIR = f"{BASE_DIR}/models"
    FEATURE_DIR = f"{BASE_DIR}/features"  
    SCORE_DIR = f"{BASE_DIR}/scores"     

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FEATURE_DIR, exist_ok=True)
    os.makedirs(SCORE_DIR, exist_ok=True)

    wfo_pipeline(exp_config)