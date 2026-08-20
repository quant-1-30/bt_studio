#! /usr/bin/env python3
from __future__ import annotations

import gc
import os
import pickle
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
import ray
from ray import train, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune.search.optuna import OptunaSearch

import optuna
import mlflow

from bt_studio.constant import (
    BASE_MARKET_COLS,
    FEATURE_DIR,
    TUNE_BASE_DIR,
    TUNE_MODEL_DIR,
    TUNE_SCORE_DIR,
    TUNE_COLLAPSE_DIR,
)
from bt_studio.compiler.ast import compile_ast, compile_recipe
from bt_studio.pipeline.inference import FSMPredictor
from bt_studio.pipeline.metrics import (
    find_pareto_front,
    select_best_model_from_pareto,
    validate_parameter_plateau_fanova,
)
from bt_studio.pipeline.patterns import (
    discover_fsm_pattern,
    evaluate_and_build_fsm,
    prepare_curves,
)
from bt_studio.pipeline.preprocess import (
    build_fsm_panel,
    build_static_panel,
    extract_curves_from_panel,
    prepare_macro,
    universe_sample,
    prepare_tick,
)
from bt_studio.utils.diagnostics.hpo_check import run_collapse_check
from bt_studio.utils.months import paths_for_months
from bt_studio.utils.io import atomic_save_parquet, atomic_save_pickle

# ==============================================================================
# Date and Int32 
# ==============================================================================

def _expr_month_id(col_name: str = "day") -> pl.Expr:
    """
        Int32 YYYYMM ---> Date, Datetime, Int32/Int64 (20210101) and  Utf8 ('2021-01-01')
    """
    col = pl.col(col_name)
    # fixbug SafeExpr
    return (
        pl.coalesce([
            # 1. Date/Datetime
            col.dt.year() * 100 + col.dt.month(),
            # 2. YYYYMMDD -> YYYYMM
            col.cast(pl.Int64) // 100,
            # 3. 'YYYY-MM-DD' -> YYYYMM
            col.cast(pl.Utf8).str.replace_all("-", "").str.slice(0, 6).cast(pl.Int64),
        ])
        .cast(pl.Int32)
        .alias("_month_id")
    )


# ==============================================================================
# Node 1: Macro & Universe 
# ==============================================================================

def node_prepare_macro(common_config: dict, warm: int = 10000) -> Dict[str, Any]:
    os.makedirs(TUNE_BASE_DIR, exist_ok=True)
    dret_path = f"{TUNE_BASE_DIR}/global_daily.parquet"

    if os.path.exists(dret_path):
        print(f"✅ [Cache Hit] Daily Universe: {dret_path}")
    else:
        print("🚀 [Cache Miss] generate daily ret...")
        universe_lf, daily_lf = prepare_macro(
            start_date=common_config["start_date"] - warm,
            end_date=common_config["end_date"],
            benchmark=str(common_config["benchmark"]).encode(),
        )
        filtered_uni_lf = universe_sample(universe_lf, daily_lf, common_config)
        filtered_uni_lf.sink_parquet(dret_path)

    final_scan_lf = pl.scan_parquet(dret_path)
    schema = final_scan_lf.collect_schema().names()
    date_col = "day" if "day" in schema else "date"

    dynamic_sids_by_month = dict(
        final_scan_lf.select([
            _expr_month_id(date_col).alias("month_id"),
            pl.col("sid"),
        ])
        .group_by("month_id")
        .agg(pl.col("sid").unique())
        .collect(engine="streaming")
        .iter_rows()
    )
    return {"dret_path": dret_path, "universe_sids": dynamic_sids_by_month}


# ==============================================================================
# Node 2: Feature Extraction & PIT Partitioning
# ==============================================================================

def _prepare_tick_for_months(missing_ymonths: List[int], universe_sids: Dict[int, List[Any]]) -> Optional[pl.LazyFrame]:
    start_date = min(missing_ymonths) * 100 + 1
    end_date = max(missing_ymonths) * 100 + 31

    sids = []
    for ym in missing_ymonths:
        sids.extend(universe_sids.get(ym, []))
    sids = list(set(sids))
    if not sids:
        return None

    tick_dict = prepare_tick(start_date, end_date, sids, warm=0)

    lazy_frames = []
    for sid, lf in tick_dict.items():
        if "minute_idx" in lf.collect_schema().names():
            lf = lf.rename({"minute_idx": "bar_idx"})
        lazy_frames.append(lf)
        
    if not lazy_frames:
        return None

    return pl.concat(lazy_frames)


def node_extract_feature_monthly(
    ymonths: List[int],
    universe_sids: Dict[int, List[Any]],
    common_config: dict,
    ast_recipe: Union[List[dict], dict],
    feature_col: Optional[str] = None,
    raw_hf_lf: Optional[pl.LazyFrame] = None,
) -> List[str]:
    if not ymonths:
        return []

    os.makedirs(FEATURE_DIR, exist_ok=True)
    generated_paths: List[str] = []
    missing_ymonths: List[int] = []

    # fixbug to determin real feature_col
    if feature_col is None:
        feature_col = common_config.get("feature_col")

    compiled_exprs = None
    if feature_col is None:
        if isinstance(ast_recipe, list):
            compiled_exprs = compile_recipe(ast_recipe, check_causal=True)
            feature_col = ast_recipe[-1].get("name", f"recipe_feat")
        else:
            single_expr, inferred_name = compile_ast(ast_recipe, check_causal=True)
            feature_col = inferred_name
            compiled_exprs = [single_expr.alias(feature_col)]

    # check cache
    for ym in ymonths:
        p = f"{FEATURE_DIR}/hf_{feature_col}_{ym}.parquet"
        if os.path.exists(p):
            generated_paths.append(p)
        else:
            missing_ymonths.append(ym)

    if not missing_ymonths:
        return sorted(list(set(generated_paths)))

    if raw_hf_lf is None:
        raw_hf_lf = _prepare_tick_for_months(missing_ymonths, universe_sids)
        if raw_hf_lf is None:
            return sorted(list(set(generated_paths)))

    combined_lf = raw_hf_lf
    if compiled_exprs is not None:
        for e in compiled_exprs:
            combined_lf = combined_lf.with_columns(e)
    elif isinstance(ast_recipe, list):
        for e in compile_recipe(ast_recipe, check_causal=True):
            combined_lf = combined_lf.with_columns(e)
    else:
        expr, _ = compile_ast(ast_recipe, check_causal=True)
        combined_lf = combined_lf.with_columns(expr.alias(feature_col))

    schema_names = combined_lf.collect_schema().names()
    base_cols = [c for c in BASE_MARKET_COLS if c in schema_names]
    selected_cols = list(dict.fromkeys(base_cols + [feature_col]))

    feat_df = (
        combined_lf.select(selected_cols)
        .with_columns(_expr_month_id("day").alias("month_id"))
        .collect(engine="streaming")
    )

    if feat_df.height == 0:
        return sorted(list(set(generated_paths)))

    for ym in missing_ymonths:
        out_path = f"{FEATURE_DIR}/hf_{feature_col}_{ym}.parquet"
        valid_sids = universe_sids.get(ym, [])
        if not valid_sids:
            continue

        month_df = (
            feat_df.filter((pl.col("month_id") == ym) & (pl.col("sid").is_in(valid_sids)))
            .drop("month_id")
        )

        if month_df.height > 0:
            atomic_save_parquet(month_df, out_path)
            generated_paths.append(out_path)
            print(f"  [Persist PIT] -> {out_path}")

    return sorted(list(set(generated_paths)))


# ==============================================================================
# Node 3: OOS Decay Check
# ==============================================================================

def node_check_decay_monthly(
    prev_model_id: Optional[int],  # train_month[-1]
    prev_oos_yms: List[int],
    dret_path: str,
    train_paths: List[str],
    common_config: dict,
) -> bool:
    if prev_model_id is None:
        print(f"Cold start (prev_model_id is None) -> Trigger Tune")
        return True

    prev_model_path = f"{TUNE_MODEL_DIR}/model_{prev_model_id}.pkl"
    if not os.path.exists(prev_model_path):
        print(f"NotFound: {prev_model_id} -> Trigger Tune")
        return True

    try:
        with open(prev_model_path, "rb") as f:
            model_ckpt = pickle.load(f)
    except Exception as e:
        print(f"Parse Model Error: {e} -> Trigger Tune")
        return True

    prev_oos_paths = paths_for_months(train_paths, prev_oos_yms)
    aligned_lfs = [pl.scan_parquet(p) for p in prev_oos_paths if os.path.exists(p)]
    
    if not aligned_lfs:
        print(f"⚠️ OOS ({prev_oos_yms}) NotFound -> Trigger Tune")
        return True

    panel_lf = build_fsm_panel(
        pl.concat(aligned_lfs),
        pl.scan_parquet(dret_path),
        model_ckpt["config"],
        common_config,
        is_train=False,
    )
    panel_df = panel_lf.collect(engine="streaming")

    curves_2d = prepare_curves(panel_df, model_ckpt["config"], common_config)
    result = evaluate_and_build_fsm(
        panel_df, curves_2d, model_ckpt["motif"], model_ckpt["config"], common_config
    )

    if result.get("status") != "success":
        print(f"Model {prev_model_id} in OOS failed evaluation -> Trigger Retune")
        return True

    u_pval = result.get("u_pval", 1.0)
    max_pval = common_config.get("u_pval", 0.05)
    if u_pval <= max_pval:
        print(f"Model {prev_model_id} healthy in OOS ({prev_oos_yms[0]}-{prev_oos_yms[-1]}) (P-val: {u_pval:.4f} <= {max_pval})")
        return False

    print(f"Model {prev_model_id} decay (P-val: {u_pval:.4f} > {max_pval}) -> Trigger Retune")
    return True


# ==============================================================================
# Node 4a: Train Data Preparation
# ==============================================================================

def node_prepare_train_data(
    dret_path: str,
    train_paths: List[str],
    common_config: dict,
) -> Optional[Dict[str, Any]]:
    lazy_frames = [pl.scan_parquet(p) for p in train_paths if os.path.exists(p)]
    if not lazy_frames:
        return None

    hf_lazy = pl.concat(lazy_frames)
    dret_lazy = pl.scan_parquet(dret_path).select(["day", "sid", "close"])

    try:
        static_lazy = build_static_panel(hf_lazy, dret_lazy, common_config, is_train=True)
        hf_pa = hf_lazy.collect(engine="streaming").to_arrow()
        dret_pa = dret_lazy.collect(engine="streaming").to_arrow()
        static_pa = static_lazy.collect(engine="streaming").to_arrow()
    except Exception as e:
        print(f"Train Dataset Preparation Failure: {e}")
        return None

    return {
        "hf_ref": ray.put(hf_pa),
        "dret_ref": ray.put(dret_pa),
        "static_ref": ray.put(static_pa),
        "hf_lazy": hf_lazy,
        "dret_lazy": dret_lazy,
    }


# ==============================================================================
# Node 4b: Ray Tune Trainable & HPO
# ==============================================================================

def trainable_fsm_worker(config: dict, hf_ref: Any, dret_ref: Any, static_ref: Any, common_config: dict):
    trial_cfg = config.copy()

    downsample = max(1, int(trial_cfg["downsample"]))
    motif_minutes = int(trial_cfg["motif_minutes"])
    threshold_r = float(np.clip(trial_cfg["threshold_r"], 0.0, 0.9999))

    m = int(motif_minutes // downsample)
    threshold_d = float(np.sqrt(2 * m * (1.0 - threshold_r)))

    trial_cfg["m"] = m
    trial_cfg["threshold_d"] = threshold_d

    hf_lf = pl.from_arrow(hf_ref).lazy()
    dret_lf = pl.from_arrow(dret_ref).lazy()
    static_lf = pl.from_arrow(static_ref).lazy()

    curve_lf = extract_curves_from_panel(hf_lf, dret_lf, trial_cfg, common_config)
    panel_lf = curve_lf.join(static_lf, on=["day", "sid"], how="inner")

    result = discover_fsm_pattern(panel_lf, trial_cfg, common_config)

    train.report({
        "metrics_score": result.get("metrics_score", -500.0),
        "u_pval": result.get("u_pval", 1.0),
        "trigger_count": result.get("trigger_count", 0),
        "valid_sample_ratio": result.get("valid_sample_ratio", 0.0),
        "autocorr": result.get("autocorr", 0.0),
    })

    del panel_lf, curve_lf, static_lf, hf_lf, dret_lf, trial_cfg
    gc.collect()


def node_tune_monthly(
    prev_model_id: Optional[Union[int, str]],
    model_id: int,  # train_months[-1] 
    train_data: dict,
    exp_config: dict,
) -> Optional[int]:
    os.makedirs(TUNE_MODEL_DIR, exist_ok=True)

    # fixbug None  
    common_config = exp_config.get("common_config") or exp_config.get("common_params", {})
    search_config = exp_config.get("search_bounds") or exp_config.get("search_config", {})

    hf_ref, dret_ref, static_ref = train_data["hf_ref"], train_data["dret_ref"], train_data["static_ref"]
    hf_lazy, dret_lazy = train_data["hf_lazy"], train_data["dret_lazy"]

    r_bounds = search_config.get("threshold_r", [0.55, 0.85])
    search_space = {
        "downsample": tune.choice(search_config.get("downsample", [3, 4, 5])),
        "motif_minutes": tune.choice(search_config.get("motif_minutes", [30, 45, 60])),
        "threshold_r": tune.uniform(r_bounds[0], r_bounds[-1]) if isinstance(r_bounds, (list, tuple)) and len(r_bounds) >= 2 else tune.choice(r_bounds),
    }

    # Prior Warm Start
    points_to_evaluate = None
    if prev_model_id:
        prev_model_path = f"{TUNE_MODEL_DIR}/model_{prev_model_id}.pkl"
        if os.path.exists(prev_model_path):
            try:
                with open(prev_model_path, "rb") as f:
                    prior_cfg = pickle.load(f)["config"]
                    points_to_evaluate = [{k: prior_cfg[k] for k in search_space if k in prior_cfg}]
            except Exception:
                points_to_evaluate = None

    optuna_sampler = optuna.samplers.TPESampler(
        n_startup_trials=common_config.get("n_startup_trials", 20),
        multivariate=True,
        seed=common_config.get("seed", 42),
    )

    search_alg = OptunaSearch(
        sampler=optuna_sampler,
        points_to_evaluate=points_to_evaluate,
    )

    wrapped_trainable = tune.with_resources(
        tune.with_parameters(
            trainable_fsm_worker,
            hf_ref=hf_ref,
            dret_ref=dret_ref,
            static_ref=static_ref,
            common_config=common_config,
        ),
        resources={"cpu": 1, "gpu": 0},
    )

    mlflow_callback = None
    try:
        import requests
        _mlflow_uri = mlflow.get_tracking_uri()
        if _mlflow_uri and _mlflow_uri.startswith("http"):
            requests.get(_mlflow_uri, timeout=1.5)
            mlflow_callback = MLflowLoggerCallback(
                tracking_uri=_mlflow_uri,
                experiment_name="FSM_Production_Models",
                save_artifact=False,
            )
    except Exception as _e:
        pass

    num_trials = common_config.get("num_trials") or search_config.get("num_trials", 300)

    tuner = tune.Tuner(
        wrapped_trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="metrics_score",
            mode="max",
            search_alg=search_alg,
            num_samples=num_trials,
            max_concurrent_trials=common_config.get("num_workers", 8),
            reuse_actors=True,
        ),
        run_config=tune.RunConfig(
            name=exp_config.get("run_name", f"fsm_hpo_{model_id}"),
            storage_path=common_config.get("storage_path", "/tmp/ray_results"),
            callbacks=[mlflow_callback] if mlflow_callback else [],
        ),
    )

    results = tuner.fit()
    df_results = pl.from_pandas(results.get_dataframe())

    # Space Collapse Gating
    is_healthy, verdict, _ = run_collapse_check(df_results, exp_config, model_id)
    if not is_healthy:
        print(f"❌ [HPO Gating] Model {model_id} rejected due to {verdict.upper()}")
        return None

    if df_results.height == 0:
        return None

    max_pval = common_config.get("u_pval", 0.05)
    min_triggers = common_config.get("trigger", 30)

    stats_valid_trials = df_results.filter(
        (pl.col("u_pval") <= max_pval) & (pl.col("trigger_count") >= min_triggers)
    )
    if stats_valid_trials.height == 0:
        print(f"⚠️ [Failed] {model_id} zero valid statistical samples")
        return None

    best_valid_row = stats_valid_trials.sort("metrics_score", descending=True).row(0, named=True)
    best_valid_score = best_valid_row["metrics_score"]
    best_valid_config = {k.replace("config/", ""): v for k, v in best_valid_row.items() if k.startswith("config/")}

    # fANOVA Stability Check
    is_plateau = validate_parameter_plateau_fanova(df_results, best_valid_config, best_valid_score)
    if not is_plateau:
        print(f"❌ {model_id} fANOVA spike detected")
        return None

    # Pareto
    pareto_front_df = find_pareto_front(stats_valid_trials, common_config)
    best_model_dict = select_best_model_from_pareto(pareto_front_df)
    if not best_model_dict:
        return None

    best_config = {k.replace("config/", ""): v for k, v in best_model_dict.items() if k.startswith("config/")}
    if "m" not in best_config:
        best_config["m"] = int(best_config["motif_minutes"] // best_config["downsample"])
    if "threshold_d" not in best_config:
        best_config["threshold_d"] = float(np.sqrt(2 * best_config["m"] * (1.0 - best_config["threshold_r"])))

    # Motif
    final_panel_lf = build_fsm_panel(hf_lazy, dret_lazy, best_config, common_config, is_train=True)
    final_result = discover_fsm_pattern(final_panel_lf, best_config, common_config)

    raw_motif = final_result.get("learned_motif", [])
    motif_arr = np.asarray(raw_motif, dtype=np.float64) if raw_motif else np.empty((0, 0), dtype=np.float64)

    model_ckpt = {
        "config": best_config,
        "motif": motif_arr,
        "fsm_matrix": final_result.get("fsm_matrix", {}),
        "train_end_month": model_id,
        "feature_col": common_config.get("feature_col"),
    }

    pkl_path = f"{TUNE_MODEL_DIR}/model_{model_id}.pkl"
    atomic_save_pickle(model_ckpt, pkl_path)

    print(f"✅ 模型成功落盘: {pkl_path}")
    return model_id


# ==============================================================================
# Node 5: Matrix Update (Retain Motif)
# ==============================================================================

def node_update_fsm_matrix(
    model_id: int,       
    prev_model_id: int, 
    dret_path: str,
    train_paths: List[str],
    common_config: dict,
) -> Optional[int]:
    prev_model_path = f"{TUNE_MODEL_DIR}/model_{prev_model_id}.pkl"
    if not os.path.exists(prev_model_path):
        return None

    with open(prev_model_path, "rb") as f:
        pre_model_ckpt = pickle.load(f)

    prev_tune_config = pre_model_ckpt["config"]
    prev_motif = pre_model_ckpt["motif"]

    hf_lfs = [pl.scan_parquet(p) for p in train_paths if os.path.exists(p)]
    if not hf_lfs:
        return None

    panel_df = build_fsm_panel(
        pl.concat(hf_lfs),
        pl.scan_parquet(dret_path),
        prev_tune_config,
        common_config,
        is_train=True,
    ).collect(engine="streaming")

    curves = prepare_curves(panel_df, prev_tune_config, common_config)
    result = evaluate_and_build_fsm(
        panel_df, curves, prev_motif, prev_tune_config, common_config, skip_stats=True
    )

    if result.get("status") != "success":
        print(f"⚠️ {model_id} Fsm Matrix Failure: {result.get('reason')}")
        return None

    new_ckpt = {
        "config": prev_tune_config,
        "motif": prev_motif,
        "fsm_matrix": result["fsm_matrix"],
        "train_end_month": model_id,  
        "feature_col": common_config.get("feature_col"),
    }
    pkl_path = f"{TUNE_MODEL_DIR}/model_{model_id}.pkl"
    atomic_save_pickle(new_ckpt, pkl_path)

    print(f"✅ {model_id} inherit {prev_model_id} Motif and Update Fsm Matrix")
    return model_id


# ==============================================================================
# Node 6: OOS Inference
# ==============================================================================

def node_oos_inference_monthly(
    model_id: int,  
    dret_path: str,
    oos_yms: List[int],
    oos_paths: List[str],
    warmup_paths: List[str],
    common_config: dict,
) -> None:
    os.makedirs(TUNE_SCORE_DIR, exist_ok=True)
    model_path = f"{TUNE_MODEL_DIR}/model_{model_id}.pkl"
    if not os.path.exists(model_path):
        print(f"⚠️ NotFound {model_path} and Skip OOS Inference")
        return

    with open(model_path, "rb") as f:
        model_ckpt = pickle.load(f)

    all_paths = [p for p in (warmup_paths + oos_paths) if os.path.exists(p)]
    if not all_paths:
        return

    panel_lf = build_fsm_panel(
        pl.concat([pl.scan_parquet(p) for p in all_paths]),
        pl.scan_parquet(dret_path),
        model_ckpt["config"],
        common_config,
        is_train=False,
    )

    scored_df = FSMPredictor(model_ckpt, common_config).predict(panel_lf)

    if scored_df.height > 0:
        scored_df = (
            scored_df.with_columns(_expr_month_id("day").alias("_month_id"))
            .filter(pl.col("_month_id").is_in(oos_yms))
            .drop("_month_id")
        )

        if scored_df.height > 0:
            out_path = f"{TUNE_SCORE_DIR}/scores_{model_id}.parquet"
            atomic_save_parquet(scored_df, out_path)
            print(f"{model_id} OOS ({oos_yms[0]} ~ {oos_yms[-1]}) Saved to {out_path}, Rows: {scored_df.height}")
