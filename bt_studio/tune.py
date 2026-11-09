#! /usr/bin/env python3

from __future__ import annotations

import gc
import os
import numpy as np
import optuna
import polars as pl
import ray

from ray import tune
from ray.tune.search.optuna import OptunaSearch
from typing import Any, Dict, List, Optional, Union

from bt_studio.compiler.ast import ast_fingerprint, compile_feature_lazy
from bt_studio.constant import (
    DEFAULT_EXCLUDE_BARS,
    DEFAULT_WARMUP_MONTHS,
    FEATURE_DIR,
    TUNE_BASE_DIR,
    TUNE_MODEL_DIR,
    TUNE_SCORE_DIR,
)
from bt_studio.default_config import (
    PRODUCTION_SEARCH_SPEC,
    build_ray_tune_search_space,
    common_params_of,
    create_mlflow_callback,
    resolve_feature_col,
    resolve_warm_start_points,
    search_spec_of,
)
from bt_studio.pipeline.inference import FSMPredictor
from bt_studio.pipeline.metrics import select_plateau_verified_config
from bt_studio.pipeline.patterns import (
    discover_fsm_pattern,
    evaluate_and_build_fsm,
    prepare_curves,
)
from bt_studio.pipeline.preprocess import (
    build_fsm_panel,
    build_oos_panel,
    build_static_panel,
    extract_curves_from_panel,
    prepare_macro,
    prepare_tick,
    universe_sample,
)
from bt_studio.utils.math_func import resolve_search_hyperparams
from bt_studio.utils.diagnostics.hpo_check import run_collapse_check
from bt_studio.utils.io import (
    atomic_file_write,
    atomic_save_json,
    atomic_save_parquet,
    atomic_save_pickle,
)
from bt_studio.utils.manifest import (
    safe_load_model_ckpt,
    write_model_manifest as _write_model_manifest,
)
from bt_studio.utils.months import (
    day_month_id_expr,
    find_model_path,
    calc_warmup_start_date,
    is_macro_cache_valid,
    month_id_expr,
    partition_cached_and_missing_months,
    retrieve_available_months,
    resolve_oos_paths_with_warmup,
    ym_range,
)


# ==============================================================================
# Node 1: Macro & Universe
# ==============================================================================

def node_prepare_macro(common_config: dict) -> Dict[str, Any]:
    tune_base = common_config.get("tune_base_dir", TUNE_BASE_DIR)
    os.makedirs(tune_base, exist_ok=True)
    dret_path = os.path.join(tune_base, "global_daily.parquet")

    # 1. calculate warmup start date for universe fetch
    warmup_months = int(common_config.get("warmup_months", DEFAULT_WARMUP_MONTHS))
    fetch_start = calc_warmup_start_date(common_config["start_date"], warmup_months=warmup_months)
    end_date = int(common_config["end_date"])

    # 2. examine cache complete 
    cache_valid, missing_months = is_macro_cache_valid(
        dret_path, fetch_start, end_date, common_config
    )

    if cache_valid:
        print(f"✅ [Cache Hit] Daily Universe: {dret_path}")
        final_scan_lf = pl.scan_parquet(dret_path)
    else:
        if missing_months:
            print(f"[macro cache] missing months: {len(missing_months)}")
        print("🚀 [Cache Miss/Stale] Regenerating Universe...")
        universe_lf, daily_lf = prepare_macro(
            start_date=fetch_start,
            end_date=end_date,
            benchmark=str(common_config["benchmark"]).encode(),
        )
        filtered_uni_lf = universe_sample(universe_lf, daily_lf, common_config)

        with atomic_file_write(dret_path) as tmp_dret_path:
            filtered_uni_lf.sink_parquet(tmp_dret_path)
        atomic_save_json(
            {
                "fetch_start": int(fetch_start),
                "end_date": end_date,
                "benchmark": str(common_config["benchmark"]),
                "days_since_ipo": int(common_config.get("days_since_ipo", 120)),
            },
            dret_path + ".meta.json",
        )
        final_scan_lf = filtered_uni_lf

    # 3. Extract deterministic dynamic universe sids by month (YYYYMM)
    schema = final_scan_lf.collect_schema().names()
    date_col = "day" if "day" in schema else "date"

    df_sids = (
        final_scan_lf.select([
            day_month_id_expr(final_scan_lf, date_col).alias("month_id"),
            pl.col("sid"),
        ])
        .group_by("month_id")
        .agg(pl.col("sid").unique().sort())
        .sort("month_id")
        .collect(engine="streaming")
    )

    dynamic_sids_by_month = dict(
        zip(df_sids["month_id"].to_list(), df_sids["sid"].to_list())
    )

    return {"dret_path": dret_path, "universe_sids": dynamic_sids_by_month}


# ==============================================================================
# Node 2: Feature Extraction & PIT Partitioning
# ==============================================================================

def _prepare_tick_for_months(
    missing_ymonths: List[int],
    universe_sids: Dict[int, List[bytes]],
    common_config: dict,
) -> Optional[pl.LazyFrame]:
    raw_start_date = min(missing_ymonths) * 100 + 1
    end_date = max(missing_ymonths) * 100 + 31

    warmup_months = int(common_config.get("warmup_months", DEFAULT_WARMUP_MONTHS))
    fetch_start_date = calc_warmup_start_date(raw_start_date, warmup_months=warmup_months)

    sids: List[Any] = []
    for ym in missing_ymonths:
        sids.extend(universe_sids.get(ym, []))

    unique_sids = list(dict.fromkeys(sids))
    if not unique_sids:
        return None

    hf = prepare_tick(fetch_start_date, end_date, unique_sids)
    if isinstance(hf, pl.LazyFrame):
        return hf if hf.select(pl.len()).collect().item() > 0 else None
    return None


def node_extract_feature_monthly(
    ymonths: List[int],
    universe_sids: Dict[int, List[bytes]],
    common_config: dict,
    ast_recipe: Union[List[dict], dict],
    raw_hf_lf: Optional[pl.LazyFrame] = None,
) -> List[str]:
    if not ymonths:
        return []

    feature_col = resolve_feature_col(common_config, ast_recipe)
    common_config["feature_col"] = feature_col

    feature_dir = common_config.get("feature_dir", FEATURE_DIR)
    os.makedirs(feature_dir, exist_ok=True)

    agent_dir = (
        os.path.join(FEATURE_DIR, ast_fingerprint(ast_recipe))
        if ast_recipe is not None
        else None
    )
    generated_paths, missing_ymonths = partition_cached_and_missing_months(
        ymonths, feature_col, feature_dir, agent_dir=agent_dir
    )
    if not missing_ymonths:
        return sorted(list(dict.fromkeys(generated_paths)))

    if raw_hf_lf is None:
        raw_hf_lf = _prepare_tick_for_months(missing_ymonths, universe_sids, common_config)
        if raw_hf_lf is None:
            return sorted(list(dict.fromkeys(generated_paths)))

    lazy_feat = compile_feature_lazy(raw_hf_lf, feature_col, ast_recipe)
    feat_df = lazy_feat.collect(engine="streaming")
    if feat_df.height == 0:
        return sorted(list(dict.fromkeys(generated_paths)))


    parts_by_ym = feat_df.partition_by("month_id", as_dict=True, include_key=False)
    for ym in missing_ymonths:
        valid_sids = universe_sids.get(ym, [])
        part_df = parts_by_ym.get((ym,))
        if part_df is None:
            part_df = parts_by_ym.get(ym)
        if not valid_sids or part_df is None or part_df.height == 0:
            continue
        month_df = part_df.filter(pl.col("sid").is_in(valid_sids))
        if month_df.height > 0:
            out_path = f"{feature_dir}/hf_{feature_col}_{ym}.parquet"
            atomic_save_parquet(month_df, out_path)
            generated_paths.append(out_path)
            print(f"  [Persist PIT] -> {out_path}")

    return sorted(list(dict.fromkeys(generated_paths)))


# ==============================================================================
# Node 3: OOS Decay Check
# ==============================================================================

def node_check_decay_monthly(
    prev_model_id: Optional[int],
    prev_oos_yms: List[int],
    dret_path: str,
    train_paths: List[str],
    common_config: dict,
) -> bool:
    if prev_model_id is None:
        print("Cold start (prev_model_id is None) -> Trigger Tune")
        return True

    prev_model_path = find_model_path(prev_model_id, common_config)
    if prev_model_path is None:
        print(f"NotFound: {prev_model_id} -> Trigger Tune")
        return True

    try:
        model_ckpt = safe_load_model_ckpt(prev_model_path)
    except Exception as e:
        print(f"Parse Model Error: {e} -> Trigger Tune")
        return True

    eval_n = max(1, int(common_config.get("decay_eval_months", len(prev_oos_yms))))
    eval_yms = prev_oos_yms[-eval_n:]

    oos_panel_lf = build_oos_panel(
        train_paths,
        eval_yms,
        dret_path,
        model_ckpt["config"],
        common_config,
    )
    if oos_panel_lf is None:
        print(f"Model {prev_model_id} in OOS has no valid panel paths -> Trigger Retune")
        return True

    panel_df = oos_panel_lf.collect(engine="streaming")

    curves_2d = prepare_curves(panel_df, model_ckpt["config"], common_config)
    result = evaluate_and_build_fsm(
        panel_df,
        curves_2d,
        model_ckpt["motif"],
        model_ckpt["config"],
        common_config,
        stats_yms=eval_yms,
    )

    u_pval = float(result.get("u_pval", 1.0)) if result.get("status") == "success" else 1.0
    max_pval = float(common_config.get("u_pval", 0.10))
    needs_retune = result.get("status") != "success" or u_pval > max_pval

    collapse_dir = common_config.get("tune_collapse_dir")
    feat_col = common_config.get("feature_col")
    if collapse_dir and feat_col:
        os.makedirs(str(collapse_dir), exist_ok=True)
        atomic_save_json(
            {
                "model_id": int(prev_model_id),
                "feature_col": str(feat_col),
                "eval_yms": list(eval_yms),
                "status": result.get("status", "failed"),
                "u_pval": round(u_pval, 6),
                "max_pval": max_pval,
                "metrics_score": float(result.get("metrics_score", -500.0)),
                "trigger_count": int(result.get("trigger_count", 0)),
                "valid_sample_ratio": float(result.get("valid_sample_ratio", 0.0)),
                "needs_retune": bool(needs_retune),
            },
            os.path.join(str(collapse_dir), f"decay_{prev_model_id}_{feat_col}.json"),
        )

    if result.get("status") != "success":
        print(f"Model {prev_model_id} in OOS failed evaluation -> Trigger Retune")
        return True

    if not needs_retune:
        print(f"Model {prev_model_id} healthy in OOS ({prev_oos_yms[0]}-{prev_oos_yms[-1]}) (P-val: {u_pval:.4f} <= {max_pval})")
        return False

    print(f"Model {prev_model_id} decay (P-val: {u_pval:.4f} > {max_pval}) -> Trigger Retune")
    return True


# ==============================================================================
# Node 4a: Train Data Preparation
# ==============================================================================

def node_prepare_train(
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

_WORKER_PANEL_CACHE: Dict[tuple, pl.DataFrame] = {}


def trainable_fsm_worker(config: dict, hf_ref: Any, dret_ref: Any, static_ref: Any, common_config: dict):
    trial_cfg = resolve_search_hyperparams(config)
    ds = int(trial_cfg["downsample"])
    cache_key = (id(hf_ref), id(static_ref), ds, common_config.get("feature_col"))

    panel_df = _WORKER_PANEL_CACHE.get(cache_key)
    if panel_df is None:
        hf_lf = pl.from_arrow(hf_ref).lazy()
        static_lf = pl.from_arrow(static_ref).lazy()
        curve_lf = extract_curves_from_panel(hf_lf, trial_cfg, common_config)
        panel_df = curve_lf.join(static_lf, on=["day", "sid"], how="inner").collect(engine="streaming")
        if len(_WORKER_PANEL_CACHE) >= 8:
            _WORKER_PANEL_CACHE.clear()
        _WORKER_PANEL_CACHE[cache_key] = panel_df

    result = discover_fsm_pattern(panel_df, trial_cfg, common_config)

    tune.report({
        "metrics_score": result.get("metrics_score", -500.0),
        "u_pval": result.get("u_pval", 1.0),
        "trigger_count": result.get("trigger_count", 0),
        "valid_sample_ratio": result.get("valid_sample_ratio", 0.0),
        "autocorr": result.get("autocorr", 0.0),
    })


def node_tune_monthly(
    prev_model_id: Optional[Union[int, str]],
    model_id: int,
    train_data: dict,
    exp_config: dict,
) -> Optional[int]:
    if train_data is None:
        print(f"⚠️ [Failed] {model_id} train_data is None (no usable train features)")
        return None

    common_config = common_params_of(exp_config)
    if "feature_col" not in common_config:
        common_config["feature_col"] = resolve_feature_col(
            common_config, exp_config.get("ast_recipe"), exp_config.get("feature_col")
        )

    model_dir = common_config.get("tune_model_dir", common_config.get("tune_model_dir", TUNE_MODEL_DIR))
    os.makedirs(model_dir, exist_ok=True)
    pkl_path = f"{model_dir}/model_{model_id}.pkl"
    if (
        os.path.exists(pkl_path)
        and os.path.exists(f"{model_dir}/model_{model_id}.manifest.json")
        and not common_config.get("force_retune")
    ):
        try:
            safe_load_model_ckpt(pkl_path)
            print(f"✅ [Cache Hit] Model {model_id} already exists at {pkl_path}, skipping HPO")
            return model_id
        except Exception:
            pass

    # 1. search space and Optuna TPE
    search_spec = search_spec_of(exp_config)
    search_space = build_ray_tune_search_space(search_spec)
    points_to_evaluate = resolve_warm_start_points(prev_model_id, common_config, search_spec, search_space)

    search_alg = OptunaSearch(
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=common_config.get("n_startup_trials", 20),
            multivariate=True,
            seed=common_config.get("seed", 42),
        ),
        points_to_evaluate=points_to_evaluate,
    )

    wrapped_trainable = tune.with_resources(
        tune.with_parameters(
            trainable_fsm_worker,
            hf_ref=train_data["hf_ref"],
            dret_ref=train_data["dret_ref"],
            static_ref=train_data["static_ref"],
            common_config=common_config,
        ),
        resources={"cpu": 1, "gpu": 0},
    )

    num_trials = common_config.get("num_trials") or exp_config.get("num_trials", 300)
    trial_timeout_s = float(common_config.get("trial_timeout_s", 600.0) or 600.0)
    max_concurrent = int(common_config.get("num_workers") or 12)
    
    mlflow_callback = create_mlflow_callback(common_config)

    tuner = tune.Tuner(
        wrapped_trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="metrics_score",
            mode="max",
            search_alg=search_alg,
            num_samples=num_trials,
            max_concurrent_trials=max_concurrent,
            reuse_actors=True,
            time_budget_s=trial_timeout_s * max(1, num_trials),
        ),
        run_config=tune.RunConfig(
            name=exp_config.get("run_name", f"fsm_hpo_{model_id}"),
            storage_path=common_config.get("storage_path", "/tmp/ray_results"),
            stop={"time_total_s": trial_timeout_s},
            callbacks=[mlflow_callback] if mlflow_callback else [],
        ),
    )

    # 2. HPO and collapse/statistical gating check
    df_results = pl.from_pandas(tuner.fit().get_dataframe())
    if df_results.height == 0:
        return None

    is_healthy, verdict, _ = run_collapse_check(
        df_results,
        exp_config,
        model_id,
        collapse_dir=common_config.get("tune_collapse_dir", common_config.get("tune_collapse_dir")),
    )
    if not is_healthy:
        print(f"❌ [HPO Gating] Model {model_id} rejected due to {verdict.upper()}")
        return None

    stats_valid_trials = df_results.filter(
        (pl.col("u_pval") <= common_config.get("u_pval", 0.10))
        & (pl.col("trigger_count") >= common_config.get("trigger", 30))
    )
    if stats_valid_trials.height == 0:
        print(f"⚠️ [Failed] {model_id} zero valid statistical samples")
        return None

    # 3. Pareto  + fANOVA 
    best_config = select_plateau_verified_config(
        df_results=df_results,
        stats_valid_trials=stats_valid_trials,
        common_config=common_config,
        search_spec=search_spec,
    )
    if not best_config:
        print(f"❌ {model_id} fANOVA spike detected or empty Pareto front -> Reject")
        return None

    # 4. Overfit FSM and Manifest
    final_panel_lf = build_fsm_panel(
        train_data["hf_lazy"], train_data["dret_lazy"], best_config, common_config, is_train=True
    )
    final_result = discover_fsm_pattern(final_panel_lf, best_config, common_config)

    raw_motif = final_result.get("learned_motif", [])
    motif_arr = np.asarray(raw_motif, dtype=np.float64) if raw_motif else np.empty((0, 0), dtype=np.float64)

    model_ckpt = {
        "config": best_config,
        "motif": motif_arr,
        "fsm_matrix": final_result.get("fsm_matrix", {}),
        "train_end_month": model_id,
        "feature_col": common_config.get("feature_col"),
        "T1_rets": common_config.get("T1_rets"),
        "decay_minutes": common_config.get("decay_minutes"),
    }

    pkl_path = f"{model_dir}/model_{model_id}.pkl"
    atomic_save_pickle(model_ckpt, pkl_path)
    _write_model_manifest(
        model_dir, model_ckpt, common_config, exp_config,
        source="tune",
        gates={"collapse_verdict": verdict, "u_pval": common_config.get("u_pval"),
               "trigger": common_config.get("trigger"), "min_excess_ret": common_config.get("min_excess_ret")},
    )

    print(f"✅ Model {model_id} trained and saved to: {pkl_path}")
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
    model_dir = common_config.get("tune_model_dir", TUNE_MODEL_DIR)
    prev_model_path = find_model_path(prev_model_id, common_config)
    if prev_model_path is None:
        return None

    pre_model_ckpt = safe_load_model_ckpt(prev_model_path)

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
        "T1_rets": pre_model_ckpt.get("T1_rets", common_config.get("T1_rets")),
        "decay_minutes": pre_model_ckpt.get("decay_minutes", common_config.get("decay_minutes")),
    }
    pkl_path = f"{model_dir}/model_{model_id}.pkl"
    atomic_save_pickle(new_ckpt, pkl_path)
    _write_model_manifest(
        model_dir, new_ckpt, common_config, None,
        source="inherit", gates={"decay_eval_months": common_config.get("decay_eval_months")},
    )

    print(f"✅ {model_id} inherit {prev_model_id} Motif and updated FSM Matrix")
    return model_id


# ==============================================================================
# Node 6: OOS Inference
# ==============================================================================

def node_oos_inference_monthly(
    model_id: int,
    dret_path: str,
    oos_yms: List[int],
    oos_paths: List[str],
    common_config: dict,
    train_paths: Optional[List[str]] = None,
) -> None:
    model_dir = common_config.get("tune_model_dir", TUNE_MODEL_DIR)
    score_dir = common_config.get("tune_score_dir", TUNE_SCORE_DIR)
    os.makedirs(score_dir, exist_ok=True)
    model_path = f"{model_dir}/model_{model_id}.pkl"
    if not os.path.exists(model_path):
        print(f"⚠️ NotFound {model_path} and Skip OOS Inference")
        return

    model_ckpt = safe_load_model_ckpt(model_path)

    valid_paths = [p for p in oos_paths if os.path.exists(p)]
    if not valid_paths:
        return

    # 训练尾月 + OOS 月合并供 warmup 解析:
    # ranking 回看月 (ceil(ranking_window/20)) 从训练尾月取, OOS 月本身必须可解析
    all_paths = list(dict.fromkeys(list(train_paths or []) + list(valid_paths)))
    panel_lf = build_oos_panel(
        all_paths,
        oos_yms,
        dret_path,
        model_ckpt["config"],
        common_config,
    )
    if panel_lf is None:
        return

    scored_df = FSMPredictor(model_ckpt, common_config).predict(panel_lf)

    if scored_df.height > 0:
        scored_df = (
            scored_df.with_columns(month_id_expr("day").alias("_month_id"))
            .filter(pl.col("_month_id").is_in(oos_yms))
            .drop("_month_id")
        )

        if scored_df.height > 0:
            feature_id = str(model_ckpt.get("feature_col") or common_config.get("feature_col") or "")
            scored_df = scored_df.with_columns([
                pl.lit(model_id).cast(pl.Int32).alias("model_id"),
                pl.lit(feature_id).alias("feature_id"),
                pl.lit("minute_bar_entry_next_end_labeled").alias("label_basis"),
            ])
            out_path = f"{score_dir}/scores_{model_id}.parquet"
            atomic_save_parquet(scored_df, out_path)
            print(f"✅ {model_id} OOS ({oos_yms[0]} ~ {oos_yms[-1]}) Saved to {out_path}, Rows: {scored_df.height}")
