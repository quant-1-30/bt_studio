#! /usr/bin/env python3

import polars as pl
import numpy as np
from typing import List, Dict, Any, Optional

from .astc import compute_dtw_distances
from bt_studio.pipeline.metrics import HPOScoreInputs, calculate_hpo_score
from bt_studio.pipeline.preprocess.macro import compute_macro_states
from bt_studio.utils.math_func import calculate_decay_weights
from bt_studio.utils.months import month_id_expr
from bt_studio.pipeline.metrics.gates import (
    daily_paired_pval,
    daily_trigger_autocorr,
    raw_excess_gate,
)


def extract_fsm_matrix(
    triggers: pl.DataFrame,
    state_cols: list,
    target_names: list,
    n_macro_states: int = 3,
    n_ret_states: int = 5,
) -> dict:
    """Compute Laplace-smoothed FSM transition matrices and empirical state return vectors."""
    select_cols = ["macro_state"] + state_cols
    valid_chain = triggers.drop_nulls(subset=select_cols)

    fsm_dict: dict = {}
    if valid_chain.height > 0:
        num_windows = len(target_names)
        trans_macro_t0 = np.ones((n_macro_states, n_ret_states), dtype=np.float64)
        trans_t_t = [np.ones((n_ret_states, n_ret_states), dtype=np.float64) for _ in range(num_windows - 1)]

        mat_arr = valid_chain.select(select_cols).to_numpy().astype(np.int64, copy=False)
        ms_col = mat_arr[:, 0]
        b0_col = mat_arr[:, 1]
        valid_m0 = (ms_col >= 0) & (ms_col < n_macro_states) & (b0_col >= 0) & (b0_col < n_ret_states)
        np.add.at(trans_macro_t0, (ms_col[valid_m0], b0_col[valid_m0]), 1.0)

        for i in range(num_windows - 1):
            bi = mat_arr[:, i + 1]
            bj = mat_arr[:, i + 2]
            valid_ij = (bi >= 0) & (bi < n_ret_states) & (bj >= 0) & (bj < n_ret_states)
            np.add.at(trans_t_t[i], (bi[valid_ij], bj[valid_ij]), 1.0)

        denom_macro = trans_macro_t0.sum(axis=1, keepdims=True)
        fsm_dict[f"P({target_names[0]}|Macro)"] = np.round(trans_macro_t0 / denom_macro, 6).tolist()

        for i in range(num_windows - 1):
            mat = trans_t_t[i]
            denom_mat = mat.sum(axis=1, keepdims=True)
            fsm_dict[f"P({target_names[i+1]}|{target_names[i]})"] = np.round(mat / denom_mat, 6).tolist()

    state_return_vectors = {}
    expected_states = list(range(n_ret_states))
    for target_name in target_names:
        state_col = f"state_{target_name}"
        ret_col = f"raw_{target_name}"
        agg_rows = (
            triggers.group_by(state_col)
            .agg(pl.col(ret_col).median().alias("ret"))
            .drop_nulls()
            .iter_rows(named=True)
        )
        ret_map = {int(r[state_col]): float(r["ret"]) for r in agg_rows}
        value_vector = [ret_map.get(s, 0.0) for s in expected_states]
        state_return_vectors[target_name] = np.round(value_vector, 6).tolist()
    fsm_dict["state_return_vectors"] = state_return_vectors

    return fsm_dict


def prepare_fsm_base_df(
    panel_df: pl.DataFrame,
    common_config: dict,
) -> Optional[pl.DataFrame]:
    """Precompute motif-independent panel columns (macro_state, rank_*, state_*, and blended trajectories)."""
    if panel_df["sid"].dtype != pl.Binary:
        panel_df = panel_df.with_columns(pl.col("sid").cast(pl.Binary))

    targets = common_config["T1_rets"]
    target_names = [t for t in targets if f"raw_{t}" in panel_df.columns]
    if not target_names:
        return None

    daily_macro = compute_macro_states(panel_df, common_config)
    base_df = panel_df.join(
        daily_macro.select(["day", "macro_state"]),
        on="day",
        how="left",
    ).with_columns(pl.col("macro_state").fill_null(1))

    rank_exprs = [
        pl.when(pl.col(f"raw_{t}").is_null())
        .then(None)
        .when(pl.col(f"raw_{t}").count().over("day") > 1)
        .then(pl.col(f"raw_{t}").rank(method="average").over("day") / pl.col(f"raw_{t}").count().over("day"))
        .otherwise(0.5)
        .alias(f"rank_{t}")
        for t in target_names
    ]
    base_df = base_df.with_columns(rank_exprs)

    state_exprs = [
        pl.when(pl.col(f"rank_{t}").is_null()).then(None)
        .when(pl.col(f"rank_{t}") <= 0.10).then(0)
        .when(pl.col(f"rank_{t}") <= 0.30).then(1)
        .when(pl.col(f"rank_{t}") <= 0.70).then(2)
        .when(pl.col(f"rank_{t}") <= 0.90).then(3)
        .otherwise(4).cast(pl.Int32).alias(f"state_{t}")
        for t in target_names
    ]
    base_df = base_df.with_columns(state_exprs)

    weights = calculate_decay_weights(targets, half_life_minutes=common_config["decay_minutes"])
    total_w = sum(weights[t] for t in target_names)

    def _blend(prefix: str) -> pl.Expr:
        return sum(pl.col(f"{prefix}_{t}") * weights[t] for t in target_names) / total_w

    return base_df.with_columns([
        _blend("rank").alias("rank_trajectory"),
        _blend("intra").alias("momentum_trajectory"),
        _blend("raw").alias("raw_trajectory"),
    ])


def evaluate_and_build_fsm(
    panel_df: pl.DataFrame,
    curves_2d: np.ndarray,
    motif: np.ndarray,
    tune_config: dict,
    common_config: dict,
) -> dict:
    skip_stats = bool(common_config.get("_skip_stats", False))
    stats_yms = common_config.get("_stats_yms")
    targets = common_config["T1_rets"]
    target_names = [t for t in targets if f"raw_{t}" in panel_df.columns]
    if not target_names:
        return {"status": "failed", "reason": "panel has no raw_* target columns", "metrics_score": -500.0}

    # 1. Compute subsequence DTW distances
    threshold_d = float(tune_config["threshold_d"])
    distances = compute_dtw_distances(curves_2d, motif, threshold_d, common_config)

    # 2-4. Reuse precomputed base_df columns if already present on panel_df, else compute once
    base_df = (
        panel_df
        if "rank_trajectory" in panel_df.columns
        else prepare_fsm_base_df(panel_df, common_config)
    )
    if base_df is None:
        return {"status": "failed", "reason": "panel has no raw_* target columns", "metrics_score": -500.0}

    target_states = [f"state_{t}" for t in target_names]
    dist_series = pl.Series("distance", distances, dtype=pl.Float64)
    eval_df = base_df.with_columns([
        dist_series,
        (dist_series <= threshold_d).cast(pl.Int8).alias("is_trigger"),
    ])

    if stats_yms is not None:
        eval_df = eval_df.filter(month_id_expr("day").is_in(stats_yms))

    triggers = eval_df.filter(pl.col("distance") <= threshold_d)
    complementary_df = eval_df.filter(pl.col("distance") > threshold_d)

    if triggers.height < common_config["trigger"]:
        return {
            "status": "failed",
            "reason": f"DTW (n={triggers.height}) < trigger ({common_config['trigger']})",
            "metrics_score": -500.0,
        }

    if skip_stats:
        return {
            "status": "success",
            "fsm_matrix": extract_fsm_matrix(triggers, target_states, target_names),
            "trigger_count": triggers.height,
            "learned_motif": np.round(motif, 6).tolist(),
        }

    # Fail-fast statistical & economic gates BEFORE computing FSM transition matrix & autocorr
    cond_ranks = triggers["rank_trajectory"].drop_nulls().to_numpy()
    uncond_ranks = complementary_df["rank_trajectory"].drop_nulls().to_numpy()
    min_test_samples = max(10, common_config["trigger"] // 3)
    min_uncond_samples = max(3, min(500, eval_df.height // 100))

    if (
        len(cond_ranks) < min_test_samples
        or np.std(cond_ranks) < 1e-8
        or len(uncond_ranks) < min_uncond_samples
    ):
        return {
            "status": "failed",
            "reason": f"Rank trajectory valid triggers ({len(cond_ranks)}) or control ({len(uncond_ranks)} < {min_uncond_samples}) insufficient",
            "metrics_score": -500.0,
        }

    gate_passed, gate_reason = raw_excess_gate(triggers, complementary_df, common_config)
    if not gate_passed:
        return {
            "status": "failed",
            "reason": gate_reason,
            "metrics_score": -500.0,
        }

    cond_data = triggers.select(["raw_trajectory", "momentum_trajectory"]).drop_nulls().to_numpy()
    if len(cond_data) <= 2:
        return {
            "status": "failed",
            "reason": "insufficient valid triggered samples",
            "metrics_score": -500.0,
        }

    cond_rets = cond_data[:, 0]
    cond_momentum = cond_data[:, 1]
    uncond_rets = complementary_df["raw_trajectory"].drop_nulls().to_numpy()
    cond_rets = cond_rets[np.isfinite(cond_rets)]
    cond_momentum = cond_momentum[np.isfinite(cond_momentum)]
    uncond_rets = uncond_rets[np.isfinite(uncond_rets)]
    if len(cond_rets) <= 2 or len(uncond_rets) == 0:
        return {
            "status": "failed",
            "reason": "insufficient finite scored samples",
            "metrics_score": -500.0,
        }

    t1_missing_ratio = float(1.0 - (len(cond_rets) / max(1, triggers.height)))
    max_t1_missing = float(common_config.get("max_t1_missing_ratio", 0.25))
    if t1_missing_ratio > max_t1_missing:
        return {
            "status": "failed",
            "reason": f"T+1 outcome truncation ratio {t1_missing_ratio:.1%} > {max_t1_missing:.1%}",
            "metrics_score": -500.0,
        }

    u_pval = daily_paired_pval(triggers, complementary_df, common_config["alternative"])
    valid_sample_ratio = triggers.height / max(eval_df.height, 1)
    autocorr = daily_trigger_autocorr(eval_df)
    fsm_matrix = extract_fsm_matrix(triggers, target_states, target_names)

    score = calculate_hpo_score(
        HPOScoreInputs(
            u_pval=u_pval,
            trigger_count=len(cond_rets),
            cond_rets=cond_rets,
            uncond_rets=uncond_rets,
            cond_momentum=cond_momentum,
            tune_config=tune_config,
            common_config=common_config,
        )
    )
    return {
        "status": "success",
        "fsm_matrix": fsm_matrix,
        "trigger_count": triggers.height,
        "t1_missing_ratio": round(t1_missing_ratio, 4),
        "learned_motif": np.round(motif, 6).tolist(),
        "metrics_score": round(score, 6),
        "valid_sample_ratio": valid_sample_ratio,
        "autocorr": autocorr,
        "u_pval": round(float(u_pval), 6),
    }
