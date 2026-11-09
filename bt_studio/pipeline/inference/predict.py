#! /usr/bin/env python3

from typing import Any, Dict, List, Optional

from bt_studio.pipeline.patterns.astc import compute_dtw_distances, prepare_curves
from bt_studio.pipeline.preprocess.macro import compute_macro_states
from bt_studio.utils.io import collect_lazy_safe
from bt_studio.utils.math_func import calculate_decay_weights
import numpy as np
import polars as pl


class FSMPredictor:
    """Production motif + joint-state FSM inference predictor."""

    def __init__(self, model_ckpt: Dict[str, Any], common_config: Dict[str, Any]):
        self.tune_config = model_ckpt["config"]
        self.common_config = common_config

        # 1. Motif hyperparameters and raw motif array
        self.m = int(self.tune_config["m"])
        self.threshold_d = float(self.tune_config["threshold_d"])
        self.raw_motif = np.asarray(model_ckpt["motif"], dtype=np.float64)

        # 2. Target horizon definitions and exponential decay weights
        self.t1_rets = model_ckpt.get("T1_rets") or self.common_config["T1_rets"]
        self.decay_minutes = float(
            model_ckpt.get("decay_minutes") or self.common_config.get("decay_minutes", 15.0)
        )
        self.target_names: List[str] = list(self.t1_rets.keys())
        self.traj_weights = calculate_decay_weights(
            self.t1_rets, half_life_minutes=self.decay_minutes
        )

        # 3. Precompute discounted expected returns per macro state
        self.macro_expected_returns = self._precompute_macro_state_scores(
            fsm_matrix=model_ckpt["fsm_matrix"]
        )

    def _precompute_macro_state_scores(self, fsm_matrix: Dict[str, Any]) -> Dict[int, float]:
        """Precompute expected discounted returns per macro state at initialization time."""
        target_0 = self.target_names[0]
        p_macro_t0 = np.asarray(fsm_matrix[f"P({target_0}|Macro)"], dtype=np.float64)
        n_macro_states = p_macro_t0.shape[0]
        return_vectors = fsm_matrix.get("state_return_vectors")
        if not isinstance(return_vectors, dict):
            raise ValueError("Invalid fsm_matrix: missing 'state_return_vectors' dictionary.")

        def _get_reward_vec(fw: str) -> np.ndarray:
            vec = return_vectors.get(str(fw))
            if vec is None or len(vec) != p_macro_t0.shape[1]:
                raise ValueError(
                    f"Invalid fsm_matrix: missing or mismatched state_return_vectors[{fw!r}] "
                    f"(expected length {p_macro_t0.shape[1]})."
                )
            return np.asarray(vec, dtype=np.float64)

        p_curr = p_macro_t0.copy()
        total_weight = sum(self.traj_weights.get(fw, 0.0) for fw in self.target_names) or 1.0
        scores = np.zeros(n_macro_states, dtype=np.float64)

        for idx, fw in enumerate(self.target_names):
            if idx > 0:
                trans = np.asarray(fsm_matrix[f"P({fw}|{self.target_names[idx - 1]})"], dtype=np.float64)
                p_curr = p_curr @ trans
            scores += (p_curr @ _get_reward_vec(fw)) * self.traj_weights.get(fw, 0.0)

        scores /= total_weight
        return {ms: float(scores[ms]) for ms in range(n_macro_states)}

    def _calculate_fsm_score(self, triggers: pl.DataFrame) -> pl.DataFrame:
        """Compute final confidence score from expected return and DTW proximity."""
        return (
            triggers
            .with_columns(
                pl.col("macro_state")
                .replace_strict(self.macro_expected_returns, default=0.0, return_dtype=pl.Float64)
                .alias("expected_return")
            )
            .with_columns(
                (
                    pl.col("expected_return")
                    * (1.0 - (pl.col("distance") / self.threshold_d)).clip(lower_bound=0.0)
                ).alias("fsm_score")
            )
            .select(["day", "sid", "distance", "macro_state", "expected_return", "fsm_score"])
            .sort(["day", "fsm_score"], descending=[False, True])
        )

    def _predict(self, panel_df: pl.DataFrame) -> pl.DataFrame:
        # 1. Deterministic (day, sid) ordering
        sorted_panel = panel_df.sort(["day", "sid"])

        # 2. Build 2D curve matrix
        curves_2d = prepare_curves(sorted_panel, self.tune_config, self.common_config)
        if curves_2d.size == 0:
            return pl.DataFrame()

        # 3. Compute subsequence DTW distances
        distances = compute_dtw_distances(
            curves_2d, self.raw_motif, self.threshold_d, self.common_config
        )

        # 4. Filter triggered samples by threshold_d
        matched_df = sorted_panel.with_columns(
            pl.Series("distance", distances, dtype=pl.Float64)
        )
        triggers = matched_df.filter(pl.col("distance") <= self.threshold_d)
        if triggers.height == 0:
            return pl.DataFrame()

        # 5. Join causal macro state
        daily_macro = compute_macro_states(sorted_panel, self.common_config)
        valid_triggers = triggers.join(daily_macro, on="day", how="left").drop_nulls(
            subset=["macro_state"]
        )
        if valid_triggers.height == 0:
            return pl.DataFrame()

        # 6. Compute FSM score
        return self._calculate_fsm_score(valid_triggers)

    def predict(self, panel_lf: pl.LazyFrame) -> pl.DataFrame:
        """Collect LazyFrame and run batch FSM inference."""
        panel_df = collect_lazy_safe(panel_lf)
        if panel_df.height == 0:
            return pl.DataFrame()
        return self._predict(panel_df)