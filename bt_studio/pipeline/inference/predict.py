import numpy as np
import polars as pl
from concurrent.futures import ThreadPoolExecutor
from bt_studio.pipeline.patterns.astc import calc_min_subseq_dtw, prepare_curves
from bt_studio.utils.common import calculate_decay_weights


class FSMPredictor:
    """
    Motif + FSM Network OSS Inference Predictor (Highly Optimized)
    """
    
    def __init__(self, model_ckpt: dict, common_config: dict):
        self.tune_config = model_ckpt["config"]
        self.motif = np.array(model_ckpt["motif"])
        self.common_config = common_config
        
        self.m = self.tune_config["m"]
        self.threshold_d = self.tune_config["threshold_d"]
        self.dtw_w = max(3, int(self.m * self.common_config["dtw_window_frac"]))
        
        self.target_names = list(self.common_config.get("T1_rets", {}).keys())

        # =========================================================================
        # Transfer Matrix
        # =========================================================================
        fsm_matrix = model_ckpt["fsm_matrix"]
        
        self.p_macro_t0 = np.array(fsm_matrix[f"P(T{self.target_names[0]}|Macro)"])
        self.trans_matrices = []

        for i in range(1, len(self.target_names)):
            key = f"P(T{self.target_names[i]}|T{self.target_names[i-1]})"
            self.trans_matrices.append(np.array(fsm_matrix[key]))

        self.state_return_vectors = fsm_matrix["state_return_vectors"]
        self.traj_weights = calculate_decay_weights(self.target_names, half_life=common_config["decay_minutes"]) 
        
    def _get_macro_state(self, panel_df: pl.DataFrame) -> pl.DataFrame:
        rank_window = self.common_config["ranking_window"]

        daily_macro_lf = (
            panel_df.lazy()
            .select(["day", "sid", pl.col("lag_0").list.sum().alias("sid_ofi_sum")])
            .group_by("day")
            .agg(pl.col("sid_ofi_sum").median().alias("daily_ofi_median"))
            .sort("day") 
            .with_columns(pl.col("daily_ofi_median").shift(1).alias("prev_ofi_median"))
            .with_columns([
                pl.col("prev_ofi_median")
                .rolling_quantile(quantile=0.33, window_size=rank_window, min_periods=max(1, rank_window//2))
                .alias("p33"),
                pl.col("prev_ofi_median")
                .rolling_quantile(quantile=0.67, window_size=rank_window, min_periods=max(1, rank_window//2))
                .alias("p67")
            ])
            .with_columns(
                pl.when(pl.col("prev_ofi_median") <= pl.col("p33")).then(0)
                .when(pl.col("prev_ofi_median") <= pl.col("p67")).then(1)
                .otherwise(2)
                .fill_null(1) 
                .cast(pl.Int32)
                .alias("macro_state")
            )
            .drop(["p33", "p67", "daily_ofi_median", "prev_ofi_median"])
        )
        return daily_macro_lf.collect()

    def _calculate_fsm_score(self, triggers: pl.DataFrame) -> pl.DataFrame:
        state_scores = np.zeros(3, dtype=np.float64)
        
        def get_weights(fw):
            w = self.state_return_vectors.get(str(fw)) 
            if w is None:
                w = [0.0, 0.0, 0.0, 0.0] 
            return np.array(w)
        
        for macro_state in [0, 1, 2]:
            p_curr = self.p_macro_t0[macro_state] 
            expected_scores = []
            
            w_curr = get_weights(self.target_names[0])
            expected_scores.append(np.dot(p_curr, w_curr))
            
            for i in range(1, len(self.target_names)):
                p_curr = p_curr @ self.trans_matrices[i-1] 
                w_next = get_weights(self.target_names[i])
                expected_scores.append(np.dot(p_curr, w_next))
                
            final_score = 0.0
            for idx, fw in enumerate(self.target_names):
                w = self.traj_weights.get(fw, 0.0)
                final_score += expected_scores[idx] * w
                
            state_scores[macro_state] = float(final_score)

        # Polars replace ---> when.then
        mapping = {0: state_scores[0], 1: state_scores[1], 2: state_scores[2]}
        
        return (
            triggers
            .with_columns(
                pl.col("macro_state").replace(mapping).alias("expected_return")
            )
            .with_columns(
                (
                    pl.col("expected_return") * 
                    (1.0 - (pl.col("distance") / self.threshold_d))
                ).alias("fsm_score")
            )
            .select(["day", "sid", "distance", "macro_state", "fsm_score"])
            .filter(pl.col("fsm_score") != 0.0)
            .sort(["day", "fsm_score"], descending=[False, True])
        )

    def _predict(self, panel_df: pl.DataFrame) -> pl.DataFrame:
        curves_2d = prepare_curves(panel_df, self.tune_config, self.common_config)
        if curves_2d.size == 0: 
            return pl.DataFrame()

        z_motif = np.ascontiguousarray((self.motif - np.mean(self.motif)) / (np.std(self.motif) + 1e-8), dtype=np.float64)
        
        # ThreadPoolExecutor DTW because dtaidistance release GIL
        def _calc_dist(i):
            return calc_min_subseq_dtw(curves_2d[i], z_motif, self.m, self.dtw_w, self.threshold_d)
            
        with ThreadPoolExecutor() as executor:
            distances = list(executor.map(_calc_dist, range(curves_2d.shape[0])))
        
        panel_df = panel_df.with_columns(pl.Series("distance", distances))
        triggers = panel_df.filter(pl.col("distance") <= self.threshold_d)
        if triggers.height == 0:
            return pl.DataFrame()

        daily_macro = self._get_macro_state(panel_df)
        
        valid_triggers = triggers.join(daily_macro, on="day", how="left").drop_nulls(subset=["macro_state"])
        if valid_triggers.height == 0:
            return pl.DataFrame()
            
        return self._calculate_fsm_score(valid_triggers)

    def predict(self, panel_lf: pl.LazyFrame) -> pl.DataFrame:
        panel_df = panel_lf.collect(streaming=True)
        if panel_df.height == 0: 
            return pl.DataFrame()
        return self._predict(panel_df)

