import numpy as np
import polars as pl
from bt_studio.pipeline.patterns.astc import calc_min_subseq_dtw, prepare_curves
from bt_studio.utils.common import calculate_decay_weights


class FSMPredictor:
    """
        Motif + FSM Network OSS 1M/DM
    """
    
    def __init__(self, model_ckpt: dict, common_config: dict):
        self.tune_config = model_ckpt["config"]
        self.motif = np.array(model_ckpt["motif"])
        self.common_config = common_config
        
        self.m = self.tune_config["m"]
        self.threshold_d = self.tune_config["threshold_d"]
        self.dtw_w = max(3, int(self.m * self.common_config["dtw_window_frac"]))

        fsm_matrix = model_ckpt["fsm_matrix"]
        self.p_t1_macro = np.array(fsm_matrix["P(T1|Macro)"]) 
        self.p_t2_t1 = np.array(fsm_matrix["P(T2|T1)"])       
        self.p_t3_t2 = np.array(fsm_matrix["P(T3|T2)"])       

        self.bin_weights = fsm_matrix["bin_weights"]
        self.traj_weights = calculate_decay_weights(common_config["stats_windows"], half_life=common_config["decay"]) 
        
    def _get_macro_state(self, panel_df: pl.DataFrame) -> pl.DataFrame:
        rank_window = self.common_config["ranking_window"]

        daily_macro_lf = (
            panel_df.lazy()
            .select(["day", "sid", pl.col("lag_0").list.sum().alias("sid_ofi_sum")])
            .group_by("day")
            .agg(pl.col("sid_ofi_sum").mean().alias("daily_ofi_mean"))
            .sort("day") 
            .with_columns([
                pl.col("daily_ofi_mean")
                .rolling_quantile(quantile=0.33, window_size=rank_window, min_periods=5)
                .alias("p33"),
                pl.col("daily_ofi_mean")
                .rolling_quantile(quantile=0.67, window_size=rank_window, min_periods=5)
                .alias("p67")
            ])
            .with_columns(
                pl.when(pl.col("daily_ofi_mean") <= pl.col("p33")).then(0)
                .when(pl.col("daily_ofi_mean") <= pl.col("p67")).then(1)
                .otherwise(2)
                .cast(pl.Int32)
                .alias("macro_state")
            )
            .with_columns(pl.col("macro_state").shift(1))  # avoid loopahead
            .drop(["p33", "p67", "daily_ofi_mean"])
            .drop_nulls()
        )
        return daily_macro_lf.collect()

    def _calculate_fsm_score(self, triggers: pl.DataFrame) -> pl.DataFrame:
        # =========================================================================
        # Precompute Look-up Array)
        # =========================================================================
        state_scores = np.zeros(3, dtype=np.float64)
        stats_windows = self.common_config["stats_windows"]
        
        def get_weights(fw):
            w = self.bin_weights.get(fw)
            if w is None:
                w = [0.0, 0.0, 0.0, 0.0] 
            return np.array(w)
        
        for macro_state in [0, 1, 2]:
            p_curr = self.p_t1_macro[macro_state] 
            expected_scores = []
            
            # T+1
            if len(stats_windows) >= 1:
                w_1 = get_weights(stats_windows[0])
                expected_scores.append(np.dot(p_curr, w_1))
            
            # T+2
            if len(stats_windows) >= 2:
                p_curr = p_curr @ self.p_t2_t1
                w_2 = get_weights(stats_windows[1])
                expected_scores.append(np.dot(p_curr, w_2))
                
            # T+3
            if len(stats_windows) >= 3:
                p_curr = p_curr @ self.p_t3_t2
                w_3 = get_weights(stats_windows[2])
                expected_scores.append(np.dot(p_curr, w_3))
                
            # decay weight between T+1 / T+3
            final_score = 0.0
            for idx, fw in enumerate(stats_windows):
                w = self.traj_weights[fw]
                final_score += expected_scores[idx] * w
                
            state_scores[macro_state] = float(final_score)

        # =========================================================================
        # Polars C Engine
        # =========================================================================
        return (
            triggers
            # A. Base Expected Return
            .with_columns(
                pl.when(pl.col("macro_state") == 0).then(state_scores[0])
                .when(pl.col("macro_state") == 1).then(state_scores[1])
                .when(pl.col("macro_state") == 2).then(state_scores[2])
                .otherwise(0.0)
                .alias("expected_return")
            )
            # B. sid_score_ret = expected_return * (1 - distance / threshold_d)
            .with_columns(
                (
                    pl.col("expected_return") * 
                    (1.0 - (pl.col("distance") / self.threshold_d))
                ).alias("fsm_score")
            )
            .select(["day", "sid", "distance", "macro_state", "fsm_score"])
            # skip 0 score 
            .sort(["day", "fsm_score"], descending=[False, True])
        )

    def _predict(self, panel_df: pl.DataFrame) -> pl.DataFrame:
        curves_1d = prepare_curves(panel_df, self.tune_config, self.common_config)
        if curves_1d.size == 0: return pl.DataFrame()

        z_motif = np.ascontiguousarray((self.motif - np.mean(self.motif)) / (np.std(self.motif) + 1e-8), dtype=np.float64)
        distances = [calc_min_subseq_dtw(curves_1d[i], z_motif, self.m, self.dtw_w, self.threshold_d) for i in range(curves_1d.shape[0])]
        
        panel_df = panel_df.with_columns(pl.Series("distance", distances))

        triggers = panel_df.filter(pl.col("distance") <= self.threshold_d)
        if triggers.height == 0:
            return pl.DataFrame()

        daily_macro = self._get_macro_state(panel_df)
        return self._calculate_fsm_score(triggers.join(daily_macro, on="day", how="left"))

    def predict(self, panel_lf: pl.LazyFrame) -> pl.DataFrame:
        panel_df = panel_lf.collect(streaming=True)
        if panel_df.height == 0: return pl.DataFrame()
        return self._predict(panel_df)
