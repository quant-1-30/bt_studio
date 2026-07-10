import numpy as np
import polars as pl
from .m_astc import calc_min_subseq_dtw_md, prepare_mcurves


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

        self.bin_weights = np.array([-1.0, -0.5, 0.5, 1.0])
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
        
        # def calc_alpha_score(macro_state):
        #     if macro_state is None: return 0.0
        #     p_t1 = self.p_t1_macro[macro_state] 
        #     p_t2 = p_t1 @ self.p_t2_t1          
        #     p_t3 = p_t2 @ self.p_t3_t2          
        #     return float(np.dot(p_t1, self.bin_weights) * 0.5 + 
        #                  np.dot(p_t2, self.bin_weights) * 0.3 + 
        #                  np.dot(p_t3, self.bin_weights) * 0.2)

        def calc_alpha_score(macro_state):
            if macro_state is None: return 0.0
            
            # P(T1) = P(T1|Macro)
            p_curr = self.p_t1_macro[macro_state] 
            expected_scores = []
            
            # T+1
            expected_scores.append(np.dot(p_curr, self.bin_weights))
            
            # T+2
            if len(stats_windows) >= 2:
                p_curr = p_curr @ self.p_t2_t1
                expected_scores.append(np.dot(p_curr, self.bin_weights))
                
            # T+3
            if len(stats_windows) >= 3:
                p_curr = p_curr @ self.p_t3_t2
                expected_scores.append(np.dot(p_curr, self.bin_weights))
                
            final_score = 0.0
            for idx, fw in enumerate(stats_windows):
                w = self.traj_weights[fw]
                final_score += expected_scores[idx] * w
            return float(final_score)

        # macro_State 0, 1, 2
        score_map = {
            0: calc_alpha_score(0),
            1: calc_alpha_score(1),
            2: calc_alpha_score(2)
        }

        # missing macro_state
        score_map[None] = 0.0

        return (
            # Polars C replace
            triggers.with_columns(
                pl.col("macro_state").replace(score_map, default=0.0).alias("fsm_score")
            )
            .select(["day", "sid", "distance", "macro_state", "fsm_score"])
            .sort(["day", "fsm_score"], descending=[False, True])
        )

    def _predict(self, panel_df: pl.DataFrame) -> pl.DataFrame:
        curves_md = prepare_mcurves(panel_df, self.common_config, self.config)
        
        m_means = np.mean(self.motif, axis=1, keepdims=True)
        m_stds = np.std(self.motif, axis=1, keepdims=True) + 1e-8
        z_motif_t = np.ascontiguousarray(((self.motif - m_means) / m_stds).T, dtype=np.float64)

        distances = [calc_min_subseq_dtw_md(curves_md[i], z_motif_t, self.dtw_w, self.threshold_d) for i in range(curves_md.shape[0])]
        
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
