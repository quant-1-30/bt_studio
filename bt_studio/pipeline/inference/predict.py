import numpy as np
import polars as pl


class FSMPredictor:
    """
        Motif + FSM Network OSS
    """
    def __init__(self, model_ckpt: dict):
        self.config = model_ckpt["config"]
        self.motif = np.array(model_ckpt["motif"])
        self.m = self.config["m"]
        self.threshold_d = self.config["threshold_d"]
        self.dtw_w = max(1, int(self.m * self.config.get("dtw_window_frac", 0.1)))
        
        # FSM Numpy Matrix
        fsm_network = model_ckpt["fsm_network"]
        self.p_t1_macro = np.array(fsm_network["P(T1|Macro)"]) # Shape: (3, 4)
        self.p_t2_t1 = np.array(fsm_network["P(T2|T1)"])       # Shape: (4, 4)
        self.p_t3_t2 = np.array(fsm_network["P(T3|T2)"])       # Shape: (4, 4)
        
        # 0:大跌, 1:微跌, 2:微涨, 3:大涨
        self.bin_weights = np.array([-1.0, -0.5, 0.5, 1.0])

    def predict(self, panel_lf: pl.LazyFrame) -> pl.DataFrame:
        
        panel_df = panel_lf.collect()
        if panel_df.height == 0:
            return pl.DataFrame()
            
        cross_days = int(self.config["cross_days"])
        lag_cols = [f"lag_{i}" for i in reversed(range(cross_days))]
        curves_2d = np.hstack([np.vstack(panel_df[col].to_list()) for col in lag_cols])
        
        z_motif = np.ascontiguousarray((self.motif - np.mean(self.motif)) / (np.std(self.motif) + 1e-8), dtype=np.float64)
        
        distances = [
            calc_min_subseq_dtw(curve, z_motif, self.m, self.dtw_w, self.threshold_d) 
            for curve in curves_2d 
        ]
        panel_df = panel_df.with_columns(pl.Series("distance", distances))
         
        # Macro State
        daily_macro = (
            # triggers.group_by(["day", "sid"])
            panel_df.group_by(["day", "sid"])
            .agg(pl.col("daily_curve").list.sum().alias("sid_ofi_sum"))
            .group_by("day")
            .agg(pl.col("sid_ofi_sum").mean().alias("daily_ofi_mean"))
            .with_columns([
                pl.col("daily_ofi_mean").quantile(1/3).alias("p33"),
                pl.col("daily_ofi_mean").quantile(2/3).alias("p67")
            ])
            .with_columns(
                pl.when(pl.col("daily_ofi_mean") <= pl.col("p33")).then(0)
                .when(pl.col("daily_ofi_mean") <= pl.col("p67")).then(1)
                .otherwise(2).cast(pl.Int32).alias("macro_state")
            ).drop(["p33", "p67", "daily_ofi_mean"])
        )
        
        triggers = panel_df.filter(pl.col("distance") <= self.threshold_d)
        if triggers.height == 0:
            return pl.DataFrame()

        triggers = triggers.join(daily_macro, on="day", how="left")
        
        # P(T_n | Macro)
        def calc_alpha_score(macro_state):
            if macro_state is None: return 0.0
            p_t1 = self.p_t1_macro[macro_state] # P(T1) = P(T1|Macro)
            p_t2 = p_t1 @ self.p_t2_t1          # P(T2) = P(T1) * P(T2|T1)
            p_t3 = p_t2 @ self.p_t3_t2          # P(T3) = P(T2) * P(T3|T2)

            # hardcoding T+1、T+2、T+3 allocate wgt 
            score = (
                np.dot(p_t1, self.bin_weights) * 0.5 + 
                np.dot(p_t2, self.bin_weights) * 0.3 + 
                np.dot(p_t3, self.bin_weights) * 0.2
            )
            return float(score)
            
        scores = [calc_alpha_score(m) for m in triggers["macro_state"].to_list()]
        
        return (
            triggers.with_columns(pl.Series("fsm_score", scores))
            .select(["day", "sid", "distance", "macro_state", "fsm_score"])
            .sort(["day", "fsm_score"], descending=[False, True])
        )
