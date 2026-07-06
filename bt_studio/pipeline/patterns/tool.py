import polars as pl
import numpy as np


def extract_fsm_matrix(triggers: pl.DataFrame, bin_cols: list) -> dict:
    """freq count with Laplace smoothing"""
    trans_t1 = np.ones((3, 4), dtype=np.float64) 
    trans_t1_t2 = np.ones((4, 4), dtype=np.float64) 
    trans_t2_t3 = np.ones((4, 4), dtype=np.float64) 

    select_cols = ["macro_state"] + bin_cols
    valid_chain = triggers.drop_nulls(subset=select_cols) 
    
    if valid_chain.height > 0:
        for row in valid_chain.select(select_cols).iter_rows():
            ms = row[0]
            actual_bins = row[1:] 
            if len(actual_bins) >= 1: trans_t1[ms, actual_bins[0]] += 1.0
            if len(actual_bins) >= 2: trans_t1_t2[actual_bins[0], actual_bins[1]] += 1.0
            if len(actual_bins) >= 3: trans_t2_t3[actual_bins[1], actual_bins[2]] += 1.0
            
    return {
        "P(T1|Macro)": (trans_t1 / trans_t1.sum(axis=1, keepdims=True)).tolist(),
        "P(T2|T1)": (trans_t1_t2 / trans_t1_t2.sum(axis=1, keepdims=True)).tolist(),
        "P(T3|T2)": (trans_t2_t3 / trans_t2_t3.sum(axis=1, keepdims=True)).tolist()
    }
