
import numpy as np


def calculate_hpo_score(
    u_pval: float, 
    trigger_count: int, 
    cond_rets: np.ndarray, 
    uncond_rets: np.ndarray, 
    tune_config: dict,
    common_config: dict
) -> float:

    if u_pval >= common_config["u_pval"]: #  Optuna [0.01 ~ 0.15] to Seek Grad
            return -9999.0

    alternative = common_config["alternative"]
    excess_ret = np.median(cond_rets) - np.median(uncond_rets) # median stable than median
    
    win_rate = np.mean(cond_rets > 0)
    if win_rate <= common_config["win_rate"]:
        return -9999.0

    # =========================================================================
    # A Long-Only 
    # =========================================================================
    alternative = common_config["alternative"]

    if alternative == "greater":
        excess_factor = max(excess_ret, 0.0)
    elif alternative == "less":
        excess_factor = max(-excess_ret, 0.0)
    else: 
        excess_factor = abs(excess_ret)
        
    if excess_factor <= 1e-3: # avoid Friction
        return -9999.0

    # =========================================================================
    # - L = excess / P-value (P small --> L large; excess large --> L large)
    # BIC = -2 * ln(L) + k * ln(n) 
    # - maximize: -BIC = 2 * ln(L) - k * ln(n)
    # =========================================================================
    # likehood
    safe_u_pval = max(u_pval, 1e-10)
    ln_L = np.log(excess_factor) + np.log(win_rate) - np.log(safe_u_pval)
    
    # complexity k 
    cross_days = float(tune_config["cross_days"])
    m_mins = float(tune_config["motif_minutes"])
    dtw_frac = float(common_config["dtw_window_frac"]) 
    
    k = cross_days + (dtw_frac * 10.0) + (m_mins / 30.0)

    # bic calculate 
    n = max(2, len(cond_rets))
    neg_bic = 2 * ln_L - k * np.log(n)
    return float(neg_bic)
