
import numpy as np


# def calculate_hpo_score(
#     u_pval: float, 
#     trigger_count: int, 
#     cond_rets: np.ndarray, 
#     uncond_rets: np.ndarray, 
#     tune_config: dict,
#     common_config: dict
# ) -> float:
    
#     p_score = -np.log10(max(u_pval, 1e-10))
#     n_penalty = np.sqrt(trigger_count)
#     excess_ret = np.mean(cond_rets) - np.mean(uncond_rets)
    
#     # =========================================================================
#     # A Long-Only 
#     # =========================================================================
#     alternative = common_config["alternative"]
#     if alternative == "greater":
#         excess_factor = max(excess_ret, 0.0)
#     elif alternative == "less":
#         excess_factor = max(-excess_ret, 0.0)
#     else: 
#         excess_factor = abs(excess_ret)
        
#     if excess_factor <= 1e-6:
#         return -9999.0
        
#     dtw_window_frac = float(common_config["dtw_window_frac"]) 
#     cross_days = float(tune_config["cross_days"])
#     threshold_r = float(tune_config["threshold_r"])
#     motif_minutes = float(tune_config["motif_minutes"])
    
#     complexity = cross_days * ((motif_minutes / float(tune_config["downsample"])) * dtw_window_frac) * (1.0 - threshold_r)
#     complexity = max(1e-4, complexity)
    
#     raw_score = p_score * n_penalty * excess_factor * 10000.0
#     return float(raw_score / complexity)


def calculate_hpo_score(
    u_pval: float, 
    trigger_count: int, 
    cond_rets: np.ndarray, 
    uncond_rets: np.ndarray, 
    tune_config: dict,
    common_config: dict
) -> float:
    alternative = common_config["alternative"]
    excess_ret = np.mean(cond_rets) - np.mean(uncond_rets)

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
        
    if excess_factor <= 1e-6:
        return -9999.0

    # =========================================================================
    # - L = excess / P-value (P small --> L large; excess large --> L large)
    # BIC = -2 * ln(L) + k * ln(n) 
    # - maximize: -BIC = 2 * ln(L) - k * ln(n)
    # =========================================================================
    # likehood
    safe_pval = max(u_pval, 1e-10)
    ln_L = np.log(excess_factor) - np.log(safe_pval)
    
    # complexity k 
    cross_days = float(tune_config["cross_days"])
    m_mins = float(tune_config["motif_minutes"])
    dtw_frac = float(common_config["dtw_window_frac"]) 
    
    k = cross_days + (dtw_frac * 10.0) + (m_mins / 30.0)

    # bic calculate 
    n = max(2, len(cond_rets))
    neg_bic = 2 * ln_L - k * np.log(n)
    return float(neg_bic)
