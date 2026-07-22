
import numpy as np


def calculate_hpo_score(
    u_pval: float, 
    trigger_count: int, 
    cond_rets: np.ndarray, 
    uncond_rets: np.ndarray,
    cond_z_gaps: np.ndarray, 
    cond_intra: np.ndarray,
    tune_config: dict,
    common_config: dict
) -> float:
    
    excess_ret = np.median(cond_rets) - np.median(uncond_rets) 
    win_rate = np.mean(cond_rets > 0)
    intra_win_rate = np.mean(cond_intra > 0)
    eps = common_config["eps"]
    
    alternative = common_config["alternative"]
    # avoid log 0
    if alternative == "greater":
        excess_factor = max(excess_ret, eps) 
    elif alternative == "less":
        excess_factor = max(-excess_ret, eps)
    else: 
        excess_factor = max(abs(excess_ret), eps)
        
    safe_u_pval = max(u_pval, eps)
    safe_win_rate = max(win_rate, eps)
    safe_intra_rate = max(intra_win_rate, eps)
    
    # ln(L) penalty high p-val and low win_rate
    ln_L = np.log(excess_factor) + np.log(safe_win_rate) + np.log(safe_intra_rate) - np.log(safe_u_pval)
    
    # complexity k 
    targets = common_config.get("T1_rets", {})
    m_mins = float(tune_config["motif_minutes"])
    dtw_frac = float(common_config["dtw_window_frac"])

    if targets:
        max_offset = max(targets.values())
    else:
        max_offset = 240 
        
    # panelty increase by one hour
    k = (max_offset / 60.0) + (dtw_frac * 10.0) + (m_mins / 30.0)

    # =========================================================================
    # - L = excess / P-value (P small --> L large; excess large --> L large)
    # BIC = -2 * ln(L) + k * ln(n) 
    # - maximize: -BIC = 2 * ln(L) - k * ln(n)
    # =========================================================================
    n = max(2, len(cond_rets))
    neg_bic = 2 * ln_L - k * np.log(n)
    return float(neg_bic)
