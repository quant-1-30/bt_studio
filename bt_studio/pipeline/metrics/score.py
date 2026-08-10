
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
    
    gap_z_lower = common_config.get("gap_z_lower", -2.0)
    gap_z_upper = common_config.get("gap_z_upper", 3.0)
    is_executed = (cond_z_gaps >= gap_z_lower) & (cond_z_gaps <= gap_z_upper)
    execution_rate = np.mean(is_executed) if len(is_executed) > 0 else 0.0

    realized_rets = np.where(is_executed, cond_rets, 0.0)
    
    excess_ret = np.median(realized_rets) - np.median(uncond_rets) 
    win_rate = np.mean(realized_rets > 0)
    intra_win_rate = np.mean(cond_intra > 0)
    eps = common_config.get("eps", 1e-4)
    
    alternative = common_config["alternative"]
    if alternative == "greater":
        excess_factor = max(excess_ret, eps) 
    elif alternative == "less":
        excess_factor = max(-excess_ret, eps)
    else: 
        excess_factor = max(abs(excess_ret), eps)
        
    safe_u_pval = max(u_pval, 1e-10)
    safe_win_rate = max(win_rate, eps)
    safe_intra_rate = max(intra_win_rate, eps)
    safe_exec_rate = max(execution_rate, eps)
    
    ln_L = np.log(excess_factor) + np.log(safe_win_rate) + np.log(safe_intra_rate) + np.log(safe_exec_rate) - np.log(safe_u_pval)
    
    # complexity k 
    targets = common_config.get("T1_rets", {})
    m_mins = float(tune_config["motif_minutes"])
    dtw_frac = float(common_config["dtw_window_frac"])
    max_offset = max(targets.values()) if targets else 240
    
    k = (max_offset / 60.0) + (dtw_frac * 10.0) + (m_mins / 30.0)
    # =========================================================================
    # - L = excess / P-value (P small --> L large; excess large --> L large)
    # BIC = -2 * ln(L) + k * ln(n) 
    # - maximize: -BIC = 2 * ln(L) - k / ln(n)
    # =========================================================================

    n = max(2, len(cond_rets))
    score = ln_L - (k / np.sqrt(n))
    
    return float(score)
