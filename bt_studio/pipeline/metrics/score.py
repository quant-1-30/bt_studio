
import numpy as np


def calculate_hpo_score(
    u_pval: float, 
    trigger_count: int, 
    cond_rets: np.ndarray, 
    uncond_rets: np.ndarray, 
    alternative: str
    ) -> float:

    if u_pval > 0.10 or trigger_count < 5:
        return 0.0

    # low pval --> high score
    p_score = -np.log10(max(u_pval, 1e-10))
    
    # sample penaly 
    n_penalty = np.sqrt(trigger_count)
    
    # Excess Return determines score
    excess_ret = np.mean(cond_rets) - np.mean(uncond_rets)
    
    # =========================================================================
    # A Long-Only 
    # =========================================================================
    if alternative == "greater":
        # cond > uncond 
        excess_factor = max(excess_ret, 0.0)
    elif alternative == "less":
        excess_factor = max(-excess_ret, 0.0)
    else: 
        # "two-sided" 
        excess_factor = abs(excess_ret)
        
    if excess_factor <= 1e-6:
        return 0.0
        
    return float(p_score * n_penalty * excess_factor * 10000.0)
