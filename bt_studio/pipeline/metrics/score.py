
import numpy as np


def calculate_hpo_score(
    u_pval: float, 
    trigger_count: int, 
    cond_rets: np.ndarray, 
    uncond_rets: np.ndarray, 
    tune_config: dict,
    common_config: dict
    ) -> float:
    # standard for pure score avoid subjective penalty
    if u_pval > 0.10 or trigger_count < 5:
        return 0.0

    # low pval --> high score
    p_score = -np.log10(max(u_pval, 1e-10))
    
    # Excess Return determines score
    excess_ret = np.mean(cond_rets) - np.mean(uncond_rets)

    # =========================================================================
    # A Long-Only 
    # =========================================================================
    alternative = common_config["alternative"]

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

    # raw_score = float(p_score * n_penalty * excess_factor * 10000.0)
    raw_score = float(p_score * excess_factor * 10000.0)

    n_samples = len(cond_rets) 
    bic_score = bic_like_score(raw_score, tune_config, n_samples)
    return bic_score 


def bic_like_score(raw_score: float, tune_config: dict, n_samples: int) -> float:
    if raw_score <= 1e-6: 
        return 0.0

    # complexity k
    cross_days = float(tune_config.get("cross_days", 1.0))
    dtw_frac = float(tune_config.get("dtw_window_frac", 0.10))
    m_mins = float(tune_config.get("motif_minutes", 60.0))
    k = cross_days + (dtw_frac * 10.0) + (m_mins / 30.0)
    
    n = max(2, n_samples)
    
    # BIC = -2 * ln(L) + k * ln(n) 
    ln_L = np.log(raw_score)
    # minimize BIC / maximize -BIC
    neg_bic = 2 * ln_L - k * np.log(n)
    
    # exp Optuna
    bic_exp_score = np.exp(neg_bic / 10.0) * 1000.0
    return float(bic_exp_score)
