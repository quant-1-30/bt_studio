
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
    final_score = bic_like_score(raw_score, tune_config, n_samples)
    return bic_score 


# def bic_score(raw_score: float, tune_config: dict):
#     """
#         # BIC = -2 \ln(\hat{L}) + k \ln(n) $$
#         * **$\hat{L}$ (LikeHood)**: `raw_score` (P-value  $\times$ Excess Return)
#         * **$n$ **: `trigger_count`。
#         * **$k$ **: $k = \text{cross\_days} + (\text{dtw\_window\_frac} \times 10) + (\text{motif\_minutes} / 30)$
#     """

#     if raw_score <= 1e-6: return 0.0

#     # k logic
#     cross_days = float(tune_config.get("cross_days", 1.0))
#     dtw_frac = float(tune_config.get("dtw_window_frac", 0.10))
#     m_mins = float(tune_config.get("motif_minutes", 60.0))
#     k = cross_days + (dtw_frac * 10.0) + (m_mins / 30.0)
    
#     n = max(2, trigger_count)
#     ln_L = np.log(raw_score)
    
#     bic = -2 * ln_L + k * np.log(n)
    
#     # bic less means better and exp ensure + ensure Optuna
#     final_score = np.exp(-bic / 10.0) * 10000.0 
#     return float(final_score)


def bic_like_score(raw_score: float, tune_config: dict, n_samples: int) -> float:
    if raw_score <= 1e-6: 
        return 0.0

    # complexity k
    cross_days = float(tune_config.get("cross_days", 1.0))
    dtw_frac = float(tune_config.get("dtw_window_frac", 0.10))
    m_mins = float(tune_config.get("motif_minutes", 60.0))
    k = cross_days + (dtw_frac * 10.0) + (m_mins / 30.0)
    
    n = max(2, n_samples)
    
    # BIC = -2 * ln(L) + k * ln(n) and minimize BIC / maximize -BIC
    ln_L = np.log(raw_score)
    neg_bic = 2 * ln_L - k * np.log(n)
    
    # exp Optuna
    final_score = np.exp(neg_bic / 10.0) * 1000.0
    return float(final_score)
