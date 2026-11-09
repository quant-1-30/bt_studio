#! /usr/bin/env python3

from typing import Optional, Union
import numpy as np

from bt_studio.schema import HPOScoreInputs


def calculate_hpo_score(
    u_pval: Union[float, HPOScoreInputs],
    trigger_count: int = 0,
    cond_rets: Optional[np.ndarray] = None,
    uncond_rets: Optional[np.ndarray] = None,
    cond_momentum: Optional[np.ndarray] = None,
    tune_config: Optional[dict] = None,
    common_config: Optional[dict] = None,
) -> float:
    """
      base_score = ln(1+ excess_positive_bp ) - ln(1+|excess_negative_bp|)   
                 + ln(winrate) + ln(momentum_win_rate)             
                 + pval
      score = base_score - k/sqrt(n)                        
    u_pval (rank_trajectory)
    """
    if isinstance(u_pval, HPOScoreInputs):
        inp = u_pval
        u_pval = inp.u_pval
        trigger_count = inp.trigger_count
        cond_rets = inp.cond_rets
        uncond_rets = inp.uncond_rets
        cond_momentum = inp.cond_momentum
        tune_config = inp.tune_config
        common_config = inp.common_config

    tune_config = tune_config or {}
    common_config = common_config or {}
    if cond_rets is None or cond_momentum is None:
        return -500.0
    if uncond_rets is None:
        uncond_rets = np.empty(0, dtype=np.float64)
    n_samples = int(trigger_count)
    if n_samples <= 0 or len(cond_rets) == 0:
        return -500.0

    n_ret = len(cond_rets)
    n_momentum = len(cond_momentum)
    if n_ret == 0 or n_momentum == 0:
        return -500.0

    # 1. Jeffreys Beta (k + 0.5) / (n + 1.0) 
    excess_ret = float(np.median(cond_rets) - np.median(uncond_rets)) if len(uncond_rets) > 0 else 0.0
    _dir = -1.0 if common_config.get("alternative", "greater") == "less" else 1.0
    win_rate = (np.sum(_dir * cond_rets > 0) + 0.5) / (n_ret + 1.0)
    momentum_win_rate = (np.sum(_dir * cond_momentum > 0) + 0.5) / (n_momentum + 1.0)

    # cond_rets (raw_trajectory)
    # ln(1+x_bp)
    #     (fanova worst_drop 2xIQR) misjudge ISOLATED_SPIKE
    alternative = common_config.get("alternative", "greater")
    excess_bp = excess_ret * 1e4
    if alternative == "less":
        excess_bp = -excess_bp
    elif alternative == "two-sided":
        excess_bp = abs(excess_bp)
    pos = max(excess_bp, 0.0)
    neg = max(-excess_bp, 0.0)
    excess_term = float(np.log1p(pos) - np.log1p(neg))

    # 3. p-value soft saturation: avoid extremely small p-values (e.g., 1e-50) overwhelming the excess return and win rate terms
    safe_u_pval = max(float(u_pval), 1e-300)
    max_neg_ln_p = float(common_config.get("max_neg_ln_p", 10.0))
    neg_ln_p = max_neg_ln_p * (1.0 - np.exp(-(-np.log(safe_u_pval)) / max_neg_ln_p))

    base_score = (
        excess_term
        + np.log(win_rate)
        + np.log(momentum_win_rate)
        + neg_ln_p
    )

    # 4. Occam's Razor complexity penalty k / sqrt(n)
    m_mins = float(tune_config.get("motif_minutes", 30))
    k = m_mins / 30.0
    n = max(2, n_ret)
    return float(base_score - (k / np.sqrt(n)))
