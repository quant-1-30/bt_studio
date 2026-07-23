#! /usr/bin/env python3
# -*- encondig: utf-8 -*-

import numpy as np


def calculate_decay_weights(
    rets_window: dict[str, int], 
    half_life_minutes: float = 15.0
) -> dict[str, float]:
    """
    T+1 open Offset Targets exp weight

    Parameters
    ----------
    rets_window : dict[str, int]
        {"open_5m": 5, "open_15m": 15, "open_30m": 30}
    half_life_minutes : float, optional
        default: 30

    Returns
    -------
    dict[str, float]
    """
    if not rets_window:
        return {}

    decay_const = np.log(2) / half_life_minutes

    names = list(rets_window.keys())
    minutes = np.array(list(rets_window.values()), dtype=np.float64)

    raw_weights = np.exp(-decay_const * minutes)

    sum_w = np.sum(raw_weights)
    norm_weights = raw_weights / sum_w

    return {name: float(w) for name, w in zip(names, norm_weights)}
