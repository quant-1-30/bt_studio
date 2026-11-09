#! /usr/bin/env python3

from __future__ import annotations

import numba
import numpy as np
import polars as pl

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

ExprLike = Union[str, int, float, bool, pl.Expr]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_expr(x: ExprLike) -> pl.Expr:
    if isinstance(x, pl.Expr):
        return x
    if isinstance(x, str):
        return pl.col(x)
    if isinstance(x, (int, float, bool)):
        return pl.lit(x)
    raise TypeError(f"Expected str, scalar or pl.Expr, got {type(x).__name__}: {x!r}")


# ---------------------------------------------------------------------------
# Unary operators (arity 1, element-wise)
# ---------------------------------------------------------------------------

def _abs(expr: ExprLike) -> pl.Expr:
    return _to_expr(expr).abs()


def _log(expr: ExprLike) -> pl.Expr:
    base = _to_expr(expr)
    return pl.when(base > 1e-8).then(base.log()).otherwise(None)


def _sign(expr: ExprLike) -> pl.Expr:
    return _to_expr(expr).sign()


def _neg(expr: ExprLike) -> pl.Expr:
    return -_to_expr(expr)


def _sqrt(expr: ExprLike) -> pl.Expr:
    """负数安全映射为 Null。"""
    base = _to_expr(expr)
    return pl.when(base >= 0.0).then(base.sqrt()).otherwise(None)


def _square(expr: ExprLike) -> pl.Expr:
    base = _to_expr(expr)
    return base * base


# ---------------------------------------------------------------------------
# Cumulative operators (arity 1, Time-Series Family)
# ---------------------------------------------------------------------------

def _cum_sum(expr: ExprLike) -> pl.Expr:
    return _to_expr(expr).cum_sum().over("sid")


def _cum_max(expr: ExprLike) -> pl.Expr:
    return _to_expr(expr).cum_max().over("sid")


def _cum_min(expr: ExprLike) -> pl.Expr:
    return _to_expr(expr).cum_min().over("sid")


# ---------------------------------------------------------------------------
# Rolling operators (arity 1..2, Time-Series Family)
# ---------------------------------------------------------------------------

def _ref(expr: ExprLike, n: int = 1) -> pl.Expr:
    return _to_expr(expr).shift(int(n)).over("sid")


def _delta(expr: ExprLike, n: int = 1) -> pl.Expr:
    base = _to_expr(expr)
    return (base - base.shift(int(n))).over("sid")


def _mean(expr: ExprLike, n: int = 5) -> pl.Expr:
    n_i = int(n)
    return _to_expr(expr).rolling_mean(window_size=n_i, min_samples=max(1, n_i // 2)).over("sid")


def _std(expr: ExprLike, n: int = 5) -> pl.Expr:
    n_i = int(n)
    return _to_expr(expr).rolling_std(window_size=n_i, min_samples=max(1, n_i // 2)).over("sid")


def _var(expr: ExprLike, n: int = 5) -> pl.Expr:
    n_i = int(n)
    return _to_expr(expr).rolling_var(window_size=n_i, min_samples=max(1, n_i // 2)).over("sid")


def _max(expr: ExprLike, n: int = 5) -> pl.Expr:
    n_i = int(n)
    return _to_expr(expr).rolling_max(window_size=n_i, min_samples=max(1, n_i // 2)).over("sid")


def _min(expr: ExprLike, n: int = 5) -> pl.Expr:
    n_i = int(n)
    return _to_expr(expr).rolling_min(window_size=n_i, min_samples=max(1, n_i // 2)).over("sid")


def _sum(expr: ExprLike, n: int = 5) -> pl.Expr:
    n_i = int(n)
    return _to_expr(expr).rolling_sum(window_size=n_i, min_samples=max(1, n_i // 2)).over("sid")


def _ema(expr: ExprLike, n: int = 5) -> pl.Expr:
    return _to_expr(expr).ewm_mean(span=int(n), min_samples=1).over("sid")


def _zscore(expr: ExprLike, n: int = 20) -> pl.Expr:
    """ (x - rolling_mean) / rolling_std  [-3.0, 3.0]"""
    base = _to_expr(expr)
    n_i = int(n)
    min_s = max(2, n_i // 2)
    m = base.rolling_mean(window_size=n_i, min_samples=min_s).over("sid")
    s = base.rolling_std(window_size=n_i, min_samples=min_s).over("sid")
    z = pl.when(s > 1e-8).then((base - m) / s).otherwise(0.0)
    return z.clip(-3.0, 3.0)


def _corr(left: ExprLike, right: ExprLike, n: int = 10) -> pl.Expr:
    a = _to_expr(left).cast(pl.Float64)
    b = _to_expr(right).cast(pl.Float64)
    n_i = int(n)
    min_s = max(2, n_i // 2)
    return (
        pl.rolling_corr(a, b, window_size=n_i, min_samples=min_s)
        .over("sid")
        .fill_nan(0.0)
        .clip(-1.0, 1.0)
    )


def _cov(left: ExprLike, right: ExprLike, n: int = 10) -> pl.Expr:
    a = _to_expr(left).cast(pl.Float64)
    b = _to_expr(right).cast(pl.Float64)
    n_i = int(n)
    min_s = max(2, n_i // 2)
    return (
        pl.rolling_cov(a, b, window_size=n_i, min_samples=min_s)
        .over("sid")
        .fill_nan(0.0)
    )


@numba.njit(nogil=True, cache=True)
def _ts_rank_nb(arr: np.ndarray, n_i: int, min_s: int) -> np.ndarray:
    """Compiled GIL-free rolling percentile rank over trailing window n_i."""
    length = len(arr)
    out = np.full(length, np.nan, dtype=np.float64)
    for i in range(length):
        cur = arr[i]
        if np.isnan(cur):
            continue
        start = 0 if i - n_i + 1 < 0 else i - n_i + 1
        n_valid = 0
        less = 0
        equal = 0
        for j in range(start, i + 1):
            v = arr[j]
            if not np.isnan(v):
                n_valid += 1
                if v < cur:
                    less += 1
                elif v == cur:
                    equal += 1
        if n_valid >= min_s:
            out[i] = (less + 0.5 * equal) / n_valid
    return out


def _rank(expr: ExprLike, n: int = 5) -> pl.Expr:
    """Causal time-series rolling percentile rank in [0.0, 1.0] accelerated via Numba."""
    base = _to_expr(expr)
    n_i = int(n)
    min_s = max(2, n_i // 2)

    def _vec_ts_rank(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64, copy=False)
        if len(arr) == 0:
            return pl.Series(values=[], dtype=pl.Float64)
        out = _ts_rank_nb(np.ascontiguousarray(arr, dtype=np.float64), n_i, min_s)
        return pl.Series(out, dtype=pl.Float64).fill_nan(None)

    return base.map_batches(_vec_ts_rank, return_dtype=pl.Float64).over("sid")


# ---------------------------------------------------------------------------
# Cross-sectional operators (Cross-Sectional Family: over ["day", "bar_idx"])
# ---------------------------------------------------------------------------

def _cs_rank(expr: ExprLike) -> pl.Expr:
    base = _to_expr(expr)
    cs_keys = ["day", "bar_idx"]
    valid_count = base.count().over(cs_keys)
    return (
        pl.when(valid_count > 0)
        .then(base.rank(method="average").over(cs_keys) / valid_count)
        .otherwise(None)
    )


def _cs_demean(expr: ExprLike) -> pl.Expr:
    base = _to_expr(expr)
    return (base - base.mean()).over(["day", "bar_idx"])


def _cs_zscore(expr: ExprLike) -> pl.Expr:
    """Z-Score: (x - median) / (1.4826 * MAD) [-3.0, 3.0] """
    base = _to_expr(expr)
    cs_keys = ["day", "bar_idx"]
    med = base.median().over(cs_keys)
    mad = (base - med).abs().median().over(cs_keys)
    scale = 1.4826 * mad

    z = pl.when(mad < 1e-7).then(0.0).otherwise((base - med) / scale)
    return z.clip(-3.0, 3.0)


# ---------------------------------------------------------------------------
# Momentum & TA-Lib Style Operators
# ---------------------------------------------------------------------------

def _roc(close_expr: ExprLike, n: int = 12) -> pl.Expr:
    """Rate of Change (ROC)"""
    close = _to_expr(close_expr)
    n_i = int(n)
    prev = close.shift(n_i).over("sid")
    return (
        pl.when(prev.is_null() | close.is_null())
        .then(None)
        .when(prev.abs() > 1e-8)
        .then((close - prev) / prev * 100.0)
        .otherwise(0.0)
    )


def _atr(
    high_expr: ExprLike,
    low_expr: ExprLike,
    close_expr: ExprLike,
    n: int = 14,
) -> pl.Expr:
    """Average True Range (ATR)。"""
    high = _to_expr(high_expr)
    low = _to_expr(low_expr)
    close = _to_expr(close_expr)
    n_i = int(n)

    prev_close = close.shift(1).over("sid")
    tr = pl.max_horizontal(
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    )
    return tr.rolling_mean(window_size=n_i, min_samples=max(1, n_i // 2)).over("sid")


def _stoch_k(
    high_expr: ExprLike,
    low_expr: ExprLike,
    close_expr: ExprLike,
    n: int = 14,
) -> pl.Expr:
    """Stochastic Oscillator (%K)"""
    high = _to_expr(high_expr)
    low = _to_expr(low_expr)
    close = _to_expr(close_expr)
    n_i = int(n)

    highest = high.rolling_max(window_size=n_i, min_samples=max(1, n_i // 2)).over("sid")
    lowest = low.rolling_min(window_size=n_i, min_samples=max(1, n_i // 2)).over("sid")
    diff = highest - lowest

    return (
        pl.when(diff < 1e-8)
        .then(50.0)
        .otherwise((close - lowest) / diff * 100.0)
        .over("sid")
    )


# ---------------------------------------------------------------------------
# Pair operators (arity 2, element-wise)
# ---------------------------------------------------------------------------

def _add(left: ExprLike, right: ExprLike) -> pl.Expr:
    return _to_expr(left) + _to_expr(right)


def _sub(left: ExprLike, right: ExprLike) -> pl.Expr:
    return _to_expr(left) - _to_expr(right)


def _mul(left: ExprLike, right: ExprLike) -> pl.Expr:
    return _to_expr(left) * _to_expr(right)


def _div(left: ExprLike, right: ExprLike) -> pl.Expr:
    l = _to_expr(left)
    r = _to_expr(right)
    return pl.when(r.abs() > 1e-8).then(l / r).otherwise(None)


def _gte(left: ExprLike, right: ExprLike) -> pl.Expr:
    return (_to_expr(left) >= _to_expr(right)).cast(pl.Float64)


def _lte(left: ExprLike, right: ExprLike) -> pl.Expr:
    return (_to_expr(left) <= _to_expr(right)).cast(pl.Float64)


# ---------------------------------------------------------------------------
# Macro & Multi-Dimensional Covariance Operator
# ---------------------------------------------------------------------------

def _mdp_weight(sa_expr: ExprLike, imp_expr: ExprLike, liq_expr: ExprLike) -> pl.Expr:
    sa = _to_expr(sa_expr)
    imp = _to_expr(imp_expr)
    liq = _to_expr(liq_expr)
    min_w = 0.05
    cs_keys = ["day", "bar_idx"]

    sa_dm = sa - sa.mean().over(cs_keys)
    imp_dm = imp - imp.mean().over(cs_keys)
    liq_dm = liq - liq.mean().over(cs_keys)

    rho_sa_imp = (sa_dm * imp_dm).mean().over(cs_keys)
    rho_sa_liq = (sa_dm * liq_dm).mean().over(cs_keys)
    rho_imp_liq = (imp_dm * liq_dm).mean().over(cs_keys)

    w_sa_raw = 1.0 - rho_imp_liq**2 - rho_sa_imp - rho_sa_liq + rho_sa_imp * rho_imp_liq + rho_sa_liq * rho_imp_liq
    w_imp_raw = 1.0 - rho_sa_liq**2 - rho_sa_imp - rho_imp_liq + rho_sa_imp * rho_sa_liq + rho_sa_liq * rho_imp_liq
    w_liq_raw = 1.0 - rho_sa_imp**2 - rho_sa_liq - rho_imp_liq + rho_sa_imp * rho_sa_liq + rho_sa_imp * rho_imp_liq

    w_sa_pos = pl.max_horizontal(w_sa_raw, pl.lit(min_w))
    w_imp_pos = pl.max_horizontal(w_imp_raw, pl.lit(min_w))
    w_liq_pos = pl.max_horizontal(w_liq_raw, pl.lit(min_w))

    w_sum = w_sa_pos + w_imp_pos + w_liq_pos

    return (
        pl.when(w_sum > 1e-8)
        .then((w_sa_pos / w_sum) * sa + (w_imp_pos / w_sum) * imp + (w_liq_pos / w_sum) * liq)
        .otherwise((sa + imp + liq) / 3.0)
    )


# ---------------------------------------------------------------------------
# Operator Registry & Arity Specifications (SSOT)
# ---------------------------------------------------------------------------

SAFE_OPS: Dict[str, Dict[str, Any]] = {
    # ------------------ 一元算子 (1, 1, family: element) ------------------
    "abs":        {"fn": _abs,        "arity": (1, 1), "defaults": [],   "family": "elem"},
    "log":        {"fn": _log,        "arity": (1, 1), "defaults": [],   "family": "elem"},
    "sign":       {"fn": _sign,       "arity": (1, 1), "defaults": [],   "family": "elem"},
    "neg":        {"fn": _neg,        "arity": (1, 1), "defaults": [],   "family": "elem"},
    "sqrt":       {"fn": _sqrt,       "arity": (1, 1), "defaults": [],   "family": "elem"},
    "square":     {"fn": _square,     "arity": (1, 1), "defaults": [],   "family": "elem"},

    # ------------------ 累计算子 (1, 1, family: ts) ------------------
    "cum_sum":    {"fn": _cum_sum,    "arity": (1, 1), "defaults": [],   "family": "ts"},
    "cum_max":    {"fn": _cum_max,    "arity": (1, 1), "defaults": [],   "family": "ts"},
    "cum_min":    {"fn": _cum_min,    "arity": (1, 1), "defaults": [],   "family": "ts"},

    # ------------------ 滚动时序算子 (1, 2, family: ts) ------------------
    "ref":        {"fn": _ref,        "arity": (1, 2), "defaults": [1],  "family": "ts"},
    "delta":      {"fn": _delta,      "arity": (1, 2), "defaults": [1],  "family": "ts"},
    "mean":       {"fn": _mean,       "arity": (1, 2), "defaults": [5],  "family": "ts"},
    "std":        {"fn": _std,        "arity": (1, 2), "defaults": [5],  "family": "ts"},
    "var":        {"fn": _var,        "arity": (1, 2), "defaults": [5],  "family": "ts"},
    "max":        {"fn": _max,        "arity": (1, 2), "defaults": [5],  "family": "ts"},
    "min":        {"fn": _min,        "arity": (1, 2), "defaults": [5],  "family": "ts"},
    "sum":        {"fn": _sum,        "arity": (1, 2), "defaults": [5],  "family": "ts"},
    "ema":        {"fn": _ema,        "arity": (1, 2), "defaults": [5],  "family": "ts"},
    "rank":       {"fn": _rank,       "arity": (1, 2), "defaults": [5],  "family": "ts"},
    "zscore":     {"fn": _zscore,     "arity": (1, 2), "defaults": [20], "family": "ts"},

    # ------------------ 双序列滚动时序算子 (2, 3, family: ts) ------------------
    "corr":       {"fn": _corr,       "arity": (2, 3), "defaults": [10], "family": "ts"},
    "cov":        {"fn": _cov,        "arity": (2, 3), "defaults": [10], "family": "ts"},

    # ------------------ 截面算子 (1, 1, family: cs) ------------------
    "cs_rank":    {"fn": _cs_rank,    "arity": (1, 1), "defaults": [],   "family": "cs"},
    "cs_demean":  {"fn": _cs_demean,  "arity": (1, 1), "defaults": [],   "family": "cs"},
    "cs_zscore":  {"fn": _cs_zscore,  "arity": (1, 1), "defaults": [],   "family": "cs"},

    # ------------------ 二元点对算子 (2, 2, family: element) ------------------
    "add":        {"fn": _add,        "arity": (2, 2), "defaults": [],   "family": "elem"},
    "sub":        {"fn": _sub,        "arity": (2, 2), "defaults": [],   "family": "elem"},
    "mul":        {"fn": _mul,        "arity": (2, 2), "defaults": [],   "family": "elem"},
    "div":        {"fn": _div,        "arity": (2, 2), "defaults": [],   "family": "elem"},
    "gte":        {"fn": _gte,        "arity": (2, 2), "defaults": [],   "family": "elem"},
    "lte":        {"fn": _lte,        "arity": (2, 2), "defaults": [],   "family": "elem"},

    # ------------------ 动量与微观算子 (family: ts) ------------------
    "roc":        {"fn": _roc,        "arity": (1, 2), "defaults": [12], "family": "ts"},
    "atr":        {"fn": _atr,        "arity": (3, 4), "defaults": [14], "family": "ts"},
    "stoch_k":    {"fn": _stoch_k,    "arity": (3, 4), "defaults": [14], "family": "ts"},

    # ------------------ 宏观协方差加权算子 (family: cs) ------------------
    "mdp_weight": {"fn": _mdp_weight, "arity": (3, 3), "defaults": [],   "family": "cs"},
}


def list_ops() -> List[str]:
    return sorted(SAFE_OPS.keys())


def compile_expr(ast_node: Any, check_causal: bool = True) -> pl.Expr:
    """AST compiler nodes directly to Polars expressions (default enabled causal guard, block forward view bypass)."""
    from .ast import compile_ast

    expr, _ = compile_ast(ast_node, check_causal=check_causal)
    return expr
