#! /usr/bin/env python3

from __future__ import annotations

import polars as pl
import talib
import numpy as np

from typing import Any, Callable, Dict, Union


ExprLike = Union[str, int, float, bool, pl.Expr]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_expr(x: ExprLike) -> pl.Expr:
    """
    ---> pl.Expr:
    - str        -> pl.col(str)
    - int/float  -> pl.lit(val)
    - pl.Expr    -> return
    """
    if isinstance(x, pl.Expr):
        return x
    if isinstance(x, str):
        return pl.col(x)
    if isinstance(x, (int, float, bool)):
        return pl.lit(x)
    raise TypeError(f"Expected str, scalar or pl.Expr, got {type(x).__name__}: {x!r}")


# ---------------------------------------------------------------------------
# Unary operators (arity 1)
# ---------------------------------------------------------------------------

def _abs(expr: ExprLike) -> pl.Expr:
    return _to_expr(expr).abs()


def _log(expr: ExprLike) -> pl.Expr:
    """negative """
    base = _to_expr(expr)
    return pl.when(base > 1e-8).then(base.log()).otherwise(None)


def _sign(expr: ExprLike) -> pl.Expr:
    return _to_expr(expr).sign()


def _neg(expr: ExprLike) -> pl.Expr:
    return -_to_expr(expr)


def _sqrt(expr: ExprLike) -> pl.Expr:
    """negative None"""
    base = _to_expr(expr)
    return pl.when(base >= 0.0).then(base.sqrt()).otherwise(None)


def _square(expr: ExprLike) -> pl.Expr:
    base = _to_expr(expr)
    return base * base


# ---------------------------------------------------------------------------
# Cumulative operators (arity 1)
# ---------------------------------------------------------------------------

def _cum_sum(expr: ExprLike) -> pl.Expr:
    return _to_expr(expr).cum_sum()


def _cum_max(expr: ExprLike) -> pl.Expr:
    return _to_expr(expr).cum_max()


def _cum_min(expr: ExprLike) -> pl.Expr:
    return _to_expr(expr).cum_min()


# ---------------------------------------------------------------------------
# Rolling operators (arity 1..2): (expr, N=default)
# ---------------------------------------------------------------------------

def _ref(expr: ExprLike, n: int = 1) -> pl.Expr:
    return _to_expr(expr).shift(n)


def _delta(expr: ExprLike, n: int = 1) -> pl.Expr:
    base = _to_expr(expr)
    return base - base.shift(n)


def _mean(expr: ExprLike, n: int = 5) -> pl.Expr:
    return _to_expr(expr).rolling_mean(window_size=n, min_periods=max(1, n // 2))


def _std(expr: ExprLike, n: int = 5) -> pl.Expr:
    return _to_expr(expr).rolling_std(window_size=n, min_periods=max(1, n // 2))


def _var(expr: ExprLike, n: int = 5) -> pl.Expr:
    return _to_expr(expr).rolling_var(window_size=n, min_periods=max(1, n // 2))


def _max(expr: ExprLike, n: int = 5) -> pl.Expr:
    return _to_expr(expr).rolling_max(window_size=n, min_periods=max(1, n // 2))


def _min(expr: ExprLike, n: int = 5) -> pl.Expr:
    return _to_expr(expr).rolling_min(window_size=n, min_periods=max(1, n // 2))


def _sum(expr: ExprLike, n: int = 5) -> pl.Expr:
    return _to_expr(expr).rolling_sum(window_size=n, min_periods=max(1, n // 2))


def _ema(expr: ExprLike, n: int = 5) -> pl.Expr:
    return _to_expr(expr).ewm_mean(span=n, min_periods=1)


def _rank(expr: ExprLike, n: int = 5) -> pl.Expr:
    """Ts_Rank"""
    def _calc_rank(s: pl.Series) -> float:
        valid = s.drop_nulls()
        if len(valid) == 0:
            return float("nan")
        return float(valid.rank(method="average")[-1] / len(valid))

    return _to_expr(expr).rolling_map(
        _calc_rank,
        window_size=n,
        min_periods=max(2, n // 2),
    )


# ---------------------------------------------------------------------------
# Cross-sectional operators (across instruments, same day)
# ---------------------------------------------------------------------------

def _cs_rank(expr: ExprLike) -> pl.Expr:
    """[0, 1]"""
    base = _to_expr(expr)
    return (base.rank(method="average") / pl.len()).over("day")


def _cs_demean(expr: ExprLike) -> pl.Expr:
    base = _to_expr(expr)
    return base - base.mean().over("day")


def _cs_zscore(expr: ExprLike) -> pl.Expr:
    """
    Z-Score: (x - median) / (1.4826 * MAD)
    (MAD < eps) ---> 0.0
    """
    base = _to_expr(expr)
    med = base.median().over("day")
    mad = (base - med).abs().median().over("day")
    eps = 1e-7

    scale = 1.4826 * mad
    z = pl.when(mad < eps).then(0.0).otherwise((base - med) / scale)
    return z.clip(-3.0, 3.0)


# ---------------------------------------------------------------------------
# TA-Lib Style Momentum Operators (variable arity)
# ---------------------------------------------------------------------------

def _roc(close_expr: ExprLike, n: int = 12) -> pl.Expr:
    """Rate of Change (ROC)"""
    close = _to_expr(close_expr)
    prev = close.shift(n)
    return pl.when(prev.abs() > 1e-8).then((close - prev) / prev * 100.0).otherwise(0.0)


def _atr(
    high_expr: ExprLike,
    low_expr: ExprLike,
    close_expr: ExprLike,
    n: int = 14,
) -> pl.Expr:
    """Average True Range (ATR)"""
    high = _to_expr(high_expr)
    low = _to_expr(low_expr)
    close = _to_expr(close_expr)

    prev_close = close.shift(1)
    tr = pl.max_horizontal(
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    )
    return tr.rolling_mean(window_size=n, min_periods=max(1, n // 2))


def _stoch_k(
    high_expr: ExprLike,
    low_expr: ExprLike,
    close_expr: ExprLike,
    n: int = 14,
) -> pl.Expr:
    
    high = _to_expr(high_expr)
    low = _to_expr(low_expr)
    close = _to_expr(close_expr)

    highest = high.rolling_max(window_size=n, min_periods=max(1, n // 2))
    lowest = low.rolling_min(window_size=n, min_periods=max(1, n // 2))
    diff = highest - lowest

    return pl.when(diff < 1e-8).then(50.0).otherwise((close - lowest) / diff * 100.0)


# ---------------------------------------------------------------------------
# Pair operators (arity 2): (left, right)
# ---------------------------------------------------------------------------

def _add(left: ExprLike, right: ExprLike) -> pl.Expr:
    return _to_expr(left) + _to_expr(right)


def _sub(left: ExprLike, right: ExprLike) -> pl.Expr:
    return _to_expr(left) - _to_expr(right)


def _mul(left: ExprLike, right: ExprLike) -> pl.Expr:
    return _to_expr(left) * _to_expr(right)


def _div(left: ExprLike, right: ExprLike) -> pl.Expr:
    """safe div 0"""
    l = _to_expr(left)
    r = _to_expr(right)
    return pl.when(r.abs() > 1e-8).then(l / r).otherwise(None)


def _gte(left: ExprLike, right: ExprLike) -> pl.Expr:
    return (_to_expr(left) >= _to_expr(right)).cast(pl.Float64)


def _lte(left: ExprLike, right: ExprLike) -> pl.Expr:
    return (_to_expr(left) <= _to_expr(right)).cast(pl.Float64)


# ---------------------------------------------------------------------------
# MD (Multi-Dimensional) Covariance Weighting Macro Operator
# ---------------------------------------------------------------------------

def _mdp_weight(sa_expr: ExprLike, imp_expr: ExprLike, liq_expr: ExprLike) -> pl.Expr:
    sa = _to_expr(sa_expr)
    imp = _to_expr(imp_expr)
    liq = _to_expr(liq_expr)
    min_w = 0.05

    # collation
    rho_sa_imp = (sa * imp).mean().over(["day", "bar_idx"])
    rho_sa_liq = (sa * liq).mean().over(["day", "bar_idx"])
    rho_imp_liq = (imp * liq).mean().over(["day", "bar_idx"])

    # reverse matrix
    w_sa_raw = 1.0 - rho_imp_liq**2 - rho_sa_imp - rho_sa_liq + rho_sa_imp * rho_imp_liq + rho_sa_liq * rho_imp_liq
    w_imp_raw = 1.0 - rho_sa_liq**2 - rho_sa_imp - rho_imp_liq + rho_sa_imp * rho_sa_liq + rho_sa_liq * rho_imp_liq
    w_liq_raw = 1.0 - rho_sa_imp**2 - rho_sa_liq - rho_imp_liq + rho_sa_imp * rho_sa_liq + rho_sa_imp * rho_imp_liq

    w_sa_pos = pl.max_horizontal(w_sa_raw, pl.lit(min_w))
    w_imp_pos = pl.max_horizontal(w_imp_raw, pl.lit(min_w))
    w_liq_pos = pl.max_horizontal(w_liq_raw, pl.lit(min_w))

    w_sum = w_sa_pos + w_imp_pos + w_liq_pos

    return (w_sa_pos / w_sum) * sa + (w_imp_pos / w_sum) * imp + (w_liq_pos / w_sum) * liq


# ---------------------------------------------------------------------------
# Universal TA-Lib Bridge
# ---------------------------------------------------------------------------

def _talib(func_name: str, *expr_args: ExprLike, **kwargs) -> pl.Expr:

    talib_func = getattr(talib, func_name.upper(), None)
    if talib_func is None:
        all_funcs = talib.get_functions()
        match = [f for f in all_funcs if f.upper() == func_name.upper()]
        if match:
            talib_func = getattr(talib, match[0])
        else:
            raise ValueError(f"TA-Lib '{func_name}' no found")

    compiled_exprs = [_to_expr(e) for e in expr_args]
    n_args = len(compiled_exprs)

    def _apply_talib(s: pl.Series) -> pl.Series:
        numpy_args = []
        for i in range(n_args):
            # fill_null(NaN) 
            field_series = s.struct.field(f"_talib_in_{i}").fill_null(float("nan"))
            numpy_args.append(field_series.to_numpy().astype(np.float64))

        result = talib_func(*numpy_args, **kwargs)
        if isinstance(result, tuple):
            result = result[0]

        return pl.Series(result)

    struct_parts = [e.alias(f"_talib_in_{i}") for i, e in enumerate(compiled_exprs)]
    return pl.struct(struct_parts).map_batches(_apply_talib)


# ---------------------------------------------------------------------------
# Operator Registry & Arity Specifications
# ---------------------------------------------------------------------------

SAFE_OPS: Dict[str, Dict[str, Any]] = {
    # ------------------ Unary (1, 1) ------------------
    "abs":        {"fn": _abs,        "arity": (1, 1), "defaults": []},
    "log":        {"fn": _log,        "arity": (1, 1), "defaults": []},
    "sign":       {"fn": _sign,       "arity": (1, 1), "defaults": []},
    "neg":        {"fn": _neg,        "arity": (1, 1), "defaults": []},
    "sqrt":       {"fn": _sqrt,       "arity": (1, 1), "defaults": []},
    "square":     {"fn": _square,     "arity": (1, 1), "defaults": []},

    # ------------------ Cumulative (1, 1) ------------------
    "cum_sum":    {"fn": _cum_sum,    "arity": (1, 1), "defaults": []},
    "cum_max":    {"fn": _cum_max,    "arity": (1, 1), "defaults": []},
    "cum_min":    {"fn": _cum_min,    "arity": (1, 1), "defaults": []},

    # ------------------ Rolling (1, 2) ------------------
    "ref":        {"fn": _ref,        "arity": (1, 2), "defaults": [1]},    # 缺省回看 1 期
    "delta":      {"fn": _delta,      "arity": (1, 2), "defaults": [1]},    # 缺省差分 1 期
    "mean":       {"fn": _mean,       "arity": (1, 2), "defaults": [5]},    # 缺省窗口 5
    "std":        {"fn": _std,        "arity": (1, 2), "defaults": [5]},    # 缺省窗口 5
    "var":        {"fn": _var,        "arity": (1, 2), "defaults": [5]},    # 缺省窗口 5
    "max":        {"fn": _max,        "arity": (1, 2), "defaults": [5]},    # 缺省窗口 5
    "min":        {"fn": _min,        "arity": (1, 2), "defaults": [5]},    # 缺省窗口 5
    "sum":        {"fn": _sum,        "arity": (1, 2), "defaults": [5]},    # 缺省窗口 5
    "ema":        {"fn": _ema,        "arity": (1, 2), "defaults": [5]},    # 缺省窗口 5
    "rank":       {"fn": _rank,       "arity": (1, 2), "defaults": [5]},    # 缺省时序排名窗口 5

    # ------------------ Cross-sectional (1, 1) ------------------
    "cs_rank":    {"fn": _cs_rank,    "arity": (1, 1), "defaults": []},
    "cs_demean":  {"fn": _cs_demean,  "arity": (1, 1), "defaults": []},
    "cs_zscore":  {"fn": _cs_zscore,  "arity": (1, 1), "defaults": []},

    # ------------------ Pair-wise (2, 2) ------------------
    "add":        {"fn": _add,        "arity": (2, 2), "defaults": []},
    "sub":        {"fn": _sub,        "arity": (2, 2), "defaults": []},
    "mul":        {"fn": _mul,        "arity": (2, 2), "defaults": []},
    "div":        {"fn": _div,        "arity": (2, 2), "defaults": []},
    "gte":        {"fn": _gte,        "arity": (2, 2), "defaults": []},
    "lte":        {"fn": _lte,        "arity": (2, 2), "defaults": []},

    # ------------------ Momentum (Vary) ------------------
    "roc":        {"fn": _roc,        "arity": (1, 2), "defaults": [12]},   # 缺省周期 12
    "atr":        {"fn": _atr,        "arity": (3, 4), "defaults": [14]},   # [high, low, close, n=14]
    "stoch_k":    {"fn": _stoch_k,    "arity": (3, 4), "defaults": [14]},   # [high, low, close, n=14]

    # ------------------ Macro / Covariance (3, 3) ------------------
    "mdp_weight": {"fn": _mdp_weight, "arity": (3, 3), "defaults": []},   # [sa, imp, liq]

    # ------------------ Universal TA-Lib Bridge ------------------
    # (func_name + >=1 input)
    "talib":      {"fn": _talib,      "arity": (2, None), "defaults": []},
}

def list_ops() -> list[str]:
    return sorted(SAFE_OPS.keys())


def compile_expr(ast_node: Any) -> pl.Expr:
    """AST Node ---> pl.Expr"""
    from .ast import compile_ast

    expr, _ = compile_ast(ast_node, check_causal=False)
    return expr

from dataclasses import dataclass
from typing import Callable, Dict, List
import polars as pl


@dataclass
class FeatureSpec:
    name: str
    build_fn: Callable
    output_col: str
    description: str = ""
    causal: bool = True


class FeatureRegistry:
    def __init__(self):
        self._specs: Dict[str, FeatureSpec] = {}

    def register(self, spec: FeatureSpec):
        if spec.name in self._specs:
            raise ValueError(f"Feature {spec.name!r} already registered")
        if not spec.causal:
            print(f"  WARNING [Registry] Feature {spec.name!r} declares causal=False")
        self._specs[spec.name] = spec

    def get(self, name: str) -> FeatureSpec:
        return self._specs[name]

    def list_features(self) -> List[FeatureSpec]:
        return list(self._specs.values())

    def names(self) -> List[str]:
        return list(self._specs.keys())

    def __len__(self):
        return len(self._specs)

    def __contains__(self, name):
        return name in self._specs
