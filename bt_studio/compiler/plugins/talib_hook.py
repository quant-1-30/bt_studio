#! /usr/bin/env python3

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import polars as pl

from ..ops import ExprLike, SAFE_OPS, _to_expr
from . import talib_ops

__all__ = ["register_talib_whitelist", "make_talib_fn", "talib_validate_hook"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registration bridge
# ---------------------------------------------------------------------------

def make_talib_fn(spec: talib_ops.TalibOpSpec) -> Callable[..., pl.Expr]:
    """ TA-Lib ---> SAFE_OPS """
    n_expected = len(spec.inputs) + len(spec.params)

    default_vals = [
        getattr(p, "default", None)
        for p in spec.params
        if getattr(p, "default", None) is not None
    ]

    def _fn(*expr_args: ExprLike) -> pl.Expr:
        if len(expr_args) != n_expected:
            raise ValueError(
                f"{spec.op}: expected {n_expected} args "
                f"({len(spec.inputs)} fields + {len(spec.params)} params), "
                f"got {len(expr_args)}"
            )

        field_exprs = [_to_expr(a) for a in expr_args[: len(spec.inputs)]]

        raw_params = list(expr_args[len(spec.inputs) :])
        kwargs: Dict[str, Any] = {}

        for p_spec, val in zip(spec.params, raw_params):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(
                    f"{spec.op}: param '{p_spec.name}' must be a number, got {val!r}"
                )

            # avoid TA-Lib C TypeError
            if isinstance(val, float) and val.is_integer():
                val = int(val)
            elif isinstance(val, float) and getattr(p_spec, "type", None) == "int":
                val = int(round(val))

            kwargs[p_spec.name] = val

        # Polars Struct Batch
        output_idx = spec.output_index if spec.output_index is not None else 0

        def _apply_talib_c(s: pl.Series) -> pl.Series:
            import talib

            func = getattr(talib, spec.func.upper())
            numpy_args = [
                s.struct.field(f"_talib_in_{i}")
                .fill_null(float("nan"))
                .to_numpy()
                .astype(np.float64)
                for i in range(len(spec.inputs))
            ]

            result = func(*numpy_args, **kwargs)

            if isinstance(result, tuple):
                result = result[output_idx]

            return pl.Series(result)

        struct_parts = [e.alias(f"_talib_in_{i}") for i, e in enumerate(field_exprs)]
        return pl.struct(struct_parts).map_batches(_apply_talib_c)

    return _fn


def register_talib_whitelist() -> Dict[str, Dict[str, Any]]:
    if talib_ops.talib is None:
        logger.info("[safeops] talib not installed")
        return {}

    registered: Dict[str, Dict[str, Any]] = {}

    for op, spec in talib_ops.REGISTRY.items():
        n_args = len(spec.inputs) + len(spec.params)
        
        defaults = [
            p.default for p in spec.params 
            if hasattr(p, "default") and p.default is not None
        ]

        min_args = n_args - len(defaults)
        
        entry = {
            "fn": make_talib_fn(spec),
            "arity": (min_args, n_args),
            "defaults": defaults,
        }
        SAFE_OPS[op] = entry
        registered[op] = entry

    if registered:
        names = ", ".join(sorted(registered))
        logger.debug(f"[safeops] talib register: {names}")

    return registered


# ---------------------------------------------------------------------------
# Static validation hook (AST Static Test)
# ---------------------------------------------------------------------------

def talib_validate_hook(ast_node: Any) -> Optional[str]:
    if not isinstance(ast_node, dict) or "op" not in ast_node:
        return None

    op_name = ast_node.get("op")
    if op_name not in talib_ops.REGISTRY:
        return None

    errors = talib_ops.validate_node(ast_node)
    return "; ".join(errors) if errors else None


register_talib_whitelist()
