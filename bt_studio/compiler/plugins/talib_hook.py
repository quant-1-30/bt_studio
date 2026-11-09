#! /usr/bin/env python3

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from . import talib_ops
from typing import Any, Callable, Dict, List, Optional, Union

__all__ = ["register_talib_whitelist", "make_talib_fn", "talib_validate_hook"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal Helper (解耦对 ops._to_expr 的直接依赖，阻断循环导入)
# ---------------------------------------------------------------------------

def _ensure_expr(x: Any) -> pl.Expr:
    if isinstance(x, pl.Expr):
        return x
    if isinstance(x, str):
        return pl.col(x)
    if isinstance(x, (int, float, bool)):
        return pl.lit(x)
    raise TypeError(f"Expected str, scalar or pl.Expr, got {type(x).__name__}: {x!r}")


# ---------------------------------------------------------------------------
# Registration Bridge
# ---------------------------------------------------------------------------

def make_talib_fn(spec: talib_ops.TalibOpSpec) -> Callable[..., pl.Expr]:
    """将 TA-Lib 白名单算子包装为自适应参数补全、类型安全且按 sid 分区的 Polars 表达式函数。"""
    n_inputs = len(spec.inputs)
    n_params = len(spec.params)
    output_idx = spec.output_index if spec.output_index is not None else 0

    def _fn(*expr_args: Any) -> pl.Expr:
        n_args = len(expr_args)
        if n_args < n_inputs:
            raise ValueError(
                f"{spec.op}: expected at least {n_inputs} input field(s), got {n_args}"
            )

        # 1. 提取行情列与入参
        field_exprs = [_ensure_expr(a) for a in expr_args[:n_inputs]]
        passed_params = list(expr_args[n_inputs:])

        # 2. 补齐缺失的默认参数(防御式补齐，杜绝因 AST 未回填导致的崩溃)
        kwargs: Dict[str, Any] = {}
        for i, p_spec in enumerate(spec.params):
            if i < len(passed_params):
                val = passed_params[i]
            else:
                val = p_spec.default
                if val is None:
                    raise ValueError(f"{spec.op}: missing required parameter '{p_spec.name}'")

            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(
                    f"{spec.op}: param '{p_spec.name}' must be a number, got {val!r}"
                )

            # 严格规避 TA-Lib C 底层类型检查错误
            if isinstance(val, float) and val.is_integer():
                val = int(val)
            elif isinstance(val, float) and getattr(p_spec, "type", None) == "int":
                val = int(round(val))

            kwargs[p_spec.name] = val

        # 3. 动态获取底层 C 函数
        talib_module = talib_ops.talib
        if talib_module is None:
            raise RuntimeError("TA-Lib C extension is not installed")
        c_func = getattr(talib_module, spec.func.upper(), None)
        if c_func is None:
            raise ValueError(f"TA-Lib function '{spec.func.upper()}' not found")

        # 4. 向量化切片执行核
        def _apply_talib_c(s: pl.Series) -> pl.Series:
            chunk_len = len(s)
            if chunk_len == 0:
                return pl.Series(values=[], dtype=pl.Float64)

            numpy_args = [
                s.struct.field(f"_talib_in_{i}")
                .fill_null(float("nan"))
                .to_numpy()
                .astype(np.float64)
                for i in range(n_inputs)
            ]

            try:
                result = c_func(*numpy_args, **kwargs)
                if isinstance(result, tuple):
                    result = result[output_idx]
                if result is None:
                    result = np.full(chunk_len, np.nan, dtype=np.float64)
            except Exception as e:
                logger.warning(f"TA-Lib {spec.op} execution failed on chunk: {e}")
                result = np.full(chunk_len, np.nan, dtype=np.float64)

            return pl.Series(result, dtype=pl.Float64)

        struct_parts = [e.alias(f"_talib_in_{i}") for i, e in enumerate(field_exprs)]

        return (
            pl.struct(struct_parts)
            .map_batches(_apply_talib_c, return_dtype=pl.Float64)
            .over("sid")
        )

    return _fn


def register_talib_whitelist(target_ops: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
    """将白名单算子全量注入 SAFE_OPS 注册表。支持外部显式传入 target_ops 防止模块循环导入。"""
    if talib_ops.talib is None:
        logger.info("[safeops] talib not installed, skipping registration")
        return {}

    if target_ops is None:
        from bt_studio.compiler.ops import SAFE_OPS
        target_ops = SAFE_OPS

    registered: Dict[str, Dict[str, Any]] = {}

    for op, spec in talib_ops.REGISTRY.items():
        n_inputs = len(spec.inputs)
        # 必填项 = 输入列数 + (无默认值且未冻结的参数数量)
        min_args = n_inputs + len([p for p in spec.params if p.fixed is None and p.default is None])
        max_args = n_inputs + len(spec.params)
        defaults = [p.default for p in spec.params if p.default is not None]

        entry = {
            "fn": make_talib_fn(spec),
            "arity": (min_args, max_args),
            "defaults": defaults,
            "family": "ts",
        }
        target_ops[op] = entry
        registered[op] = entry

    if registered:
        logger.debug(f"[safeops] talib registered {len(registered)} ops")

    return registered


# ---------------------------------------------------------------------------
# Static validation hook (AST Static Test)
# ---------------------------------------------------------------------------

def talib_validate_hook(ast_node: Any) -> Optional[str]:
    """主编译器静态语法与因果律校验钩子 (拒绝通用 'talib' 旁路，严格校验白名单算子)。"""
    if not isinstance(ast_node, dict) or "op" not in ast_node:
        return None

    op_name = ast_node.get("op")
    if op_name == "talib":
        return "禁止使用通用 'talib' 桥接算子绕过白名单，请直接调用白名单算子名 (如 rsi, macd, adx)"
    if op_name not in talib_ops.REGISTRY:
        return None

    errors = talib_ops.validate_node(ast_node)
    return "; ".join(errors) if errors else None