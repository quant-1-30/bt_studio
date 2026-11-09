from __future__ import annotations

import copy
import hashlib
import inspect
import json
import re
import polars as pl

from typing import Any, Dict, List, Optional, Set, Tuple


from bt_studio.constant import BASE_MARKET_COLS, MAX_AST_DEPTH, MAX_RECIPE_DEPTH
from bt_studio.utils.months import month_id_expr
from .ops import SAFE_OPS
from .plugins.talib_hook import talib_validate_hook


# ---------------------------------------------------------------------------
# Exceptions & Constants
# ---------------------------------------------------------------------------

class ASTCompilationError(Exception):
    """Raised when a JSON AST cannot be compiled to a valid Polars expression."""

# lookahead operators that may introduce future information leakage
BANNED_STR_PATTERNS = [
    re.compile(r"shift\s*\(\s*-"),
    re.compile(r"center\s*=\s*True", re.IGNORECASE),
    re.compile(r"\b(backward_fill|bfill)\b", re.IGNORECASE),
    re.compile(r"strategy\s*=\s*['\"]backward['\"]", re.IGNORECASE),
    re.compile(r"\b(lead\b|future_return|next_)"),
]

LOOKAHEAD_OPS = frozenset({"ref", "shift", "ts_delay", "lag", "delta", "roc"})
CS_WINDOW_OPS = frozenset({"cs_rank", "cs_demean", "cs_zscore", "mdp_weight"})
TS_WINDOW_OPS = frozenset({
    "ref", "delta", "mean", "std", "var", "max", "min", "sum", "ema", "rank",
    "zscore", "corr", "cov", "cum_sum", "cum_max", "cum_min", "roc", "atr",
    "stoch_k",
})


def _ts_window_ops() -> Set[str]:
    """Return the set of all time-series window operators (static + registered SAFE_OPS)."""
    return set(TS_WINDOW_OPS) | {op for op, spec in SAFE_OPS.items() if spec.get("family") == "ts"}


# Window argument constraints: op_name -> (arg_index, min_window)
_WINDOW_CONSTRAINTS: Dict[str, Tuple[int, int]] = {
    "mean": (1, 2), "std": (1, 2), "var": (1, 2), "max": (1, 2),
    "min": (1, 2), "sum": (1, 2), "rank": (1, 2), "zscore": (1, 2),
    "ema": (1, 1), "ref": (1, 1), "delta": (1, 1), "roc": (1, 1),
    "corr": (2, 3), "cov": (2, 3),
}


def _is_ts_window_op(op_name: str) -> bool:
    """Dynamically check if op_name is a time-series window operator."""
    return op_name in _ts_window_ops()


# ---------------------------------------------------------------------------
# AST Validation & Helpers
# ---------------------------------------------------------------------------

def _arity_error(op_name: str, lo: int, hi: Optional[int], n_args: int) -> ASTCompilationError:
    if hi == lo:
        expect = f"exactly {lo}"
    elif hi is None:
        expect = f"at least {lo}"
    else:
        expect = f"{lo} to {hi}"
    return ASTCompilationError(
        f"Operator '{op_name}' expects {expect} argument(s), got {n_args}."
    )


def ast_fingerprint(ast_node: Any) -> str:
    """AST 12 SHA-256 """
    try:
        canonical_node = canonicalize_ast(ast_node)
    except ASTCompilationError:
        canonical_node = ast_node
    canonical_json = json.dumps(canonical_node, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:12]


def canonicalize_ast(ast_node: Any) -> Any:
    """Recursively canonicalize an AST dict or Recipe list (backfill defaults, normalize integer floats, strip col names)."""
    if isinstance(ast_node, list):
        return [
            {"name": str(step["name"]).strip(), "ast": canonicalize_ast(step["ast"])}
            if isinstance(step, dict) and "name" in step and "ast" in step
            else canonicalize_ast(step)
            for step in ast_node
        ]
    if not isinstance(ast_node, dict):
        return ast_node
    if "ast" in ast_node and "op" not in ast_node and "col" not in ast_node:
        inner = ast_node["ast"]
        if isinstance(inner, list):
            return canonicalize_ast(inner)
        if isinstance(inner, dict) and "name" in ast_node:
            return [{"name": str(ast_node["name"]).strip(), "ast": canonicalize_ast(inner)}]
        return canonicalize_ast(inner)

    if "col" in ast_node:
        return {"col": str(ast_node["col"]).strip()}

    if "op" in ast_node:
        op_name = ast_node["op"]
        if op_name not in SAFE_OPS:
            return ast_node

        spec = SAFE_OPS[op_name]
        min_arity, max_arity = spec["arity"]
        defaults = spec.get("defaults", [])

        # float int ---> int avoid Polars window bug
        raw_args = ast_node.get("args") or []
        normalized_args = [
            int(arg) if isinstance(arg, float) and arg.is_integer()
            else canonicalize_ast(arg)
            for arg in raw_args
        ]

        raw_kwargs = copy.deepcopy(ast_node.get("kwargs") or {})
        if raw_kwargs and max_arity is not None and len(normalized_args) < max_arity:
            param_names = list(inspect.signature(spec["fn"]).parameters.keys())
            while len(normalized_args) < len(param_names) and param_names[len(normalized_args)] in raw_kwargs:
                kw_val = raw_kwargs.pop(param_names[len(normalized_args)])
                normalized_args.append(
                    int(kw_val) if isinstance(kw_val, float) and kw_val.is_integer() else canonicalize_ast(kw_val)
                )

        n_args = len(normalized_args)
        if n_args < min_arity or (max_arity is not None and n_args > max_arity):
            raise _arity_error(op_name, min_arity, max_arity, n_args)

        missing_count = (max_arity or min_arity) - n_args
        if missing_count > len(defaults):
            raise ASTCompilationError(
                f"Operator '{op_name}' registry error: defaults {defaults} "
                f"cannot cover {missing_count} missing argument(s)"
            )
        if missing_count > 0:
            normalized_args.extend(defaults[-missing_count:])

        new_node = {"op": op_name, "args": normalized_args}
        if raw_kwargs:
            new_node["kwargs"] = raw_kwargs
        return new_node

    return ast_node


def _generate_name(ast_node: Any) -> str:
    if isinstance(ast_node, dict):
        if "col" in ast_node:
            return str(ast_node["col"])
        if "op" in ast_node:
            op = ast_node["op"]
            args = ast_node.get("args") or []
            kwargs = ast_node.get("kwargs") or {}
            parts = [_generate_name(a) for a in args]
            for k in sorted(kwargs.keys()):
                parts.append(f"{k}={kwargs[k]}")
            return f"{op}({','.join(parts)})"
    return str(ast_node)


def calc_ast_depth(ast_node: Any) -> int:
    """AST Tree depth and leap node 0 / 1 + max subleaf depath"""
    if not isinstance(ast_node, dict) or "op" not in ast_node:
        return 0
    args = ast_node.get("args") or []
    if not args:
        return 1
    return 1 + max(calc_ast_depth(arg) for arg in args)


def _expand_and_calc_depth(ast_node: Any, symbol_depths: Dict[str, int], memo: Optional[Dict[int, int]] = None) -> int:
    """Let-binding"""
    if memo is None:
        memo = {}
    node_id = id(ast_node)
    if node_id in memo:
        return memo[node_id]

    if not isinstance(ast_node, dict):
        res = 0
    elif "col" in ast_node:
        res = symbol_depths.get(str(ast_node["col"]), 0)
    elif "op" in ast_node:
        args = ast_node.get("args") or []
        if not args:
            res = 1
        else:
            res = 1 + max(_expand_and_calc_depth(arg, symbol_depths, memo) for arg in args)
    else:
        res = 0

    memo[node_id] = res
    return res


# ---------------------------------------------------------------------------
# Safety Guards
# ---------------------------------------------------------------------------

def _window_arg_violations(ast_node: Any, path: str = "root") -> List[str]:
    violations: List[str] = []
    if isinstance(ast_node, list):
        for i, item in enumerate(ast_node):
            violations.extend(_window_arg_violations(item, f"{path}[{i}]"))
        return violations
    if not isinstance(ast_node, dict) or "op" not in ast_node:
        return violations

    op_name = ast_node["op"]
    args = ast_node.get("args") or []

    if op_name in _WINDOW_CONSTRAINTS:
        arg_idx, min_w = _WINDOW_CONSTRAINTS[op_name]
        if len(args) > arg_idx:
            w = args[arg_idx]
            if isinstance(w, (int, float)) and not isinstance(w, bool):
                if not float(w).is_integer():
                    violations.append(f"{path}.{op_name}: window {w!r} 必须为整数")
                elif int(w) < min_w:
                    violations.append(f"{path}.{op_name}: window {int(w)} < 最小合法值 {min_w}")

    for i, arg in enumerate(args):
        violations.extend(_window_arg_violations(arg, f"{path}.{op_name}.args[{i}]"))
    return violations


def _family_nesting_violation(ast_node: Any, enclosing: Optional[str] = None, path: str = "root") -> Optional[str]:
    """TS and CS vertical nested """
    if not isinstance(ast_node, dict) or "op" not in ast_node:
        return None

    op_name = ast_node["op"]
    node_family = None
    if op_name in CS_WINDOW_OPS:
        node_family = "cs"
    elif _is_ts_window_op(op_name):
        node_family = "ts"

    if enclosing == "ts" and node_family == "cs":
        return (
            f"{path}.{op_name}: 截面算子跨分区嵌套在时序算子内部 "
            f"(分区键 over('sid') vs over(['day','bar_idx'])) Polars 将静默输出全 null。"
            f"请拆为 recipe 链式分步物化。"
        )
    if enclosing == "cs" and node_family == "ts":
        return (
            f"{path}.{op_name}: 时序算子跨分区嵌套在截面算子内部 "
            f"(分区键 over(['day','bar_idx']) vs over('sid')) polars 将静默输出全 null。"
            f"请拆为 recipe 链式分步物化。"
        )

    eff = node_family or enclosing
    for i, arg in enumerate(ast_node.get("args") or []):
        err = _family_nesting_violation(arg, eff, f"{path}.{op_name}.args[{i}]")
        if err is not None:
            return err
    return None


def check_causal_violations(ast_node: Any, path: str = "root") -> List[str]:
    violations: List[str] = []

    if isinstance(ast_node, list):
        for i, item in enumerate(ast_node):
            violations.extend(check_causal_violations(item, f"{path}[{i}]"))
        return violations

    if isinstance(ast_node, dict):
        op_name = ast_node.get("op", "")
        args = ast_node.get("args") or []
        kwargs = ast_node.get("kwargs") or {}

        # 1. negative shift 
        if op_name in LOOKAHEAD_OPS:
            if len(args) >= 2:
                period_arg = args[1]
                if isinstance(period_arg, (int, float)) and not isinstance(period_arg, bool) and period_arg < 0:
                    violations.append(f"{path}.{op_name}: 位置参数包含负向步长 ({period_arg})，存在未来函数")

            for kw in ("period", "periods", "step", "n", "lag", "shift"):
                if kw in kwargs:
                    v = kwargs[kw]
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and v < 0:
                        violations.append(f"{path}.{op_name}: 关键字参数 {kw}={v} 包含负向步长，存在未来函数")

        # 2. forbiden bfill / backward_fill / lead
        if op_name:
            for pat in BANNED_STR_PATTERNS:
                if pat.search(str(op_name)):
                    violations.append(f"{path}.op[{op_name}]: 使用包含前瞻风险的算子 {pat.pattern!r}")

        # 3. exzamin future and backfill strategy
        center_val = kwargs.get("center")
        if center_val is True or str(center_val).strip().lower() == "true":
            violations.append(f"{path}.{op_name}: 禁止启用未来中心滚动窗口 center=True")

        strategy_val = str(kwargs.get("strategy", "")).strip().lower()
        if strategy_val in ("backward", "bfill"):
            violations.append(f"{path}.{op_name}: 启用了未来数据填充 (strategy={strategy_val!r})")

        # 4. sensetive prompt
        if "col" in ast_node:
            col_name = str(ast_node["col"])
            for pat in BANNED_STR_PATTERNS:
                if pat.search(col_name):
                    violations.append(f"{path}.col[{col_name}]: 引用包含前瞻风险的保留特征列 {pat.pattern!r}")

        # 5. recursively leap node
        for i, arg in enumerate(args):
            violations.extend(check_causal_violations(arg, f"{path}.{op_name}.args[{i}]"))
        for k, v in kwargs.items():
            violations.extend(check_causal_violations(v, f"{path}.{op_name}.kwargs[{k}]"))

    return violations


def check_causal(ast_node: Any, path: str = "root") -> List[str]:
    return check_causal_violations(ast_node, path)


# ---------------------------------------------------------------------------
# Compilation Engine
# ---------------------------------------------------------------------------

def _compile_node(
    ast_node: Any,
    depth: int,
    cache: Dict[str, Tuple[pl.Expr, str]],
    depth_limit: Optional[int],
) -> Tuple[pl.Expr, str]:
    """compile AST node to Polars expr"""
    if depth_limit is not None and depth > depth_limit:
        raise ASTCompilationError(
            f"AST exceeds maximum allowed depth ({depth_limit}). "
            "Expression tree is too deeply nested or cyclical."
        )

    # compact key
    cache_key = json.dumps(ast_node, sort_keys=True, default=str, separators=(",", ":"))
    if cache_key in cache:
        return cache[cache_key]

    # pl.plt
    if isinstance(ast_node, (int, float)) and not isinstance(ast_node, bool):
        expr = pl.lit(ast_node)
        cache[cache_key] = (expr, str(ast_node))
        return expr, str(ast_node)

    if isinstance(ast_node, dict) and "col" in ast_node:
        col_name = str(ast_node["col"])
        expr = pl.col(col_name)
        cache[cache_key] = (expr, col_name)
        return expr, col_name

    if isinstance(ast_node, dict) and "op" in ast_node:
        op_name = ast_node["op"]
        if op_name not in SAFE_OPS:
            raise ASTCompilationError(
                f"Unknown operator '{op_name}'. Available: {sorted(SAFE_OPS.keys())}"
            )

        talib_err = talib_validate_hook(ast_node)
        if talib_err:
            raise ASTCompilationError(f"TA-Lib op rejected: {talib_err}")

        fn = SAFE_OPS[op_name]["fn"]
        raw_args = ast_node.get("args") or []
        raw_kwargs = ast_node.get("kwargs") or {}

        compiled_args: List[Any] = []
        for arg in raw_args:
            if isinstance(arg, dict):
                child_expr, _ = _compile_node(arg, depth + 1, cache, depth_limit)
                compiled_args.append(child_expr)
            else:
                compiled_args.append(arg)

        result_expr = fn(*compiled_args, **raw_kwargs)
        output_name = _generate_name(ast_node)
        cache[cache_key] = (result_expr, output_name)
        return result_expr, output_name

    raise ASTCompilationError(f"Invalid AST node structure: {ast_node!r}")


def _compile_ast_internal(
    ast_node: Any,
    check_causal: bool = True,
    effective_limit: Optional[int] = 4,
) -> Tuple[pl.Expr, str]:
    """inner compiler core and skip external api"""
    if effective_limit is not None:
        actual_depth = calc_ast_depth(ast_node)
        if actual_depth > effective_limit:
            raise ASTCompilationError(
                f"AST depth {actual_depth} exceeds allowed limit ({effective_limit}). "
                "Overfitting prevention triggered."
            )

    if check_causal:
        violations = check_causal_violations(ast_node)
        if violations:
            raise ASTCompilationError(
                "AST failed causal verification (lookahead detected):\n"
                + "\n".join(f"  - {v}" for v in violations)
            )

    window_violations = _window_arg_violations(ast_node)
    if window_violations:
        raise ASTCompilationError(
            "AST failed window-argument validation:\n"
            + "\n".join(f"  - {v}" for v in window_violations)
        )

    family_violation = _family_nesting_violation(ast_node)
    if family_violation is not None:
        raise ASTCompilationError(f"AST failed window-family check:\n  - {family_violation}")

    cache: Dict[str, Tuple[pl.Expr, str]] = {}
    return _compile_node(ast_node, depth=0, cache=cache, depth_limit=effective_limit)


# ---------------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------------

def compile_ast(
    ast_node: Any,
    check_causal: bool = True,
    max_depth: Optional[int] = 4,
) -> Tuple[pl.Expr, str]:
    """JSON AST compiler to pl.Expr, col_name"""
    canonical_node = canonicalize_ast(ast_node)
    return _compile_ast_internal(canonical_node, check_causal=check_causal, effective_limit=max_depth)


def compile_recipe(
    recipe: List[Dict[str, Any]],
    check_causal: bool = True,
    max_depth: Optional[int] = 6,
) -> List[pl.Expr]:
    """multi Let-binding ---> Recipe and compiler to Polars expr
        for expr in compile_recipe(...):
            df = df.with_columns(expr)
    """
    if not isinstance(recipe, list):
        raise ASTCompilationError("Recipe must be a list of step dicts.")

    effective_limit = max_depth if max_depth is not None else MAX_RECIPE_DEPTH
    symbol_depths: Dict[str, int] = {}
    compiled_exprs: List[pl.Expr] = []
    declared_names: Set[str] = set()

    for i, step in enumerate(recipe):
        if not isinstance(step, dict) or "name" not in step or "ast" not in step:
            raise ASTCompilationError(f"Recipe step {i} malformed. Must contain 'name' and 'ast'.")

        name = str(step["name"]).strip()
        if not name:
            raise ASTCompilationError(f"Recipe step {i} has empty output name.")

        if name in BASE_MARKET_COLS:
            raise ASTCompilationError(
                f"Recipe step '{name}' cannot shadow base market column {name!r}."
            )
        if name in declared_names:
            raise ASTCompilationError(
                f"Recipe step '{name}' duplicates an existing variable name in the chain."
            )
        declared_names.add(name)

        canonical_step_ast = canonicalize_ast(step["ast"])

        cumulative_depth = _expand_and_calc_depth(canonical_step_ast, symbol_depths)
        if cumulative_depth > effective_limit:
            raise ASTCompilationError(
                f"Recipe step '{name}' cumulative depth {cumulative_depth} exceeds "
                f"allowed limit ({effective_limit})."
            )
        symbol_depths[name] = cumulative_depth

        expr, _ = _compile_ast_internal(canonical_step_ast, check_causal=check_causal, effective_limit=None)
        compiled_exprs.append(expr.alias(name))

    return compiled_exprs


def rewrite_ast_col_refs(node: Any, renames: dict) -> Any:
    """recursive rewriter AST {"col": name} reference"""
    if isinstance(node, dict):
        if "col" in node and isinstance(node["col"], str) and node["col"] in renames:
            node = dict(node)
            node["col"] = renames[node["col"]]
            return node
        return {k: rewrite_ast_col_refs(v, renames) for k, v in node.items()}
    if isinstance(node, list):
        return [rewrite_ast_col_refs(x, renames) for x in node]
    return node


def compile_feature_lazy(
    raw_hf_lf: pl.LazyFrame,
    feature_col: str,
    ast_recipe: Any,
) -> pl.LazyFrame:
    """AST Recipe and single-step AST compiler to LazyFrame and project base market columns, feature columns, and month_id."""
    combined_lf = raw_hf_lf
    if isinstance(ast_recipe, list):
        for e in compile_recipe(ast_recipe, check_causal=True):
            combined_lf = combined_lf.with_columns(e)
    elif ast_recipe is not None:
        expr, _ = compile_ast(ast_recipe, check_causal=True)
        combined_lf = combined_lf.with_columns(expr.alias(feature_col))
    elif feature_col not in combined_lf.collect_schema().names():
        raise ValueError(
            f"Feature {feature_col!r} not found in schema and ast_recipe is None"
        )

    schema_names = combined_lf.collect_schema().names()
    base_cols = [c for c in BASE_MARKET_COLS if c in schema_names]
    selected_cols = list(dict.fromkeys(base_cols + [feature_col]))
    date_col = "day" if "day" in schema_names else "date"

    return combined_lf.select(selected_cols).with_columns(
        month_id_expr(date_col).alias("month_id")
    )