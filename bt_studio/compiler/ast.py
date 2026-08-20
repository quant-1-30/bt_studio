from __future__ import annotations

import hashlib
import json

import polars as pl

from typing import Any, Dict, Tuple, List

from .ops import SAFE_OPS, _to_expr
from .plugins.talib_hook import talib_validate_hook

from bt_studio.constant import MAX_AST_DEPTH

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class ASTCompilationError(Exception):
    """Raised when a JSON AST cannot be compiled to a valid Polars expression."""


def ast_fingerprint(ast_node: Any) -> str:
    """Return a short SHA-256 fingerprint (first 12 chars) for an AST.
    Useful for deduplication when the Agent proposes many candidate features.
    """
    canonical_node = canonicalize_ast(ast_node)
    canonical_json = json.dumps(canonical_node, sort_keys=True, default=str)
    return hashlib.sha256(canonical_json.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# compiler helper
# ---------------------------------------------------------------------------

def canonicalize_ast(ast_node: Any) -> Any:
    """
    AST Canonicalize:
    1. SAFE_OPS autofill missing
    2. ensure column fingerprint 100%
    """
    if not isinstance(ast_node, dict):
        return ast_node

    if "col" in ast_node:
        return {"col": str(ast_node["col"])}

    if "op" in ast_node:
        op_name = ast_node["op"]
        if op_name not in SAFE_OPS:
            return ast_node  

        spec = SAFE_OPS[op_name]
        min_arity, max_arity = spec["arity"]
        defaults = spec.get("defaults", [])

        # 递归规范化子节点
        raw_args = ast_node.get("args") or []
        normalized_args = [canonicalize_ast(arg) for arg in raw_args]

        # 自动补齐缺失的默认参数  delta 传1个参数 缺少 1 个从 defaults取补上
        missing_count = (max_arity or min_arity) - len(normalized_args)
        if missing_count > 0 and defaults:
            # 取 defaults 尾部的 missing_count 个默认值
            normalized_args.extend(defaults[-missing_count:])

        new_node = {"op": op_name, "args": normalized_args}
        if "kwargs" in ast_node and ast_node["kwargs"]:
            new_node["kwargs"] = ast_node["kwargs"]
        return new_node

    return ast_node


def _generate_name(ast_node: Any) -> str:
    """Generate a deterministic, human-readable column name from an AST node.

    Examples
    --------
    >>> _generate_name({"col": "close"})
    'close'
    >>> _generate_name({"op": "mean", "args": [{"col": "close"}, 10]})
    'mean(close,10)'
    >>> _generate_name({"op": "add", "args": [{"col": "ofi"}, {"col": "vol"}]})
    'add(ofi,vol)'
    """
    if isinstance(ast_node, dict) and "col" in ast_node:
        return str(ast_node["col"])

    if isinstance(ast_node, dict) and "op" in ast_node:
        op = ast_node["op"]
        args = ast_node.get("args") or []
        kwargs = ast_node.get("kwargs") or {}
        
        parts = [_generate_name(a) for a in args]
        
        for k in sorted(kwargs.keys()):
            parts.append(f"{k}={kwargs[k]}")
            
        return f"{op}({','.join(parts)})"

    return str(ast_node)


def calc_ast_depth(ast_node: Any) -> int:
    """Calculate the maximum depth (height) of the AST.
    
    Leaf nodes (columns, literals) have depth 0.
    Operator nodes have depth 1 + max(depth(args)).
    
    Used to prevent overfitting (typically depth > 4 is considered overfitted).
    """
    if not isinstance(ast_node, dict):
        return 0
        
    if "col" in ast_node:
        return 0
        
    if "op" in ast_node:
        args = ast_node.get("args", [])
        if not args:
            return 1
        return 1 + max(calc_ast_depth(arg) for arg in args)
        
    return 0


def _expand_and_calc_depth(ast_node: Any, symbol_depths: Dict[str, int]) -> int:
    """Let-binding calculate accurate depth dfs"""
    if not isinstance(ast_node, dict):
        return 0
    if "col" in ast_node:
        col_name = str(ast_node["col"])
        return symbol_depths.get(col_name, 0)
    if "op" in ast_node:
        args = ast_node.get("args") or []
        if not args:
            return 1
        return 1 + max(_expand_and_calc_depth(arg, symbol_depths) for arg in args)
    return 0


def _compile_node(
    ast_node: Any,
    depth: int,
    cache: Dict[str, Tuple[pl.Expr, str]],
    depth_limit: int,
) -> Tuple[pl.Expr, str]:
    """Recursively compile a single AST node.

    Parameters
    ----------
    ast_node
        JSON-like dict / literal.
    depth
        Current recursion depth (for overflow protection).
    cache
        Memoisation dict keyed by ``json.dumps(ast_node)``.
    depth_limit
        Hard recursion cap for this compile call (thread-safe: passed by
        value, never mutated globally).

    Returns
    -------
    (pl.Expr, output_name)
    """
    if depth > depth_limit:
        raise ASTCompilationError(
            f"AST exceeds maximum depth ({depth_limit}). "
            "The expression tree is too deeply nested or possibly cyclic."
        )

    # --- Cache lookup -------------------------------------------------------
    cache_key = json.dumps(ast_node, sort_keys=True, default=str)
    if cache_key in cache:
        return cache[cache_key]

    # --- Leaf: literal ------------------------------------------------------
    if isinstance(ast_node, (int, float)):
        expr = pl.lit(ast_node)
        cache[cache_key] = (expr, str(ast_node))
        return expr, str(ast_node)

    # --- Leaf: column reference --------------------------------------------
    if isinstance(ast_node, dict) and "col" in ast_node:
        col_name = str(ast_node["col"])
        expr = pl.col(col_name)
        cache[cache_key] = (expr, col_name)
        return expr, col_name

    # --- Op node ------------------------------------------------------------
    if isinstance(ast_node, dict) and "op" in ast_node:
        op_name = ast_node["op"]

        if op_name not in SAFE_OPS:
            raise ASTCompilationError(
                f"Unknown operator '{op_name}'. "
                f"Available: {sorted(SAFE_OPS.keys())}"
            )

        _talib_err = talib_validate_hook(ast_node)
        if _talib_err:
            raise ASTCompilationError(f"talib op rejected: {_talib_err}")

        spec = SAFE_OPS[op_name]
        arity_range = spec["arity"]          # (min, max) tuple; max=None => variadic
        fn = spec["fn"]

        raw_args = ast_node.get("args") or []
        raw_kwargs = ast_node.get("kwargs") or {}

        lo, hi = arity_range
        n_args = len(raw_args)
        if n_args < lo or (hi is not None and n_args > hi):
            if hi == lo:
                expect = f"exactly {lo}"
            elif hi is None:
                expect = f"at least {lo}"
            else:
                expect = f"{lo} to {hi}"
            raise ASTCompilationError(
                f"Operator '{op_name}' expects {expect} argument(s), "
                f"got {n_args}."
            )

        compiled_args: list[Any] = []
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

    raise ASTCompilationError(f"Invalid AST node: {ast_node!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compile_ast(
    ast_node: Any,
    check_causal: bool = True,
    max_depth: int | None = 4,
) -> Tuple[pl.Expr, str]:
    """Compile a JSON AST node into a ``(pl.Expr, output_name)`` tuple.

    Parameters
    ----------
    ast_node
        The root AST node.  Supported shapes::

            {"col": "close"}                          # column reference
            {"op": "mean", "args": [<node>, <N>]}    # operator
            5                                          # literal

    check_causal, default True
        If ``True``, run :func:`causal_guard.check_causal` on the AST before
        compilation.  Any lookahead violation raises ``ASTCompilationError``.

    max_depth, optional
        Strict constraint on semantic tree depth (to prevent overfitting).
        Default is 4. When ``None``, the constant ``MAX_AST_DEPTH``
        (read-only fallback, also 4) applies. The limit is passed down as a
        parameter — the global constant is never mutated (thread-safe).

    Returns
    -------
    (pl.Expr, str)
        The compiled Polars expression and a deterministic output column name.

    Raises
    ------
    ASTCompilationError
        If the AST is malformed, uses unknown operators, exceeds the depth
        limit, or contains lookahead patterns (when ``check_causal=True``).
    """
    ast_node = canonicalize_ast(ast_node)

    effective_limit = max_depth if max_depth is not None else MAX_AST_DEPTH

    # --- Complexity Guard ---
    actual_depth = calc_ast_depth(ast_node)
    if actual_depth > effective_limit:
        raise ASTCompilationError(
            f"AST depth {actual_depth} exceeds allowed limit ({effective_limit}). "
            "Features this complex are prone to overfitting."
        )

    # --- Causal guard ---
    if check_causal:
        violations = check_causal_violations(ast_node)
        if violations:
            raise ASTCompilationError(
                f"AST failed causal check (lookahead detected):\n"
                + "\n".join(f"  - {v}" for v in violations)
            )

    # --- Compile --------------------------------------------------------------
    cache: Dict[str, Tuple[pl.Expr, str]] = {}
    return _compile_node(ast_node, depth=0, cache=cache, depth_limit=effective_limit)


def compile_recipe(
    recipe: List[Dict[str, Any]],
    check_causal: bool = True,
    max_depth: int | None = 4,
) -> List[pl.Expr]:
    """Compile a multi-step recipe (Let-binding) into a list of Polars expressions.
    
    A recipe is a list of steps, where each step assigns a name to an AST.
    Downstream steps can reference previous step names via `{"col": "step_name"}`.
    
    Example
    -------
    [
        {"name": "temp_dir", "ast": {"op": "sign", "args": [{"op": "delta", "args": [{"col": "close"}, 1]}]}}},
        {"name": "final_feat", "ast": {"op": "mul", "args": [{"col": "temp_dir"}, {"col": "amount"}]}}}
    ]
    Fix:
        Multi Steps Recipe
    
        1. depth verify 
        2. Expr chain with columns
    """
    
    if not isinstance(recipe, list):
        raise ASTCompilationError("Recipe must be a list of step dicts.")

    # canonize Recipe step AST
    canonical_recipe = []
    for step in recipe:
        if isinstance(step, dict) and "ast" in step:
            canonical_recipe.append({
                **step,
                "ast": canonicalize_ast(step["ast"]) # overwrite
            })
        else:
            canonical_recipe.append(step)

    effective_limit = max_depth if max_depth is not None else MAX_AST_DEPTH
    symbol_depths: Dict[str, int] = {}
    compiled_exprs: List[pl.Expr] = []

    for i, step in enumerate(canonical_recipe):
        
        if not isinstance(step, dict) or "name" not in step or "ast" not in step:
            raise ASTCompilationError(f"Recipe step {i} malformed. Must contain 'name' and 'ast'.")

        name = step["name"]
        ast = step["ast"]

        # 1. cumulate true depth 
        cumulative_depth = _expand_and_calc_depth(ast, symbol_depths)
        if cumulative_depth > effective_limit:
            raise ASTCompilationError(
                f"Recipe step '{name}' cumulative depth {cumulative_depth} exceeds "
                f"allowed limit ({effective_limit}). Complex chained features are prone to overfitting."
            )
        symbol_depths[name] = cumulative_depth

        # 2. no limit max_depth
        expr, _ = compile_ast(ast, check_causal=check_causal, max_depth=None)
        compiled_exprs.append(expr.alias(name))

    return compiled_exprs


def compile_to_alias(
    ast_node: Any,
    check_causal: bool = True,
    max_depth: int | None = 4,
) -> pl.Expr:
    """Like :func:`compile_ast` but returns the expression already aliased.

    Equivalent to::

        expr, name = compile_ast(ast_node, ...)
        return expr.alias(name)
    """
    expr, name = compile_ast(ast_node, check_causal=check_causal, max_depth=max_depth)
    return expr.alias(name)

import re
from typing import Any, List, Tuple

BANNED_STR_PATTERNS = [
    re.compile(r"shift\s*\(\s*-"),                     # shift(-1), shift( -5 )
    re.compile(r"center\s*=\s*True", re.IGNORECASE),   # center=True, center = true
    re.compile(r"\b(backward_fill|bfill)\b"),          # backward_fill, bfill \b means complete 
    re.compile(r"strategy\s*=\s*['\"]backward['\"]"),  # fill_null(strategy='backward')
    re.compile(r"\b(lead|future_return|next_)\b"),     # 常见的未来特征列名/算子
]

LOOKAHEAD_OPS = {"ref", "shift", "ts_delay", "lag", "delta"}


def check_causal_violations(ast_node: Any, path: str = "root") -> List[str]:
    """
        AST Node or Expr future func 
    return: violations: list[str]
    """
    violations: List[str] = []

    # 1. recursive 
    if isinstance(ast_node, list):
        for i, item in enumerate(ast_node):
            violations.extend(check_causal_violations(item, f"{path}[{i}]"))
        return violations

    # 2. AST Node
    if isinstance(ast_node, dict):
        op_name = ast_node.get("op", "")
        args = ast_node.get("args") or []
        kwargs = ast_node.get("kwargs") or {}

        # ref(x, -1), shift(x, -1), delta(x, -5)
        if op_name in LOOKAHEAD_OPS and len(args) >= 2:
            period_arg = args[1]
            if isinstance(period_arg, (int, float)) and period_arg < 0:
                violations.append(
                    f"{path}.{op_name}: negative args ({period_arg}) Lookahead"
                )

        # kwargs center=True, strategy='backward'
        if kwargs.get("center") is True:
            violations.append(f"{path}.{op_name}: center=True")
        
        if kwargs.get("strategy") in ("backward", "bfill"):
            violations.append(f"{path}.{op_name}: 启用了后向填充 (strategy={kwargs['strategy']!r})")

        # col name 
        if "col" in ast_node:
            col_name = str(ast_node["col"])
            for pat in BANNED_STR_PATTERNS:
                if pat.search(col_name):
                    violations.append(f"{path}.col[{col_name}]: illegal pattern")

        # recursive args
        for i, arg in enumerate(args):
            violations.extend(check_causal_violations(arg, f"{path}.{op_name}.args[{i}]"))

        # recursive kwargs
        for k, v in kwargs.items():
            violations.extend(check_causal_violations(v, f"{path}.{op_name}.kwargs[{k}]"))

    elif isinstance(ast_node, str):
        for pat in BANNED_STR_PATTERNS:
            if pat.search(ast_node):
                violations.append(f"{path}: loopahead pattern {pat.pattern!r}")

    return violations


def check_causal(ast_node: Any, path: str = "root") -> List[str]:
    return check_causal_violations(ast_node, path)

def is_causal(ast_node: Any) -> bool:
    return len(check_causal_violations(ast_node)) == 0
