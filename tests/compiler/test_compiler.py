#!/usr/bin/env python3

import os
import sys
import polars as pl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bt_studio.compiler.ast import compile_ast, ast_fingerprint, ASTCompilationError
from bt_studio.compiler.ops import list_ops, SAFE_OPS


def test_registry_is_range_tuples():
    """Every op must declare arity as (min, max) with max possibly None."""
    for op, spec in SAFE_OPS.items():
        a = spec["arity"]
        assert isinstance(a, tuple) and len(a) == 2, f"{op} arity not a 2-tuple: {a!r}"
        lo, hi = a
        assert isinstance(lo, int) and lo >= 1, f"{op} invalid min: {lo}"
        assert hi is None or (isinstance(hi, int) and hi >= lo), f"{op} invalid max: {hi}"
    print(f"[OK] registry: {len(SAFE_OPS)} ops all use (min,max) tuples")


def test_normal_compiles():
    cases = [
        ("abs(1,1)",   {"op": "abs",       "args": [{"col": "close"}]}),
        ("ref(2,2)",   {"op": "ref",       "args": [{"col": "close"}, 5]}),
        ("mdp(3,3)",   {"op": "mdp_weight","args": [{"col": "a"}, {"col": "b"}, {"col": "c"}]}),
        ("atr(4,4)",   {"op": "atr",       "args": [{"col": "h"}, {"col": "l"}, {"col": "c"}, 14]}),
        ("rsi(1,2)",   {"op": "rsi",       "args": [{"col": "close"}, 14]}),
    ]
    for label, ast in cases:
        expr, name = compile_ast(ast, check_causal=False)
        assert expr is not None
        print(f"[OK] compile {label} -> {name}")


def test_error_arity():
    cases = [
        ({"op": "abs",   "args": []},                        "exactly 1"),
        ({"op": "ref",   "args": [{"col": "c"}, 5, 6]},      "1 to 2"),
        ({"op": "atr",   "args": [{"col": "h"}]},            "3 to 4"),
    ]
    for ast, expect_msg in cases:
        try:
            compile_ast(ast, check_causal=False)
            raise AssertionError(f"should have raised: {ast}")
        except ASTCompilationError as e:
            assert expect_msg in str(e), f"bad message: {e}"
            print(f"[OK] rejected {ast['op']} ({expect_msg})")


def test_compiler_ast():
    # Test 1: Simple leaf
    expr, name = compile_ast({'col': 'close'}, check_causal=False)
    print(f'Test1 leaf: name={name}')

    # Test 2: Nested AST (this was the BUG before)
    ast = {'op': 'mean', 'args': [{'op': 'ref', 'args': [{'col': 'close'}, 5]}, 10]}
    expr, name = compile_ast(ast, check_causal=True)
    print(f'Test2 nested: name={name}')

    # Test 3: Causal guard rejection
    try:
        bad_ast = {'op': 'ref', 'args': [{'col': 'close'}, -3]}
        compile_ast(bad_ast, check_causal=True)
        print('Test3 FAILED: should have raised')
    except Exception as e:
        print(f'Test3 causal-guard OK: rejected lookahead')

    # Test 4: Cross-sectional + rolling 跨族嵌套 —— polars 1.43 对不同分区键的
    # 窗口嵌套会静默输出全 null, 编译期显式拒绝; 正规写法是 recipe 分步物化
    ast4 = {'op': 'cs_zscore', 'args': [{'op': 'mean', 'args': [{'col': 'ofi_ratio'}, 10]}]}
    try:
        compile_ast(ast4, check_causal=True)
        print('Test4 FAILED: should have raised (cross-family window nesting)')
        raise AssertionError('cross-family nesting should be rejected')
    except ASTCompilationError as e:
        assert 'window-family' in str(e)
        print(f'Test4 window-family guard OK: rejected cs_zscore(mean(...))')
    from bt_studio.compiler.ast import compile_recipe
    recipe4 = [
        {'name': 'ts_step', 'ast': {'op': 'mean', 'args': [{'col': 'ofi_ratio'}, 10]}},
        {'name': 'cs_step', 'ast': {'op': 'cs_zscore', 'args': [{'col': 'ts_step'}]}},
    ]
    exprs4 = compile_recipe(recipe4, check_causal=True)
    print(f'Test4b recipe cross-step OK: {len(exprs4)} steps')

    # Test 5: Fingerprint
    fp = ast_fingerprint(ast4)
    print(f'Test5 fingerprint: {fp}')

    # Test 6: Available ops
    ops = list_ops()
    print(f'Test6 ops ({len(ops)}): {ops}')
    print('\nALL TESTS PASSED')


if __name__ == "__main__":
    raise SystemExit("run via pytest")
