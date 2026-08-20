#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bt_studio.compiler.ast import compile_ast, ASTCompilationError
from bt_studio.compiler.ops import SAFE_OPS


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
        ("talib(2,None)+kwargs",
         {"op": "talib", "args": ["RSI", {"col": "close"}],
          "kwargs": {"timeperiod": 14}}),
    ]
    for label, ast in cases:
        expr, name = compile_ast(ast, check_causal=False)
        assert expr is not None
        print(f"[OK] compile {label} -> {name}")


def test_error_arity():
    cases = [
        ({"op": "abs",   "args": []},                        "exactly 1"),
        ({"op": "ref",   "args": [{"col": "c"}, 5, 6]},      "1 to 2"),
        ({"op": "talib", "args": ["RSI"]},                   "at least 2"),
    ]
    for ast, expect_msg in cases:
        try:
            compile_ast(ast, check_causal=False)
            raise AssertionError(f"should have raised: {ast}")
        except ASTCompilationError as e:
            assert expect_msg in str(e), f"bad message: {e}"
            print(f"[OK] rejected {ast['op']} ({expect_msg})")


if __name__ == "__main__":
    test_registry_is_range_tuples()
    test_normal_compiles()
    test_error_arity()
    print("\nALL ARITY REFACTOR VERIFIED")