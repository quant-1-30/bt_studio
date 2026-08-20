#!/usr/bin/env python3

import polars as pl
from bt_studio.compiler.ast import compile_ast, ast_fingerprint
from bt_studio.compiler.ops import list_ops


def run_tests():
    print("Running Agent AST Tests...")
    
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

    # Test 4: Cross-sectional + rolling combo
    ast4 = {'op': 'cs_zscore', 'args': [{'op': 'mean', 'args': [{'col': 'ofi_ratio'}, 10]}]}
    expr4, name4 = compile_ast(ast4, check_causal=True)
    print(f'Test4 cs+zscore: name={name4}')

    # Test 5: Fingerprint
    fp = ast_fingerprint(ast4)
    print(f'Test5 fingerprint: {fp}')

    # Test 6: Available ops
    ops = list_ops()
    print(f'Test6 ops ({len(ops)}): {ops}')

    print('\nALL TESTS PASSED')


if __name__ == "__main__":
    run_tests()
