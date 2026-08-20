#!/usr/bin/env python3

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
VENV_PY = sys.executable

# (module, needs_real_data)
TESTS = [
    ("tests/test_arity.py", False),       # compiler registry & arity ranges
    ("tests/test_architecture.py", False), # core-plugin layering guard
    ("tests/test_collapse.py", False),    # collapse verdicts + report roundtrip
    ("tests/test_agent_ast.py", False),   # AST compile + causal guard
    ("tests/test_dag.py", False),        # DAG parser/executor/intake
    ("tests/test_strategy.py", True),     # FSM/strategy (requires gRPC data)
]


def main() -> int:
    fast_only = len(sys.argv) > 1 and sys.argv[1] == "fast"
    failures = []

    for mod, needs_data in TESTS:
        if fast_only and needs_data:
            print(f"[skip] {mod} (needs real data)")
            continue
        print(f"\n===== RUN {mod} =====")
        proc = subprocess.run([VENV_PY, os.path.join(ROOT, mod)],
                              capture_output=True, text=True, cwd=ROOT)
        tail = "\n".join((proc.stdout or "").strip().splitlines()[-5:])
        print(tail if tail else "(no output)")
        if proc.returncode != 0:
            failures.append(mod)
            print(f"[FAIL] {mod} (rc={proc.returncode})")
            if proc.stderr:
                print(proc.stderr.strip()[-500:])
        else:
            print(f"[PASS] {mod}")

    print("\n" + "=" * 50)
    if failures:
        print(f"RESULT: {len(failures)} FAILED -> {', '.join(failures)}")
        return 1
    print("RESULT: ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())