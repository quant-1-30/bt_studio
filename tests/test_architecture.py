#!/usr/bin/env python3

import ast
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BT_STUDIO = os.path.join(PROJECT_ROOT, "bt_studio")

CORE_PREFIXES = ("pipeline", "tune", "orchestrator", "visual", "utils", "constant", "risk")


def _iter_py_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in
                       ("__pycache__", ".git", "_libs")]
        for fn in filenames:
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


def _module_imports(path):
    """Return the set of absolute module names imported by a file."""
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                mods.add(node.module)
    return mods


def _rel(path):
    return os.path.relpath(path, PROJECT_ROOT)


def main() -> int:
    failures = []

    # ---- Rule 1: core must not import plugins ----------------------------
    for path in _iter_py_files(BT_STUDIO):
        rel = _rel(path)
        parts = rel.split(os.sep)
        if len(parts) < 2 or parts[1] not in CORE_PREFIXES:
            continue
        for mod in _module_imports(path):
            if mod == "bt_studio.plugins" or mod.startswith("bt_studio.plugins."):
                failures.append(f"[core->plugins] {rel}: imports {mod}")

    # ---- Rule 2: api_server must not import xtp_client -------------------
    api_dir = os.path.join(BT_STUDIO, "plugins", "api_server")
    for path in _iter_py_files(api_dir):
        rel = _rel(path)
        for mod in _module_imports(path):
            if "xtp_client" in mod:
                failures.append(f"[api_server->xtp] {rel}: imports {mod}")

    # ---- Rule 3: xtp_client must not import bt_studio --------------------
    xtp_dir = os.path.join(BT_STUDIO, "plugins", "xtp_client")
    for path in _iter_py_files(xtp_dir):
        rel = _rel(path)
        for mod in _module_imports(path):
            if mod == "bt_studio" or mod.startswith("bt_studio."):
                failures.append(f"[xtp->core] {rel}: imports {mod}")

    if failures:
        print("ARCHITECTURE VIOLATIONS:")
        for f_ in failures:
            print(f"  - {f_}")
        return 1
    print("[test_architecture] layering OK: core ⊥ plugins, api_server ⊥ xtp, xtp ⊥ core")
    return 0


if __name__ == "__main__":
    sys.exit(main())