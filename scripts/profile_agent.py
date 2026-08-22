#!/usr/bin/env python3

import cProfile
import io
import os
import pstats
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bt_studio.constant import LLM_PROFILE_DIR


def main() -> int:
    # Import the real entry lazily so profiling covers the whole run.
    sys.argv = [sys.argv[0]]
    import runpy
    import run_agent  # noqa: F401  (ensures path setup)

    target = "/Users/hengxinliu/startup/bt_studio/scripts/run_agent.py"

    prof = cProfile.Profile()
    try:
        prof.enable()
        runpy.run_path(target, run_name="__main__")
    except SystemExit:
        pass  # script may call sys.exit
    finally:
        prof.disable()

    os.makedirs(LLM_PROFILE_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(LLM_PROFILE_DIR, f"profile_{ts}.txt")

    stream = io.StringIO()
    stats = pstats.Stats(prof, stream=stream)
    stats.sort_stats("cumulative").print_stats(30)
    # Also dump raw binary stats for interactive exploration
    stats.dump_stats(out.replace(".txt", ".pstats"))

    with open(out, "w", encoding="utf-8") as f:
        f.write(stream.getvalue())
    print(f"\n[profile] top-30 cumulative saved: {out}")
    print(f"[profile] raw pstats: {out.replace('.txt', '.pstats')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())