#!/usr/bin/env python3

import os
import platform
import subprocess
import sys


def main():
    app_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "bt_studio", "visual", "_streamlit.py",
    )
    if not os.path.exists(app_path):
        print(f"Error: app.py not found at {app_path}")
        sys.exit(1)

    bind_host = os.environ.get("BT_STUDIO_BIND_HOST", "127.0.0.1")
    port = os.environ.get("STREAMLIT_PORT", "8501")
    is_linux = platform.system() == "Linux"
    headless = os.environ.get("STREAMLIT_HEADLESS", "true" if is_linux else "false")

    print(f"Launching Streamlit ({platform.system()}): {app_path}")
    print(f"  Bind: http://{bind_host}:{port} (headless={headless})")
    print("  Tab 1: Backtest Analyzer  - bt_core parquet visualization")
    print("  Tab 2: Tune Param Space   - Ray Tune contour/3D/collapse dashboard")
    print()
    try:
        subprocess.run(
            [
                sys.executable, "-m", "streamlit", "run", app_path,
                f"--server.address={bind_host}",
                f"--server.port={port}",
                f"--server.headless={headless}",
            ],
            check=True,
        )
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
