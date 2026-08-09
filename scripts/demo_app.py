#!/usr/bin/env python3
"""Demo script: launch Streamlit app.

Usage: python scripts/demo_app.py
"""
import os
import sys
import subprocess


def main():
    app_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "bt_studio", "visual", "st_app.py",
    )
    if not os.path.exists(app_path):
        print(f"Error: app.py not found at {app_path}")
        sys.exit(1)
    print(f"Launching Streamlit: {app_path}")
    print("  Tab 1: Backtest Analyzer  - bt_core parquet visualization")
    print("  Tab 2: Tune Param Space   - Ray Tune contour/3D/collapse dashboard")
    print()
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", app_path, "--server.port=8501"],
            check=True,
        )
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
