#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys


def launch_streamlit_desk() -> None:
    app_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "bt_studio", "visual", "_streamlit.py",
    )
    bind_host = os.environ.get("BT_STUDIO_BIND_HOST", "127.0.0.1")
    port = os.environ.get("STREAMLIT_PORT", "8501")
    headless = os.environ.get("STREAMLIT_HEADLESS", "true" if platform.system() == "Linux" else "false")
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run", app_path,
            f"--server.address={bind_host}",
            f"--server.port={port}",
            f"--server.headless={headless}",
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Online & Offline Visual Evaluation Suite")
    parser.add_argument("--mode",
        choices=("online", "collapse", "drift", "pdf"),
        default=os.environ.get("BT_VISUAL_MODE", "online"),
        help="Evaluation mode: online (Streamlit A/B/C desk) | collapse (PNGs) | drift (WFO PNG) | pdf (backtest report)",
    )
    parser.add_argument("--exp", default=None, help="Ray Tune experiment dir (required for collapse/drift)")
    parser.add_argument("--target", default="metrics_score", help="Target metric column (for collapse/drift)")
    parser.add_argument("--method", default="cubic", choices=("cubic", "linear", "rbf"), help="Surface interpolation method")
    parser.add_argument("--out", default="png", help="Output directory or PDF file path for offline artifacts")
    parser.add_argument("--log", default=None, help="bt_core parquet log for PDF export (required for pdf)")
    args = parser.parse_args()

    if args.mode == "online":
        launch_streamlit_desk()
    elif args.mode == "collapse":
        if not args.exp:
            parser.error("--mode collapse requires --exp <ray_results experiment dir>")
        from bt_studio.visual import export_collapse_pngs

        res = export_collapse_pngs(args.exp, out_dir=args.out, target=args.target, method=args.method)
        print(f"Exported collapse visuals: {res['dashboard']}")
    elif args.mode == "drift":
        if not args.exp:
            parser.error("--mode drift requires --exp <ray_results experiment dir>")
        from bt_studio.visual import export_drift_png

        res = export_drift_png(args.exp, out_dir=args.out, target=args.target)
        print(res["drift_df"].to_string(index=False))
        print(f"Saved WFO drift dashboard: {res['dashboard']}")
    elif args.mode == "pdf":
        if not args.log:
            parser.error("--mode pdf requires --log <bt_core parquet log>")
        from bt_studio.visual import convert_parquet_to_pdf

        pdf_out = args.out if args.out.endswith(".pdf") else None
        out_path = convert_parquet_to_pdf(args.log, pdf_out)
        print(f"Generated PDF report: {out_path}")


if __name__ == "__main__":
    main()
