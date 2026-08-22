#!/usr/bin/env python3

import os
import sys
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bt_studio.visual import (
    load_ray_results,
    detect_space_collapse,
    plot_tune_contour,
    plot_tune_landscape_3d,
    plot_collapse_dashboard,
)
from bt_studio.utils.diagnostics.recorder import print_collapse_report


def main():
    parser = argparse.ArgumentParser(description="Visualize Ray Tune parameter space")
    parser.add_argument(
        "--exp", default="/tmp/ray_results/fsm_hpo_201508",
        help="Ray Tune experiment directory",
    )
    parser.add_argument("--target", default="metrics_score", help="Target metric column")
    parser.add_argument("--method", default="cubic", choices=["cubic", "linear", "rbf"])
    parser.add_argument("--out", default="png", help="Output directory for PNGs")
    args = parser.parse_args()

    if not os.path.exists(args.exp):
        print(f"Error: experiment dir not found: {args.exp}")
        print("Available experiments:")
        base = os.path.dirname(args.exp)
        if os.path.isdir(base):
            for d in sorted(os.listdir(base)):
                print(f"  {os.path.join(base, d)}")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    # =========================================================================
    # 1. Load Ray Tune results
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Loading Ray Tune results from: {args.exp}")
    print(f"{'='*60}")
    df = load_ray_results(args.exp)
    print(f"Loaded {len(df)} trials, {len(df.columns)} columns")

    config_cols = [c for c in df.columns if c.startswith("config/")]
    print(f"Parameters: {[c.replace('config/', '') for c in config_cols]}")
    print(f"Target: {args.target}")
    print(f"Score range: [{df[args.target].min():.2f}, {df[args.target].max():.2f}]")

    search_bounds = {
        "downsample": [3, 4, 5],
        "motif_minutes": [30, 45, 60, 90],
        "threshold_r": [0.6, 0.8],
    }

    # =========================================================================
    # 2. Collapse Detection
    # =========================================================================
    print(f"\n{'='*60}")
    print("Collapse Detection")
    print(f"{'='*60}")
    diag = detect_space_collapse(df, args.target, search_bounds)
    print_collapse_report(diag)

    # =========================================================================
    # 3. Contour Plot
    # =========================================================================
    print(f"\n{'='*60}")
    print("Generating Contour Plot...")
    print(f"{'='*60}")
    fig, ax, cs = plot_tune_contour(
        df, "threshold_r", "downsample", args.target,
        method=args.method, show_samples=True, show_best=True,
    )
    out1 = os.path.join(args.out, "vis_contour.png")
    fig.savefig(out1, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1}")

    # =========================================================================
    # 4. 3D Landscape
    # =========================================================================
    print(f"\n{'='*60}")
    print("Generating 3D Landscape...")
    print(f"{'='*60}")
    fig, ax, surf = plot_tune_landscape_3d(
        df, "threshold_r", "downsample", args.target,
        method=args.method, show_samples=True, show_best=True,
    )
    out2 = os.path.join(args.out, "vis_3d.png")
    fig.savefig(out2, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")

    # =========================================================================
    # 5. Dashboard
    # =========================================================================
    print(f"\n{'='*60}")
    print("Generating Collapse Dashboard...")
    print(f"{'='*60}")
    fig, diag2 = plot_collapse_dashboard(
        df, args.target, search_bounds, method=args.method,
    )
    out3 = os.path.join(args.out, "vis_dashboard.png")
    fig.savefig(out3, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out3}")

    # =========================================================================
    # 6. Done
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Done! 3 PNGs generated in {args.out}/")
    print(f"  - vis_contour_test.png  (2D contour)")
    print(f"  - vis_3d_test.png       (3D landscape)")
    print(f"  - vis_dashboard_test.png (full dashboard)")
    print(f"{'='*60}\n")

    # Try to open dashboard in default viewer
    try:
        if sys.platform == "darwin":
            os.system(f"open '{out3}'")
        elif sys.platform.startswith("linux"):
            os.system(f"xdg-open '{out3}' &")
        print(f"Opened {out3} in default viewer")
    except Exception:
        pass


if __name__ == "__main__":

    # 1. 2D contour plot —— meshgrid 渲染参数响应曲面 等高线 + 采样点叠加
    # 2. 3D landscape  —— 改进版 3D 曲面 支持 RBF 外推、标注最优点
    # 3. 塌陷检测      —— spread_ratio / CV / KDE熵 / landscape方差比 判定参数是否塌陷
    # 4. 综合仪表盘    —— 多面板组合图 contour + 3D + 边缘分布 + 采样轨迹

    main()
