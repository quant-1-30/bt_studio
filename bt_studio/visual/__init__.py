from .bkh import Plot
from .collapse import (
    plot_tune_contour,
    plot_tune_landscape_3d,
    detect_space_collapse,
    print_collapse_report,
    plot_collapse_dashboard,
    load_real_ray_results,
)

# Backward-compat alias used by scripts/run_collapse.py
load_ray_results = load_real_ray_results

__all__ = [
    "Plot",
    "plot_tune_contour",
    "plot_tune_landscape_3d",
    "detect_space_collapse",
    "print_collapse_report",
    "plot_collapse_dashboard",
    "load_real_ray_results",
    "load_ray_results",
]