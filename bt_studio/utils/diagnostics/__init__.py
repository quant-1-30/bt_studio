"""Pure diagnostics layer (no plotting dependencies).

Currently hosts parameter-space collapse detection used both by the HPO
pipeline (auto-validation) and the visual dashboards.
"""

from .hpo_check import run_collapse_check, build_search_bounds  # noqa: F401
from .collapse import detect_space_collapse
from .recorder import (
    print_collapse_report,
    build_collapse_report,
    save_collapse_report,
)

__all__ = [
    "detect_space_collapse",
    "print_collapse_report",
    "build_collapse_report",
    "save_collapse_report",
]