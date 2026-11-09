#! /usr/bin/env python3

from __future__ import annotations

import atexit
import logging
import os
import platform
import threading
from typing import Any, Dict, Optional

import ray

logger = logging.getLogger(__name__)

# Set single-thread environment defaults before C/Rust libraries initialize
_THREAD_ENV_VARS = {
    "POLARS_MAX_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
}

for _k, _v in _THREAD_ENV_VARS.items():
    os.environ.setdefault(_k, _v)

_RUNTIME_ENV = {"env_vars": _THREAD_ENV_VARS}
_RAY_INIT_LOCK = threading.Lock()


def _get_system_total_ram() -> int:
    """Safely detect total physical RAM in bytes across Linux and macOS."""
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except ImportError:
        pass

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except (ValueError, AttributeError, OSError):
        return 16 * 1024 * 1024 * 1024


def is_ray_ready() -> bool:
    """Return True if the local or remote Ray cluster is initialized."""
    return ray.is_initialized()


def get_ray_cluster_resources() -> Dict[str, float]:
    """Return total cluster compute and memory resources."""
    return ray.cluster_resources() if ray.is_initialized() else {}


def _shutdown() -> None:
    """Gracefully shut down the local Ray runtime at interpreter exit."""
    try:
        if ray.is_initialized():
            ray.shutdown()
            logger.info("[Ray] Local cluster shut down cleanly.")
    except Exception as e:
        logger.debug(f"[Ray] Ignored benign shutdown error: {e}")


def _resolve_platform_ray_defaults() -> Dict[str, Any]:
    """Derive Ray object store and hardware profile for Darwin vs Linux workstations."""
    sys_name = platform.system()
    is_darwin = sys_name == "Darwin"
    is_linux = sys_name == "Linux"
    cpus = max(1, os.cpu_count() or 1)
    total_ram = _get_system_total_ram()

    env_gb = os.environ.get("RAY_OBJECT_STORE_GB")
    if env_gb:
        plasma_limit = int(float(env_gb) * (1024 ** 3))
    else:
        ratio = 0.20 if is_darwin else 0.30
        default_cap_gb = 16 if is_darwin else 64
        plasma_limit = min(int(total_ram * ratio), default_cap_gb * (1024 ** 3))

    if is_linux and os.path.exists("/dev/shm"):
        try:
            shm_stat = os.statvfs("/dev/shm")
            shm_total = shm_stat.f_frsize * shm_stat.f_blocks
            plasma_limit = min(plasma_limit, int(shm_total * 0.80))
        except Exception:
            pass

    return {
        "platform": sys_name,
        "is_darwin": is_darwin,
        "is_linux": is_linux,
        "cpus": cpus,
        "total_ram": total_ram,
        "plasma_limit": plasma_limit,
    }


def resolve_platform_concurrency() -> Dict[str, int]:
    """Derive platform-adaptive concurrency knobs (num_workers, eval_procs, tick_batch_sids)."""
    hw = _resolve_platform_ray_defaults()
    cpus = hw["cpus"]
    ram_gb = hw["total_ram"] / (1024 ** 3)

    is_linux = hw["is_linux"]
    hpo_workers = max(4, int(cpus * 0.75)) if is_linux else min(12, max(2, cpus - 2))
    eval_procs = min(24, max(4, cpus // (3 if cpus >= 32 else 2))) if is_linux else min(8, max(2, cpus // 2))
    tick_batch = (128 if ram_gb >= 128 else 32) if is_linux else (16 if ram_gb >= 64 else 8)

    return {
        "num_workers": int(os.environ.get("HPO_NUM_WORKERS", hpo_workers)),
        "eval_procs": int(os.environ.get("EVAL_PROCS", eval_procs)),
        "tick_batch_sids": int(os.environ.get("TICK_BATCH_SIDS", tick_batch)),
    }


def ensure_ray(
    num_cpus: Optional[int] = None,
    object_store_memory_bytes: Optional[int] = None,
    include_dashboard: Optional[bool] = None,
    logging_level: int = logging.WARNING,
) -> None:
    """Thread-safe singleton initializer for the local Ray runtime."""
    if ray.is_initialized():
        return

    with _RAY_INIT_LOCK:
        if ray.is_initialized():
            return

        hw = _resolve_platform_ray_defaults()
        env_cpus = os.environ.get("RAY_NUM_CPUS")
        cpus = num_cpus or (int(env_cpus) if env_cpus and env_cpus.isdigit() else hw["cpus"])
        plasma_limit = (
            object_store_memory_bytes
            if object_store_memory_bytes is not None
            else hw["plasma_limit"]
        )
        enable_dash = (
            include_dashboard
            if include_dashboard is not None
            else (os.environ.get("RAY_DASHBOARD", "0") == "1")
        )

        init_kwargs: Dict[str, Any] = {
            "num_cpus": cpus,
            "object_store_memory": plasma_limit,
            "runtime_env": _RUNTIME_ENV,
            "include_dashboard": enable_dash,
            "logging_level": logging_level,
            "ignore_reinit_error": True,
        }

        node_ip = os.environ.get("RAY_NODE_IP_ADDRESS", "").strip()
        if node_ip:
            init_kwargs["_node_ip_address"] = node_ip

        dash_host = (
            os.environ.get("RAY_DASHBOARD_HOST")
            or os.environ.get("BT_STUDIO_BIND_HOST")
            or "127.0.0.1"
        )
        if enable_dash:
            init_kwargs["dashboard_host"] = dash_host

        try:
            ray.init(**init_kwargs)
            logger.info(
                f"[Ray] Cluster initialized ({hw['platform']}): num_cpus={cpus}, "
                f"object_store_memory={plasma_limit / (1024**3):.2f}GB, "
                f"dashboard={enable_dash} ({dash_host if enable_dash else 'off'})"
            )
        except Exception as e:
            logger.error(f"[Ray] Fatal error initializing cluster: {e}")
            raise


atexit.register(_shutdown)
