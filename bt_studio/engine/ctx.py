#! /usr/bin/env python3

from __future__ import annotations

import atexit
import ray
import logging
import threading


# ensure_ray threadsafe
_RAY_INIT_LOCK = threading.Lock()

# CPU Thrashing
_RUNTIME_ENV = {
    "env_vars": {
        "POLARS_MAX_THREADS": "1",
        "RAYON_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    },
}

def is_ray_ready() -> bool:
    return ray.is_initialized()


def get_ray_cluster_resources() -> Dict[str, float]:
    if not ray.is_initialized():
        return {}
    return ray.cluster_resources()


def _shutdown() -> None:
    try:
        if ray.is_initialized():
            ray.shutdown()
            logger.info("[Ray] 本地集群已安全关闭。")
    except Exception:
        pass


def ensure_ray(
    num_cpus: Optional[int] = None,
    object_store_memory_bytes: Optional[int] = None,
    include_dashboard: bool = False,
    logging_level: int = logging.WARNING,
) -> None:
    """
    Ray no-op 
    
    Args:
        num_cpus: default os.cpu_count
        object_store_memory_bytes: Plasma object (byte), default 30% RAM
        include_dashboard: Web Dashboard 
        logging_level: int
    """
    import os
    if ray.is_initialized():
        return

    with _RAY_INIT_LOCK:
        # Double-Checked Locking
        if ray.is_initialized():
            return

        cpus = num_cpus or max(1, os.cpu_count() or 1)

        init_kwargs: Dict[str, Any] = {
            "num_cpus": cpus,
            "runtime_env": _RUNTIME_ENV,
            "ignore_reinit_error": True,
            "include_dashboard": include_dashboard,
            "logging_level": logging_level,
        }

        if object_store_memory_bytes is not None:
            init_kwargs["_memory"] = object_store_memory_bytes

        try:
            ray.init(**init_kwargs)
            logger.info(
                f"[Ray] 本地集群初始化成功: num_cpus={cpus}, "
                f"dashboard={include_dashboard}"
            )
        except Exception as e:
            logger.error(f"[Ray] 集群初始化失败: {e}")
            raise


atexit.register(_shutdown)
