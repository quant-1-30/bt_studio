#! /usr/bin/env python3
"""Unified I/O, hashing, atomic persistence, RPC stream collection, and cache/task directory stores."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pickle
import queue
import re
import shutil
import subprocess
import time
from datetime import datetime
from dotenv import load_dotenv
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy as np
import polars as pl
import reactivex.operators as ops

try:
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover
    fcntl = None

from bt_sdk.utils.util import _merge2DataFrame
from bt_studio.constant import (
    ARTIFACT_ROOT,
    CACHE_ROOT,
    EVAL_CACHE_ROOT,
    EVAL_LOGIC_VERSION,
    EVAL_SEMANTIC_KEYS,
    FEATURE_DIR,
    RAW_PIT_VERSION,
    RESULT_ROOT,
    TASK_DIR_KEYS,
    TUNE_BASE_DIR,
)
from bt_studio.schema import EvalResult

_SAFE_ID_RE = re.compile(r"[^0-9A-Za-z_.-]+")
_DATA_SCOPE_KEYS = (
    "start_date", "end_date", "benchmark",
    "days_since_ipo", "min_active_ratio", "min_amount_rank",
)


# ===========================================================================
# 1. Serialization, Hashing & Safe Readers
# ===========================================================================

def slugify(text: Any, max_len: int = 48, fallback: str = "task") -> str:
    """Sanitize arbitrary text into a filesystem-safe slug."""
    slug = _SAFE_ID_RE.sub("-", str(text)).strip("-.")[:max_len]
    return slug or fallback


@lru_cache(maxsize=1)
def get_git_sha() -> str:
    """Return short Git commit SHA with process-level caching (fallback 'unknown')."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def json_default(obj: Any) -> Any:
    """JSON serialization fallback for NumPy scalars/arrays and arbitrary objects."""
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def sha1_hex(obj: Any, length: Optional[int] = None) -> str:
    """Compute deterministic SHA-1 hex digest over any JSON-serializable object."""
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=json_default)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return digest[:length] if length is not None else digest


def read_json_safe(path: str, default: Any = None) -> Any:
    """Read a JSON file safely, returning default when missing or corrupted."""
    if not path or not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def consume_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} cost {round(time.time() - start_time, 3)}s")
        return result
    return wrapper


# ===========================================================================
# 2. File Locking & Atomic Persistence
# ===========================================================================

@contextlib.contextmanager
def exclusive_file_lock(path: str):
    """Cross-process exclusive file lock (degrades to no-op on platforms without fcntl)."""
    if fcntl is None:
        yield
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def atomic_file_write(target_path: str) -> Generator[str, None, None]:
    """Write to a PID-scoped sibling temp file and atomically replace target_path on exit."""
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    tmp_path = f"{target_path}.tmp_{os.getpid()}"
    try:
        yield tmp_path
        os.replace(tmp_path, target_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def atomic_save_json(data: Any, target_path: str, indent: Optional[int] = 2) -> None:
    with atomic_file_write(target_path) as temp_path:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent, default=json_default)
            f.flush()
            os.fsync(f.fileno())


def atomic_save_parquet(df: pl.DataFrame, target_path: str, statistics: bool = False) -> None:
    with atomic_file_write(target_path) as temp_path:
        df.write_parquet(temp_path, statistics=statistics)


def atomic_save_pickle(obj: Any, target_path: str) -> None:
    with atomic_file_write(target_path) as temp_path:
        with open(temp_path, "wb") as f:
            pickle.dump(obj, f)
            f.flush()
            os.fsync(f.fileno())


# ===========================================================================
# 3. Disk Hygiene & LRU Cache Eviction
# ===========================================================================

def clean_stale_tmp_files(root_dir: Optional[str] = None, max_age_s: float = 3600.0) -> int:
    """Remove orphan .tmp / .tmp_* files older than max_age_s seconds."""
    root = os.path.abspath(root_dir or RESULT_ROOT)
    if not os.path.isdir(root):
        return 0
    cutoff = time.time() - max_age_s
    removed = 0
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if name.endswith(".tmp") or ".tmp_" in name:
                p = os.path.join(dirpath, name)
                try:
                    if os.path.getmtime(p) < cutoff:
                        os.remove(p)
                        removed += 1
                except OSError:
                    continue
    return removed


def evict_cache_lru(
    result_root: Optional[str] = None,
    max_remove: int = 500,
    include_bars: bool = True,
) -> int:
    """Evict oldest regenerable cache/evals and cache/bars files by mtime."""
    root = os.path.abspath(result_root or RESULT_ROOT)
    removed = 0

    eval_dir = os.path.join(root, "cache", "evals")
    if os.path.isdir(eval_dir):
        candidates = []
        for dirpath, _dirs, files in os.walk(eval_dir):
            for name in files:
                if name.endswith(".json"):
                    p = os.path.join(dirpath, name)
                    try:
                        candidates.append((os.path.getmtime(p), p))
                    except OSError:
                        continue
        candidates.sort(key=lambda x: x[0])
        for _, p in candidates[:max_remove]:
            try:
                os.remove(p)
                removed += 1
            except OSError:
                continue

    raw_pit_dir = os.path.join(root, "cache", "bars", RAW_PIT_VERSION)
    if include_bars and os.path.isdir(raw_pit_dir):
        bar_packs = []
        for name in os.listdir(raw_pit_dir):
            if name.startswith("ym=") and name.endswith(".parquet"):
                p = os.path.join(raw_pit_dir, name)
                try:
                    bar_packs.append((os.path.getmtime(p), p))
                except OSError:
                    continue
        bar_packs.sort(key=lambda x: x[0])
        for _, p in bar_packs[: max(1, len(bar_packs) // 3)]:
            try:
                os.remove(p)
                meta_p = p[: -len(".parquet")] + ".meta.json"
                if os.path.exists(meta_p):
                    os.remove(meta_p)
                removed += 1
            except OSError:
                continue

    return removed


def guard_disk_watermark(target_dir: str) -> None:
    """Trigger LRU cache eviction when disk usage on target_dir exceeds BT_STUDIO_DISK_CRIT_RATIO."""
    try:
        du = shutil.disk_usage(target_dir)
        crit_thr = float(os.environ.get("BT_STUDIO_DISK_CRIT_RATIO", "0.95"))
        if du.total > 0 and (du.used / du.total) >= crit_thr:
            evict_cache_lru(max_remove=500, include_bars=True)
    except Exception:
        pass


def io_setup() -> None:
    load_dotenv()
    os.makedirs(ARTIFACT_ROOT, exist_ok=True)
    os.makedirs(CACHE_ROOT, exist_ok=True)
    clean_stale_tmp_files(max_age_s=3600.0)


# ===========================================================================
# 4. DataFrame, RPC Stream & Raw PIT Monthly Cache Store
# ===========================================================================

def collect_lazy_safe(lf_or_df: Any) -> pl.DataFrame:
    if isinstance(lf_or_df, pl.LazyFrame):
        try:
            return lf_or_df.collect(engine="streaming")
        except Exception:
            return lf_or_df.collect()
    return lf_or_df


def _collect_stream_sync(observable, timeout: float = 600.0) -> Dict[bytes, pl.DataFrame]:
    """Synchronously collect an RxPY RPC stream with explicit per-batch timeout protection."""
    q: queue.Queue = queue.Queue()
    subscription = observable.pipe(
        ops.map(lambda data: data["data"]),
        ops.share(),
    ).subscribe(
        on_next=q.put,
        on_error=q.put,
        on_completed=lambda: q.put(StopIteration),
    )

    tables = []
    try:
        while True:
            try:
                msg = q.get(timeout=timeout)
            except queue.Empty:
                raise TimeoutError(
                    f"stream collect timed out after {timeout}s ({len(tables)} batches received)"
                ) from None
            if msg is StopIteration:
                break
            if isinstance(msg, Exception):
                raise msg
            tables.append(msg)
        return _merge2DataFrame(tables)
    finally:
        try:
            subscription.dispose()
        except Exception:
            pass


def fetch_hfq_factors_safe(mdapi: Any, body: Any, topic: int) -> dict:
    """Safely fetch causal backward adjustment factors from mdapi, falling back to {} on error or mock."""
    get_factor_fn = getattr(mdapi, "get_factor", None)
    if not callable(get_factor_fn):
        return {}
    try:
        res = get_factor_fn(body, topic)
        return res if isinstance(res, dict) else {}
    except Exception:
        return {}


class RawPitCacheStore:
    """Minute-level Raw PIT monthly Parquet pack cache and cross-process atomic merger."""

    def __init__(self, raw_pit_dir: str, bars_schema: Dict[str, Any]):
        self.raw_pit_dir = raw_pit_dir
        self.bars_schema = bars_schema

    def pack_path(self, ym: int) -> str:
        return os.path.join(self.raw_pit_dir, f"ym={ym}.parquet")

    def meta_path(self, ym: int) -> str:
        return os.path.join(self.raw_pit_dir, f"ym={ym}.meta.json")

    def covered_sids(self, ym: int) -> set[str]:
        if not os.path.exists(self.pack_path(ym)):
            return set()
        meta_obj = read_json_safe(self.meta_path(ym), default={}) or {}
        return set(meta_obj.get("covered_sids", []))

    def known_sids(self, ym: int) -> set[str]:
        if not os.path.exists(self.pack_path(ym)):
            return set()
        meta_obj = read_json_safe(self.meta_path(ym), default={}) or {}
        covered = set(meta_obj.get("covered_sids", []))
        if not covered:
            return set()
        return covered | set(meta_obj.get("empty_sids", []))

    def merge_and_save_month(
        self,
        ym: int,
        new_frames: list[pl.DataFrame],
        added_sids: set[str],
        empty_sids: Optional[set[str]] = None,
    ) -> None:
        p_path = self.pack_path(ym)
        m_path = self.meta_path(ym)
        frames = [df for df in new_frames if df.height > 0]

        with exclusive_file_lock(p_path + ".lock"):
            prev_meta = read_json_safe(m_path, default={}) or {}
            latest_covered = set(prev_meta.get("covered_sids", [])) | added_sids
            latest_empty = (set(prev_meta.get("empty_sids", [])) | (empty_sids or set())) - latest_covered

            if not frames and os.path.exists(p_path):
                if latest_covered:
                    atomic_save_json(
                        {
                            "ym": ym,
                            "covered_sids": sorted(latest_covered),
                            "empty_sids": sorted(latest_empty),
                            "rows": int(prev_meta.get("rows", 0)),
                            "version": RAW_PIT_VERSION,
                        },
                        m_path,
                    )
                return

            replaced_bytes = list(dict.fromkeys(
                s for df in frames for s in df["sid"].unique().to_list()
            ))
            if os.path.exists(p_path):
                try:
                    old = pl.read_parquet(p_path)
                    kept = old.filter(~pl.col("sid").is_in(replaced_bytes)) if replaced_bytes else old
                    if kept.height > 0:
                        frames.append(kept)
                except Exception:
                    pass
            merged = (pl.concat(frames) if frames else pl.DataFrame(schema=self.bars_schema)).sort(
                ["sid", "day", "bar_idx"]
            )
            atomic_save_parquet(merged, p_path, statistics=True)
            atomic_save_json(
                {
                    "ym": ym,
                    "covered_sids": sorted(latest_covered),
                    "empty_sids": sorted(latest_empty),
                    "rows": merged.height,
                    "version": RAW_PIT_VERSION,
                },
                m_path,
            )

    def verify_and_scan(
        self,
        months: list[int],
        sids: list[bytes],
        start_date: int,
        end_date: int,
    ) -> pl.LazyFrame:
        packs = [self.pack_path(ym) for ym in months if os.path.exists(self.pack_path(ym))]
        if not packs:
            return pl.LazyFrame(schema=self.bars_schema)
        d_lo = datetime.strptime(str(start_date), "%Y%m%d").date()
        d_hi = datetime.strptime(str(end_date), "%Y%m%d").date()
        return (
            pl.scan_parquet(packs)
            .filter(pl.col("sid").is_in(list(dict.fromkeys(sids))))
            .filter(pl.col("day").is_between(d_lo, d_hi))
        )


# ===========================================================================
# 5. Stage 2 Window Evaluation Cache
# ===========================================================================

def compute_eval_data_fingerprint(
    common_config: dict,
    dret_lf: Optional[pl.LazyFrame] = None,
) -> str:
    if "_eval_data_fp" in common_config:
        return str(common_config["_eval_data_fp"])
    scope_cfg = {k: common_config.get(k) for k in _DATA_SCOPE_KEYS if k in common_config}
    dret_sig: Any = None
    try:
        schema = dret_lf.collect_schema().names() if dret_lf is not None else ()
        if "close" in schema:
            dret_sig = dret_lf.select([pl.len().alias("n"), pl.col("close").sum().alias("s")]).collect().row(0)
    except Exception:
        dret_sig = None
    return sha1_hex({"raw_pit": RAW_PIT_VERSION, "scope": scope_cfg, "dret": dret_sig})


def build_eval_cache_locator(
    common_config: dict,
    grid_config: Dict[str, list],
    feature_names: List[str],
    fingerprints: Optional[Dict[str, str]] = None,
    cache_ctx: Tuple[str, str] = (EVAL_CACHE_ROOT, EVAL_LOGIC_VERSION),
) -> Tuple[str, Callable[[str, Any], str]]:
    cache_root, logic_version = cache_ctx
    data_fp = str(
        common_config.get("_eval_data_fp")
        or sha1_hex({k: common_config.get(k) for k in _DATA_SCOPE_KEYS if k in common_config})
    )
    fps = fingerprints or {f: sha1_hex(f) for f in feature_names}
    eval_cfg = {k: common_config.get(k) for k in EVAL_SEMANTIC_KEYS if k in common_config}
    gh = sha1_hex({"grid": grid_config, "eval_cfg": eval_cfg})

    def _cache_path(feat: str, ym: Any) -> str:
        ast_fp = fps.get(feat, sha1_hex(feat))
        return os.path.join(cache_root, logic_version, data_fp[:8], ast_fp[:8], f"{ym}_{gh[:10]}.json")

    return data_fp, _cache_path


def eval_cache_load(path: str) -> Optional[EvalResult]:
    d = read_json_safe(path)
    if not isinstance(d, dict) or "feature_name" not in d:
        return None
    try:
        return EvalResult(
            feature_name=d["feature_name"],
            status=d.get("status", "failed"),
            score=float(d.get("score", -500.0)),
            u_pval=float(d.get("u_pval", 1.0)),
            trigger_count=int(d.get("trigger_count", 0)),
            valid_sample_ratio=float(d.get("valid_sample_ratio", 0.0)),
            autocorr=float(d.get("autocorr", 0.0)),
            elapsed_s=float(d.get("elapsed_s", 0.0)),
            reason=d.get("reason", ""),
            config=d.get("config", {}),
        )
    except Exception:
        return None


def eval_cache_save(res: EvalResult, path: str) -> None:
    try:
        atomic_save_json({
            "feature_name": res.feature_name,
            "status": res.status,
            "score": res.score,
            "u_pval": res.u_pval,
            "trigger_count": res.trigger_count,
            "valid_sample_ratio": res.valid_sample_ratio,
            "autocorr": res.autocorr,
            "elapsed_s": res.elapsed_s,
            "reason": res.reason,
            "config": res.config or {},
        }, path, indent=None)
    except Exception:
        pass
