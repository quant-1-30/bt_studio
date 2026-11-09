#! /usr/bin/env python3
"""Matrix Profile motif discovery (STUMPY) and Numba-accelerated subsequence DTW kernels."""

from __future__ import annotations

import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import numba
import numpy as np
import polars as pl
import stumpy

from bt_studio.constant import TRADING_BARS_PER_DAY
from bt_studio.utils.math_func import causal_keep_bars, resolve_dtw_workers

warnings.filterwarnings("ignore", category=RuntimeWarning, module="stumpy")


@numba.njit(nogil=True, cache=True)
def _compute_keogh_envelope(motif_c: np.ndarray, dtw_w: int) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute the LB_Keogh (upper, lower) envelope matching dtaidistance.dtw.lb_keogh(window=dtw_w)."""
    m = len(motif_c)
    band = m if dtw_w <= 0 else (dtw_w - 1)
    u = np.empty(m, dtype=np.float64)
    l = np.empty(m, dtype=np.float64)
    for i in range(m):
        seg = motif_c[max(0, i - band) : min(m, i + band + 1)]
        u[i] = np.max(seg)
        l[i] = np.min(seg)
    return u, l


@numba.njit(nogil=True, cache=True)
def _banded_dtw_sq(s1: np.ndarray, s2: np.ndarray, w: int, max_dist_sq: float) -> float:
    """Sakoe-Chiba banded squared DTW distance matching dtaidistance.dtw.distance(window=w, max_dist=...)."""
    r = len(s1)
    c = len(s2)
    if r == 0 or c == 0:
        return np.inf
    band = max(r, c) if w <= 0 else (w - 1)
    r_diff = max(0, r - c)
    c_diff = max(0, c - r)
    dp = np.full((r + 1, c + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, r + 1):
        row_min = np.inf
        j_start = max(1, i - r_diff - band)
        j_end = min(c, i + c_diff + band)
        for j in range(j_start, j_end + 1):
            val = (s1[i - 1] - s2[j - 1]) ** 2 + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
            dp[i, j] = val
            if val < row_min:
                row_min = val
        if row_min > max_dist_sq:
            return np.inf
    return dp[r, c] if dp[r, c] <= max_dist_sq else np.inf


@numba.njit(nogil=True, cache=True)
def _row_min_subseq_dtw_nb(
    row_curve: np.ndarray,
    z_motif: np.ndarray,
    u: np.ndarray,
    l: np.ndarray,
    dtw_w: int,
    threshold_d: float,
    eps: float = 1e-8,
) -> float:
    """Compiled GIL-free per-curve minimum subsequence DTW with inline Z-norm and LB_Keogh pruning."""
    L = len(row_curve)
    m = len(z_motif)
    if L < m or m == 0 or np.isnan(threshold_d) or threshold_d < 0.0:
        return np.inf

    min_dist_sq = threshold_d * threshold_d
    best_dist_sq = np.inf
    sub = np.empty(m, dtype=np.float64)

    for w_idx in range(L - m + 1):
        win = row_curve[w_idx : w_idx + m]
        if np.isnan(win).any():
            continue

        std = np.std(win)
        if std < eps:
            continue
        mean = np.mean(win)

        lb_sq = 0.0
        for k in range(m):
            zv = (win[k] - mean) / std
            sub[k] = zv
            if zv > u[k]:
                lb_sq += (zv - u[k]) ** 2
            elif zv < l[k]:
                lb_sq += (l[k] - zv) ** 2
        if lb_sq > min_dist_sq:
            continue

        d_sq = _banded_dtw_sq(sub, z_motif, dtw_w, min_dist_sq)
        if d_sq <= min_dist_sq:
            min_dist_sq = d_sq
            best_dist_sq = d_sq
            if best_dist_sq <= eps:
                break

    return math.sqrt(best_dist_sq)


@numba.njit(nogil=True, cache=True)
def _chunk_dtw_nb(
    curves_chunk: np.ndarray,
    z_motif: np.ndarray,
    u: np.ndarray,
    l: np.ndarray,
    dtw_w: int,
    threshold_d: float,
    eps: float,
) -> np.ndarray:
    """Process a contiguous chunk of 2D curves with GIL released."""
    n = curves_chunk.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = _row_min_subseq_dtw_nb(curves_chunk[i], z_motif, u, l, dtw_w, threshold_d, eps)
    return out


def prepare_curves(panel_df: pl.DataFrame, tune_config: dict, common_config: dict) -> np.ndarray:
    """Convert fixed-length lag_0 list column into a 2D (N, bars_per_day) float64 array with right-tail NaN causal masking."""
    if panel_df.height == 0:
        return np.empty((0, 0), dtype=np.float64)

    ds = int(tune_config["downsample"])
    total_bars = int(common_config.get("bars_per_day", TRADING_BARS_PER_DAY))
    bars_per_day = total_bars // ds

    curves_2d = np.array(
        panel_df["lag_0"].list.to_array(bars_per_day).to_numpy(),
        dtype=np.float64,
        copy=True,
    )

    exclude_bars = int(common_config.get("exclude_bars", 0))
    if exclude_bars > 0:
        keep_bars = causal_keep_bars(common_config, downsample=ds)
        n_mask = max(0, bars_per_day - keep_bars)
        if n_mask > 0:
            curves_2d[:, -n_mask:] = np.nan

    return curves_2d


def get_candidate_motifs(raw_array: np.ndarray, config: dict, common_config: dict) -> List[np.ndarray]:
    """Extract Top-K candidate motifs via Matrix Profile (accepts 1D series or 2D sampled curves with length-m NaN barriers)."""
    m = int(config["m"])
    if raw_array.size < m or m < 3:
        return []

    if raw_array.ndim == 2:
        n_curves, curve_len = raw_array.shape
        padded = np.full((n_curves, curve_len + m), np.nan, dtype=np.float64)
        padded[:, :curve_len] = raw_array
        padded[~np.isfinite(padded)] = np.nan
        raw_series = np.ascontiguousarray(padded.ravel()[:-m], dtype=np.float64)
    else:
        raw_series = np.ascontiguousarray(raw_array.ravel(), dtype=np.float64)

    if np.count_nonzero(np.isfinite(raw_series)) < m * 2:
        return []

    threshold_d = float(config.get("threshold_d", np.inf))
    topk = int(common_config.get("topk", 5))
    eps = float(common_config.get("eps", 1e-8))

    mp = stumpy.stump(raw_series, m=m)
    distances = np.ascontiguousarray(mp[:, 0], dtype=np.float64)
    distances[~np.isfinite(distances)] = np.inf

    candidates: List[np.ndarray] = []

    while len(candidates) < topk:
        anchor_idx = int(np.argmin(distances))
        if distances[anchor_idx] > threshold_d or np.isinf(distances[anchor_idx]):
            break

        motif = raw_series[anchor_idx : anchor_idx + m].copy()
        if not np.isnan(motif).any() and float(np.std(motif)) >= eps:
            candidates.append(motif)

        # Mask exclusion zone around anchor and its paired neighbor so the next argmin picks a distinct motif
        nb_idx = int(mp[anchor_idx, 1]) if mp.shape[1] > 1 and np.isfinite(mp[anchor_idx, 1]) else -1
        for idx in (anchor_idx, nb_idx):
            if idx >= 0:
                distances[max(0, idx - m) : min(distances.size, idx + m + 1)] = np.inf

    return candidates


def calc_min_subseq_dtw(
    row_curve: np.ndarray,
    z_motif: np.ndarray,
    dtw_w: int,
    threshold_d: float,
    eps: float = 1e-8,
) -> float:
    """Compute minimum subsequence DTW distance for a single intraday curve via compiled Numba kernel."""
    motif_c = np.ascontiguousarray(z_motif, dtype=np.float64)
    u, l = _compute_keogh_envelope(motif_c, int(dtw_w))
    return float(
        _row_min_subseq_dtw_nb(
            np.ascontiguousarray(row_curve, dtype=np.float64),
            motif_c,
            u,
            l,
            int(dtw_w),
            float(threshold_d),
            float(eps),
        )
    )


def compute_dtw_distances(
    curves_2d: np.ndarray,
    motif: np.ndarray,
    threshold_d: float,
    common_config: dict,
) -> np.ndarray:
    """Compute subsequence DTW distances across all curves using GIL-releasing Numba chunks."""
    n_curves = curves_2d.shape[0]
    if n_curves == 0:
        return np.empty(0, dtype=np.float64)

    raw_motif = np.asarray(motif, dtype=np.float64).ravel()
    m = len(raw_motif)
    dtw_w = max(3, int(m * common_config.get("dtw_window_frac", 0.1)))
    eps = float(common_config.get("eps", 1e-8))
    m_std = float(np.std(raw_motif))
    z_motif = np.ascontiguousarray(
        (raw_motif - np.mean(raw_motif)) / (m_std if m_std > eps else 1.0),
        dtype=np.float64,
    )
    u, l = _compute_keogh_envelope(z_motif, dtw_w)
    curves_c = np.ascontiguousarray(curves_2d, dtype=np.float64)
    max_workers = resolve_dtw_workers(common_config)

    if max_workers <= 1 or n_curves < 128:
        return _chunk_dtw_nb(curves_c, z_motif, u, l, dtw_w, float(threshold_d), eps)

    chunk_size = max(64, (n_curves + max_workers - 1) // max_workers)
    chunks = [curves_c[i : i + chunk_size] for i in range(0, n_curves, chunk_size)]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(chunks))) as executor:
        parts = list(
            executor.map(
                lambda ch: _chunk_dtw_nb(ch, z_motif, u, l, dtw_w, float(threshold_d), eps),
                chunks,
            )
        )
    return np.concatenate(parts)
