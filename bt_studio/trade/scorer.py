#! /usr/bin/env python3

from __future__ import annotations

from typing import Any, Dict

import polars as pl

from bt_studio.compiler.ast import compile_feature_lazy
from bt_studio.pipeline.inference import FSMPredictor
from bt_studio.pipeline.preprocess.panel import extract_curves_from_panel
from bt_studio.utils.math_func import limit_pct
from bt_studio.utils.sid import sid_to_bytes


def score_live(
    model_ckpt: Dict[str, Any],
    bars: pl.DataFrame,
    common_config: Dict[str, Any],
) -> pl.DataFrame:
    """Score live intraday bars against a trained FSM model.

    Parameters
    ----------
    model_ckpt:
        checkpoint with ``config`` / ``motif`` / ``fsm_matrix`` /
        ``feature_col`` / (optional) ``ast_recipe``.
    bars:
        long frame ``day, sid, bar_idx`` + base market cols
        (``open/high/low/close/volume/amount``), END-labeled 240-bar skeleton.
    common_config:
        ``T1_rets`` / ``decay_minutes`` / ``exclude_bars`` / ``ranking_window`` / ``eps``.

    Returns
    -------
    ``day, sid, distance, macro_state, expected_return, fsm_score`` sorted by
    ``(day, fsm_score desc)`` — the same schema as ``scores_*.parquet``.

    Notes
    -----
    ``macro_state`` is a *prior-day* rolling-quantile regime, so pass a window of
    at least ``ranking_window`` trading days (not just today) or every row will
    carry a null ``macro_state`` and be dropped by the predictor.
    """
    feature_col = model_ckpt["feature_col"]
    ast_recipe = model_ckpt.get("ast_recipe")
    tune_config = model_ckpt["config"]

    cc = dict(common_config)
    cc["feature_col"] = feature_col

    feat_lf = compile_feature_lazy(bars.lazy(), feature_col, ast_recipe)
    curve_lf = extract_curves_from_panel(feat_lf, tune_config, cc)
    return FSMPredictor(model_ckpt, cc).predict(curve_lf)


def build_topk(
    scored_df: pl.DataFrame,
    quotes: Dict[str, Dict[str, float]],
    topk: int,
) -> Dict[bytes, Dict[str, float]]:
    """Convert ``score_live`` output to the rich ``topk_info`` dict ``Pnc`` consumes.

    Filters ``fsm_score > 0`` (matching ``ParquetPatch``), then walks the
    score-descending rows taking the first ``topk`` *acceptable* candidates:
    a name rejected for a missing quote or an entry-side limit-up
    (``entry_close / preclose - 1 >= limit_pct``; see ``constant.limit_pct``)
    frees its slot for the next-best one, so rejected candidates never shrink
    the pool below ``topk``. ``sid`` keys are bytes.
    """
    out: Dict[bytes, Dict[str, float]] = {}
    scored = scored_df.filter(pl.col("fsm_score") > 0.0).sort("fsm_score", descending=True)
    for row in scored.iter_rows(named=True):
        if len(out) >= topk:
            break
        ticker = row["sid"]
        q = quotes.get(ticker)
        if q is None:
            continue
        preclose = float(q.get("preclose") or 0.0)
        entry_close = float(q.get("entry_close") or 0.0)
        if preclose > 0 and entry_close > 0 and entry_close / preclose - 1.0 >= limit_pct(ticker):
            continue
        out[sid_to_bytes(ticker)] = {
            "fsm_score": float(row["fsm_score"]),
            "expected_return": float(row["expected_return"]),
            "preclose": preclose,
            "entry_close": entry_close,
        }
    return out
