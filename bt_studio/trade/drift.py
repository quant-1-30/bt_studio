#! /usr/bin/env python3

from __future__ import annotations

import datetime
import logging
import os
from typing import Dict, List, Optional

import polars as pl

from bt_studio.default_config import PRODUCTION_COMMON_PARAMS
from bt_studio.utils.io import atomic_save_json, read_json_safe

log = logging.getLogger(__name__)

METRICS = ("n_trigger", "topk_score", "topk_ret")
_FIELDS = (*METRICS, "macro_state")  


def _rel(cc: dict) -> float:
    raw = os.environ.get("BT_DRIFT_REL", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            log.warning("[Drift] illegal BT_DRIFT_REL=%r, falling back to config default", raw)
    return float(cc.get("drift_rel", PRODUCTION_COMMON_PARAMS["drift_rel"]))


def signal_stats(scored_lf, topk: int) -> pl.DataFrame:
    """Per-day cross-section summary of FSM scores → one row per ``day`` (sorted).

    ``n_trigger`` = DTW-triggered names that day; ``topk_score`` / ``topk_ret`` =
    mean ``fsm_score`` / ``expected_return`` over the ``topk`` highest positive
    scores (null when the day's regime is not tradeable, i.e. no ``fsm_score > 0``).
    """
    pos = pl.col("fsm_score") > 0.0
    pos_score = pl.col("fsm_score").filter(pos)
    return (
        scored_lf
        .group_by("day")
        .agg(
            pl.len().alias("n_trigger"),
            pl.col("macro_state").first(),
            pos_score.sort(descending=True).head(topk).mean().alias("topk_score"),
            pl.col("expected_return")
            .filter(pos)
            .sort_by(pos_score, descending=True)
            .head(topk)
            .mean()
            .alias("topk_ret"),
        )
        .sort("day")
        .collect()
    )


def observed_median(stats: pl.DataFrame) -> Dict[str, Optional[float]]:
    return stats.select([pl.col(m).median() for m in METRICS]).row(0, named=True)


def breaches(stats: dict, median: dict, rel: float) -> List[str]:
    """relative deviation from the median exceeds ``rel`` (``|v − median| / |median| > rel``); null/0 medians are not compared."""
    out = []
    for m in METRICS:
        v = stats.get(m)
        med = median.get(m)
        if v is None or med is None or med == 0.0:
            continue
        if abs(v - med) / abs(med) > rel:
            out.append(m)
    return out


def drift_alerts(record: dict, median: dict, model_id: int) -> List[dict]:
    """A drifted daily ``record`` (``drifted`` = breaching metrics) → monitor-shaped alerts
    (engine log 与 monitor 共用)。"""
    out = []
    for m in record["drifted"]:
        v, med = record[m], median[m]
        out.append({
            "code": f"model_drift_{m}",
            "level": "warn",
            "message": (
                f"model {model_id} live {m}={v:.6g} deviates from OOS median {med:.6g} "
                f"({(v / med - 1.0) * 100.0:+.1f}%) on {record['day']}"
            ),
            "meta": {"model_id": model_id, "day": record["day"], "metric": m,
                     "value": v, "median": med},
        })
    return out


class DriftMonitor:
    """Observe-only daily drift sentinel held by ``TradeEngine``.

    Enabled only when the model's offline OOS scores exist (any non-empty
    ``scores_{model_id}.parquet``); otherwise every ``check`` is a no-op (the
    startup log says why), so live trading never depends on tune artifacts.
    """

    def __init__(self, common_config: dict, model_ckpt: dict, topk: int):
        self.model_id = model_ckpt.get("train_end_month")
        self.topk = int(topk)
        self.rel = _rel(common_config)
        self.med: Optional[Dict[str, Optional[float]]] = None
        self.history: List[dict] = []
        self._path: Optional[str] = None

        score_dir, collapse_dir = common_config.get("tune_score_dir"), common_config.get("tune_collapse_dir")
        if self.model_id is None or not (score_dir and collapse_dir):
            log.info("[Drift] disabled: needs model train_end_month + tune_score_dir/tune_collapse_dir")
            return
        score_path = os.path.join(score_dir, f"scores_{int(self.model_id)}.parquet")
        try:
            self.med = self._load_median(score_path)
        except Exception as e:  
            log.warning("[Drift] disabled: baseline %s unreadable (%s)", score_path, e)
        if self.med is None:
            return
        # 与月度 decay_{model_id}_{feat_col}.json 同目录同命名
        self._path = os.path.join(collapse_dir, f"drift_{int(self.model_id)}_{model_ckpt['feature_col']}.json")
        self.history = (read_json_safe(self._path) or {}).get("history", [])

    def _load_median(self, score_path: str) -> Optional[dict]:
        if not os.path.exists(score_path):
            log.info("[Drift] disabled: no offline OOS scores at %s", score_path)
            return None
        stats = signal_stats(pl.scan_parquet(score_path), self.topk)
        if stats.height == 0:
            log.info("[Drift] disabled: empty OOS scores at %s", score_path)
            return None
        return observed_median(stats)

    def check(self, scored, day: int) -> List[dict]:
        """Observe one entry-time cross-section (``day`` = YYYYMMDD); returns the day's drift alerts.

        Never raises: a monitoring failure (unreadable baseline, full disk …)
        must not block the day's plan.
        """
        if self.med is None:
            return []
        try:
            return self._record(scored, int(day))
        except Exception as e:
            log.warning("[Drift] check skipped for %s: %s", day, e)
            return []

    def _record(self, scored, day: int) -> List[dict]:
        today = datetime.datetime.strptime(str(day), "%Y%m%d").date()
        stats = (
            signal_stats(scored.lazy().filter(pl.col("day") == today), self.topk)
            if scored.width
            else pl.DataFrame()
        )
        row = stats.row(0, named=True) if stats.height else {"n_trigger": 0}
        record = {f: row.get(f) for f in _FIELDS}
        record["day"] = day
        record["drifted"] = breaches(record, self.med, self.rel)
        
        self.history = [r for r in self.history if r["day"] != day] + [record]
        atomic_save_json({"model_id": int(self.model_id), "median": self.med, "history": self.history}, self._path)
        alerts = drift_alerts(record, self.med, int(self.model_id))
        for a in alerts:
            log.warning("[Drift] %s", a["message"])
        return alerts
