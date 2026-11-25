#!/usr/bin/env python3

from __future__ import annotations

from typing import Any, Dict, List

import bt_core as bt
from bt_protocol._protocol import SnapshotBody


def _extract_score_and_ret(info: Any) -> tuple[float, float | None]:
    if isinstance(info, dict):
        sc = float(info.get("fsm_score", info.get("score", 0.0)) or 0.0)
        er = info.get("expected_return")
        return sc, (float(er) if er is not None else None)
    val = float(info) if isinstance(info, (int, float)) else 0.0
    return val, None


def sizer_params(common_config: dict) -> dict:
    """Single source for ``FSMSizer`` kwargs derived from ``common_config``.

    Backtest (``cerebro.addsizer``) and live (``make_sizer``) both build through
    this so the stake / concentration cap / tranche decay weights can never drift.
    """
    from bt_studio.utils.math_func import calculate_decay_weights

    t1_rets = dict(common_config["T1_rets"])
    sorted_horizons = sorted(t1_rets.items(), key=lambda kv: int(kv[1]))
    decay_w = calculate_decay_weights(
        t1_rets, half_life_minutes=float(common_config.get("decay_minutes", 15.0))
    )
    return {
        "stake": float(common_config.get("stake", 0.9)),
        "max_single_weight": float(common_config.get("max_single_weight", 0.5)),
        "exit_weights": [decay_w[name] for name, _ in sorted_horizons],
    }


class FSMSizer(bt.Sizer):
    """Conviction-weighted, regime-scaled sizer driven directly by fsm_score and expected_return.

    ``max_single_weight`` is a per-position *concentration* cap, not a blanket
    low weight. It only binds when the natural allocation ``stake / n`` exceeds
    it — i.e. on sparse-signal days (``n = 1`` → capped at ``max_single_weight``,
    ``n = 2`` → ``stake/2`` each). For ``n >= 2`` with the default ``stake=0.9``
    and ``max_single_weight=0.5`` the cap is inactive, so the full ``stake``
    budget is deployed. Keep it ≥ ``stake / 2`` so it never starves deployment.
    """

    def __init__(
        self,
        stake: float = 0.90,
        target_expected_ret: float = 0.003,
        max_single_weight: float = 0.5,
        exit_weights: List[float] | None = None,
    ):
        self.stake = float(stake)
        self.target_expected_ret = max(1e-6, float(target_expected_ret))
        self.max_single_weight = float(max_single_weight)
        self.exit_weights = list(exit_weights) if exit_weights else []
        self._exit_stage = 0

    def set_exit_stage(self, stage_idx: int) -> None:
        self._exit_stage = max(0, int(stage_idx))

    def _sell_fraction(self) -> float:
        if not self.exit_weights or self._exit_stage >= len(self.exit_weights) - 1:
            return 1.0
        rem_weight = sum(self.exit_weights[self._exit_stage :])
        if rem_weight <= 1e-8:
            return 1.0
        return min(1.0, max(0.0, float(self.exit_weights[self._exit_stage]) / rem_weight))

    def _getsizing(self, topk_info: Dict[bytes, Any], snapshot: SnapshotBody, isbuy: bool):
        if not isbuy:
            frac = self._sell_fraction()
            if not topk_info:
                return {p.sid: frac for p in snapshot.positions if p.size > 0}
            return {
                p.sid: (0.0 if _extract_score_and_ret(topk_info.get(p.sid))[0] > 0.0 else frac)
                for p in snapshot.positions
                if p.size > 0
            }
        if not topk_info:
            return {}

        raw_weights: Dict[bytes, float] = {}
        macro_exposures: List[float] = []

        for sid, info in topk_info.items():
            score, exp_ret = _extract_score_and_ret(info)
            if exp_ret is not None:
                macro_exposures.append(0.0 if exp_ret <= 0.0 else min(1.0, exp_ret / self.target_expected_ret))
            else:
                macro_exposures.append(1.0 if score > 0.0 else 0.0)
            raw_weights[sid] = score if (score > 0.0 and (exp_ret is None or exp_ret > 0.0)) else 0.0

        total_raw = sum(raw_weights.values())
        if total_raw <= 0.0:
            return {sid: 0.0 for sid in topk_info}

        avg_exposure = float(sum(macro_exposures) / len(macro_exposures)) if macro_exposures else 0.0
        effective_budget = self.stake * avg_exposure
        if effective_budget <= 0.0:
            return {sid: 0.0 for sid in topk_info}

        return {
            sid: min(self.max_single_weight, effective_budget * (w / total_raw)) if w > 0.0 else 0.0
            for sid, w in raw_weights.items()
        }
