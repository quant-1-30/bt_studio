#!/usr/bin/env python3
"""Portfolio position sizing plugins for bt_core backtesting."""

from __future__ import annotations

from typing import Any, Dict, List

import bt_core as bt
from bt_protocol._protocol import SnapshotBody


def _sell_all_positions(snapshot: SnapshotBody) -> Dict[bytes, float]:
    return {p.sid: 1.0 for p in snapshot.positions if p.size > 0}


class FixedSize(bt.Sizer):
    """Equal-weight sizer allocating stake / N across selected candidates."""

    def __init__(self, stake: float = 0.9, **kwargs):
        self.stake = float(stake)

    def _getsizing(self, topk_info: Dict[bytes, Any], snapshot: SnapshotBody, isbuy: bool):
        if not isbuy:
            return _sell_all_positions(snapshot)
        ratio = self.stake / len(topk_info) if len(topk_info) > 1 else self.stake
        return {sid: ratio for sid in topk_info}


class WeightedSizer(bt.Sizer):
    """Score-proportional sizer allocating stake proportional to positive candidate scores."""

    def __init__(self, stake: float = 0.9, **kwargs):
        self.stake = float(stake)

    def _getsizing(self, topk_info: Dict[bytes, Any], snapshot: SnapshotBody, isbuy: bool):
        if not isbuy:
            return _sell_all_positions(snapshot)
        if not topk_info:
            return {}
        total_score = sum(max(float(info.get("score", 0.0)), 0.0) for info in topk_info.values())
        if total_score <= 0.0:
            return {sid: 0.0 for sid in topk_info}
        return {
            sid: self.stake * max(float(info.get("score", 0.0)), 0.0) / total_score
            for sid, info in topk_info.items()
        }


class FSMSizer(bt.Sizer):
    """Regime-aware, conviction-weighted, volatility-risk-parity sizer for FSMPredictor outputs."""

    def __init__(
        self,
        stake: float = 0.90,
        target_expected_ret: float = 0.003,
        max_single_weight: float = 0.30,
        vol_parity: bool = True,
        **kwargs,
    ):
        self.stake = float(stake)
        self.target_expected_ret = max(1e-6, float(target_expected_ret))
        self.max_single_weight = float(max_single_weight)
        self.vol_parity = bool(vol_parity)
        self.default_vol_scale = max(1e-4, float(kwargs.get("default_vol_scale", 0.02)))
        self.min_vol_scale = max(1e-4, float(kwargs.get("min_vol_scale", 0.005)))

    def _resolve_vol_scale(self, info: Dict[str, Any]) -> float:
        if info.get("sid_scale") and float(info["sid_scale"]) > 0:
            return max(self.min_vol_scale, float(info["sid_scale"]))
        close = float(info.get("close") or 0.0)
        atr = float(info.get("atr") or 0.0)
        if close > 0 and atr > 0:
            return max(self.min_vol_scale, atr / close)
        return self.default_vol_scale

    def _getsizing(self, topk_info: Dict[bytes, Any], snapshot: SnapshotBody, isbuy: bool):
        if not isbuy:
            return _sell_all_positions(snapshot)
        if not topk_info:
            return {}

        raw_weights: Dict[bytes, float] = {}
        macro_exposures: List[float] = []

        for sid, info in topk_info.items():
            if not isinstance(info, dict):
                info = {"score": float(info) if isinstance(info, (int, float)) else 0.0}

            score = float(info.get("fsm_score", info.get("score", 0.0)) or 0.0)
            exp_ret = info.get("expected_return")
            if exp_ret is not None:
                exp_val = float(exp_ret)
                macro_exposures.append(0.0 if exp_val <= 0.0 else min(1.0, exp_val / self.target_expected_ret))
            else:
                macro_exposures.append(1.0 if score > 0 else 0.0)

            if score <= 0.0:
                raw_weights[sid] = 0.0
            elif self.vol_parity:
                raw_weights[sid] = score / self._resolve_vol_scale(info)
            else:
                raw_weights[sid] = score

        total_raw = sum(raw_weights.values())
        if total_raw <= 0.0:
            return {sid: 0.0 for sid in topk_info}

        avg_exposure = float(sum(macro_exposures) / len(macro_exposures)) if macro_exposures else 0.0
        effective_budget = self.stake * avg_exposure
        if effective_budget <= 0.0:
            return {sid: 0.0 for sid in topk_info}

        return {
            sid: min(self.max_single_weight, effective_budget * (w / total_raw)) if w > 0 else 0.0
            for sid, w in raw_weights.items()
        }
