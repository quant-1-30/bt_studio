#! /usr/bin/env python3

from __future__ import annotations

from typing import Any, Dict, List

from bt_studio.utils.math_func import entry_bar_idx

from .execution import reconcile_fills
from .scorer import build_topk, score_live


def current_universe(common_config: dict, ym: int) -> List[bytes]:
    """Monthly PIT Top-K sids for ``ym`` (YYYYMM), same source as ``node_prepare_macro``."""
    from bt_studio.tune import node_prepare_macro

    result = node_prepare_macro(common_config)
    return list(result.get("universe_sids", {}).get(ym, []))


def make_pnc(common_config: dict, sizer) -> "bt_core.pnc.Pnc":  # type: ignore[name-defined]
    """Build the backtest-equivalent ``Pnc`` from ``common_config`` (G4: reuse, not reimplement)."""
    from bt_core.pnc import Pnc

    return Pnc(
        sizer,
        days_held=int(common_config.get("days_held", 1)),
        dd=float(common_config.get("dd", 0.25)),
        stop_loss=float(common_config.get("stop_loss", -0.1)),
        max_positions=int(common_config.get("max_positions", 10)),
        exit_on_risk=True,
    )


def make_sizer(common_config: dict) -> "FSMSizer":  # type: ignore[name-defined]
    """Build the backtest-equivalent ``FSMSizer`` from ``common_config`` (stake/cap/exit_weights)."""
    from .sizers import FSMSizer, sizer_params

    return FSMSizer(**sizer_params(common_config))


def exit_stage_from_minutes(exit_offsets, minutes_from_open: int) -> int:
    """map ``minutes_from_open`` → tranche exit stage index (backtest & live share this).

    ``exit_offsets`` is the sorted T+1 exit horizons (e.g. 5/15/30 min); the stage
    is the index of the largest offset ``<= minutes_from_open``.
    """
    stage = 0
    for idx, m in enumerate(sorted(exit_offsets)):
        if minutes_from_open >= m:
            stage = idx
    return stage


class _Drawdown:
    """Peak-to-trough drawdown tracker (Pnc reads ``stats["drawdown"].maxdd``)."""

    def __init__(self):
        self._peak = 0.0
        self.maxdd = 0.0

    def update(self, equity: float) -> None:
        if equity > self._peak:
            self._peak = equity
        if self._peak > 0:
            dd = (self._peak - equity) / self._peak
            if dd > self.maxdd:
                self.maxdd = dd


class FsmStrategy:
    """FsmStrategy: bundle feature scoring + sizer + pnc into ONE strategy object.
    This is the shared "strategy brain" for backtest and live — both build the
    sizer/pnc from the same ``make_sizer``/``make_pnc`` and call the same
    ``plan_from_topk``/``risk_exits``/``reconcile``, so the backtest↔live drift is
    minimised to the data feed only. Live orchestration (score → drift check →
    plan) lives in ``TradeEngine.daily_plan``.
    """

    def __init__(self, model_ckpt: Dict[str, Any], common_config: dict):
        self.model_ckpt = model_ckpt
        self.cc = dict(common_config)
        self.sizer = make_sizer(self.cc)
        self.pnc = make_pnc(self.cc, self.sizer)
        self.entry_idx = entry_bar_idx(self.cc)
        self.exit_offsets = tuple(sorted(int(m) for m in self.cc["T1_rets"].values()))
        self.dd = _Drawdown()
        # 候选池 = candidate_multiplier × max_positions: 大于持仓数提供替补,
        # 涨停/流动性打不满时由次优候选补位 (收益↔风险 payoff 关键参数)。
        self.pool_size = int(self.cc.get("candidate_multiplier", 1.5) * self.cc.get("max_positions", 10))

    def set_exit_stage(self, stage: int) -> None:
        """Set the tranche exit stage on the sizer (tranche exit weights)."""
        self.sizer.set_exit_stage(int(stage))

    # -- feature (live real-time; offline feeds the same topk_info dict via ParquetPatch) --
    def score(self, bars):
        return score_live(self.model_ckpt, bars, self.cc)

    def topk_from_scored(self, scored, quotes) -> Dict[bytes, Dict[str, float]]:
        return build_topk(scored, quotes, self.pool_size)

    # -- plan (shared by backtest & live) --
    def plan_from_scored(self, scored, quotes, snapshot, current_day: int):
        """实时打分结果 → topk → Pnc.generate_plan (live 入口, 由 TradeEngine.daily_plan 调用)。"""
        return self.plan_from_topk(self.topk_from_scored(scored, quotes), snapshot, current_day)

    def plan_from_topk(self, topk_info, snapshot, current_day: int):
        """从已有 topk_info(离线 ParquetPatch)直接生成 plan (backtest 入口)。"""
        return self.pnc.generate_plan(current_day, topk_info, snapshot)

    def risk_exits(self, snapshot, current_day: int):
        return self.pnc.on_risk(snapshot, {"drawdown": self.dd}, current_day, filler=b"default")

    def reconcile(self, trades) -> None:
        reconcile_fills(self.pnc, trades)

    def update_drawdown(self, equity: float) -> None:
        self.dd.update(float(equity))
