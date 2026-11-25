#! /usr/bin/env python3

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from bt_studio.constant import BEIJING_UTC_OFFSET_SEC, SESSION_OPEN_MINUTE
from bt_studio.utils.math_func import entry_bar_clock
from bt_studio.utils.sid import sid_to_str

from .data import build_snapshot, snapshot_to_bars
from .drift import DriftMonitor
from .execution import ExecMode, make_executor
from .strategy import FsmStrategy, exit_stage_from_minutes

log = logging.getLogger(__name__)

_POLL_SECONDS = 1.0

Trigger = Tuple[datetime.time, str]


def _clock(total_min: int) -> datetime.time:
    return datetime.time(total_min // 60, total_min % 60)


def intraday_triggers(common_config: dict) -> List[Trigger]:
    """Ordered ``(clock_time, kind)`` triggers within one trading day."""
    entry_hh, entry_mm = entry_bar_clock(common_config)
    exit_minutes = sorted(int(m) for m in common_config["T1_rets"].values())
    triggers: List[Trigger] = [
        (_clock(SESSION_OPEN_MINUTE + m), f"exit:{m}") for m in exit_minutes
    ]
    triggers.append((datetime.time(entry_hh, entry_mm), "entry"))
    return sorted(triggers)


def next_trigger(triggers: List[Trigger], now: datetime.datetime) -> Optional[Tuple[datetime.datetime, str]]:
    """Return the next ``(datetime, kind)`` strictly after ``now`` today, or ``None``."""
    for t, kind in triggers:
        dt = datetime.datetime.combine(now.date(), t)
        if now < dt:
            return dt, kind
    return None


async def run_schedule(
    triggers: List[Trigger],
    is_trading_day: Callable[[datetime.date], bool],
    on_trigger: Callable[[datetime.date, str], None],
) -> None:
    """Await each trigger on trading days, firing ``on_trigger(date, kind)``.

    ``is_trading_day`` supplies the exchange calendar (weekday + holiday);
    ``on_trigger`` receives the trading date and the trigger kind.
    """
    while True:
        now = datetime.datetime.now()
        if not is_trading_day(now.date()):
            await asyncio.sleep(_POLL_SECONDS)
            continue
        target = next_trigger(triggers, now)
        if target is None:
            await asyncio.sleep(_POLL_SECONDS)
            continue
        dt, kind = target
        await asyncio.sleep((dt - now).total_seconds())
        on_trigger(dt.date(), kind)


class TradeEngine:
    """Stateful bridge between the live runtime and the shared ``FsmStrategy``.
        Trade engine: the execution/plumbing layer around the shared ``FsmStrategy``.
        ``FsmStrategy`` holds the strategy brain (feature + sizer + pnc); this engine
        only wires the live runtime — XTP query, quote cache, schedule, executor — and
        delegates planning/risk/reconcile to the strategy. bt_core stays pure backtest.
    """

    def __init__(self, model_ckpt: Dict[str, Any], common_config: dict, client: Any):
        self.model_ckpt = model_ckpt
        self.cc = dict(common_config)
        self.client = client
        self.strategy = FsmStrategy(model_ckpt, self.cc)
        self.drift = DriftMonitor(self.cc, model_ckpt, self.strategy.pool_size)
        self.entry_idx = self.strategy.entry_idx
        self._created_dt: Dict[str, int] = {}  # ticker -> entry unix ts (XTP has no entry date)
        self._fills: List[dict] = []  # raw XTP trade reports, exported for calibration
        self._cache = None
        # Execution mode is decoupled from planning: daily_plan always runs, only
        # the executor changes (manual=emit for human / auto=submit via ZmqClient).
        # Auto stays gated until the XTP scale threshold is met (auto_allowed).
        mode = self.cc.get("exec_mode") or os.environ.get("BT_LIVE_EXEC_MODE", ExecMode.MANUAL)
        auto_allowed = bool(self.cc.get("auto_allowed", False)) or os.environ.get("BT_LIVE_AUTO_ALLOWED", "") == "1"
        self.executor = make_executor(mode, client, auto_allowed=auto_allowed)

    # ------------------------------------------------------------------
    # delegate strategy brain
    # ------------------------------------------------------------------
    def daily_plan(self, bars, quotes, snapshot, current_day: int):
        """manual/auto ---> Pnc"""
        scored = self.strategy.score(bars)
        self.drift.check(scored, current_day)
        return self.strategy.plan_from_scored(scored, quotes, snapshot, current_day)

    def risk_exits(self, snapshot, current_day: int):
        return self.strategy.risk_exits(snapshot, current_day)

    def reconcile(self, trades: List[dict]) -> None:
        self.strategy.reconcile(trades)

    def export_fills(self, day: Optional[int] = None):
        """Actual fills accumulated from trade reports → ``{sid, isbuy, day, price}`` frame.
            used for ``bt_core ... finance.calibration``
        """
        from .execution import fill_frame

        if day is None:
            day = int(datetime.datetime.fromtimestamp(
                time.time() + BEIJING_UTC_OFFSET_SEC, tz=datetime.timezone.utc
            ).strftime("%Y%m%d"))
        return fill_frame(self._fills, day)

    # ------------------------------------------------------------------
    # small shared helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _day_int(day) -> int:
        return int(day.strftime("%Y%m%d"))

    @staticmethod
    def _prices(quotes: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        return {t: q["entry_close"] for t, q in quotes.items()}

    async def _snap_tracked(self):
        """Query the snapshot and fold its equity into the strategy's drawdown tracker."""
        snap = await self.snapshot()
        self.strategy.update_drawdown(float(snap.account.portfolio_value))
        return snap

    # ------------------------------------------------------------------
    # XTP query → snapshot (G6)
    # ------------------------------------------------------------------
    async def _query_asset(self) -> dict:
        frame = await self.client.call("trader.query_asset", {})
        return (frame or {}).get("data", {}) if isinstance(frame, dict) else {}

    async def _query_positions(self) -> List[dict]:
        positions: List[dict] = []
        async for frame in self.client.subscribe("trader.query_position", {"ticker": ""}):
            if isinstance(frame, dict) and frame.get("data"):
                positions.append(frame["data"])
        return positions

    async def snapshot(self):
        asset = await self._query_asset()
        positions = await self._query_positions()
        # prune the entry-timestamp book to tickers still held (size > 0)
        held = {p.get("ticker") for p in positions if int(p.get("total_qty") or 0) > 0}
        self._created_dt = {t: ts for t, ts in self._created_dt.items() if t in held}
        return build_snapshot(asset, positions, self._created_dt)

    # ------------------------------------------------------------------
    # cache → bars / quotes / prices
    # ------------------------------------------------------------------
    def _bars_quotes(self, cache, day):
        bars = snapshot_to_bars(cache.snapshot_bars(up_to_bar_idx=self.entry_idx), day)
        quotes: Dict[str, Dict[str, float]] = {}
        entry = bars.filter(bars["bar_idx"] == self.entry_idx)
        for row in entry.iter_rows(named=True):
            sid = row["sid"]
            ticker = sid_to_str(sid)
            idx = cache.sid_to_idx.get(ticker, -1)
            preclose = float(cache.preclose[idx]) if idx >= 0 else 0.0
            quotes[ticker] = {"preclose": preclose, "entry_close": float(row["close"])}
        return bars, quotes

    # ------------------------------------------------------------------
    # order execution (mode-decoupled: manual emit vs auto submit)
    # ------------------------------------------------------------------
    async def _execute(self, plans: dict, cash: float, prices: Dict[str, float]) -> None:
        await self.executor.submit(plans, cash, prices)

    # ------------------------------------------------------------------
    # triggers
    # ------------------------------------------------------------------
    async def _on_entry(self, cache, day) -> None:
        bars, quotes = self._bars_quotes(cache, day)
        snap = await self._snap_tracked()
        plans = self.daily_plan(bars, quotes, snap, self._day_int(day))
        await self._execute(plans, float(snap.account.cash), self._prices(quotes))

    async def _on_exit(self, cache, day, kind) -> None:
        # kind = "exit:5" / "exit:15" / "exit:30" → stage(same with on_risk)
        minute = int(kind.split(":", 1)[1])
        self.strategy.set_exit_stage(exit_stage_from_minutes(self.strategy.exit_offsets, minute))
        snap = await self._snap_tracked()
        sells = self.risk_exits(snap, self._day_int(day))
        # forward-filled close at entry_idx is the current limit/protection price
        quotes = self._bars_quotes(cache, day)[1]
        await self._execute({b"sell": sells}, float(snap.account.cash), self._prices(quotes))

    def _trigger(self, cache):
        async def handle(day, kind):
            if kind == "entry":
                await self._on_entry(cache, day)
            else:
                await self._on_exit(cache, day, kind)

        return handle

    # ------------------------------------------------------------------
    # fill reconciliation + position book (G7)
    # ------------------------------------------------------------------
    async def _watch_trades(self) -> None:
        from .execution import XTP_SIDE_BUY

        async for frame in self.client.subscribe("trader.subscribe_events", {}, idle_timeout=0.0):
            if not isinstance(frame, dict) or frame.get("event") != "onTradeEvent":
                continue
            data = frame.get("data") or {}
            ticker = data.get("ticker") or ""
            side = int(data.get("side") or 0)
            # `time.time()` is a UTC unix ts; Pnc's ts2intdt adds +28800 → Beijing day,
            # so it is machine-timezone independent.
            if ticker and side == XTP_SIDE_BUY:
                self._created_dt.setdefault(ticker, int(time.time()))
            self._fills.append(data)
            self.reconcile([data])

    # ------------------------------------------------------------------
    # mark-to-market drawdown (sample equity finer than 4 triggers/day)
    # ------------------------------------------------------------------
    async def _mark_to_market(self) -> None:
        interval = float(self.cc.get("mark_interval", 60.0))
        while True:
            try:
                await self._snap_tracked()
            except Exception as e:
                log.warning("live mark-to-market failed (drawdown tracking paused): %s", e)
            await asyncio.sleep(interval)

    # ------------------------------------------------------------------
    # run loop
    # ------------------------------------------------------------------
    async def run(
        self,
        sids: Sequence[str | bytes],
        hfq_scales: Dict[str, float],
        trading_days: List[int],
        is_trading_day: Callable[[Any], bool],
    ) -> None:
        from bt_studio.plugins.xtp import TopkQuoteCache

        cache = TopkQuoteCache(self.client, sids, hfq_scales)
        await cache.start()
        self._cache = cache
        self.strategy.pnc._start([int(d) for d in trading_days])
        watch = asyncio.create_task(self._watch_trades())
        mtm = asyncio.create_task(self._mark_to_market())
        try:
            await run_schedule(intraday_triggers(self.cc), is_trading_day, self._trigger(cache))
        finally:
            for task in (watch, mtm):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            await cache.stop()
