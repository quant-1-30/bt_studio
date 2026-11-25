#! /usr/bin/env python3

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import polars as pl

from bt_studio.constant import BSE_PREFIXES
from bt_studio.utils.sid import sid_to_bytes, sid_to_str

log = logging.getLogger(__name__)

# xtp_api_data_type.h
XTP_MKT_SZ_A = 1
XTP_MKT_SH_A = 2
XTP_SIDE_BUY = 1
XTP_SIDE_SELL = 2
XTP_PRICE_LIMIT = 1          
XTP_PRICE_BEST5_OR_CANCEL = 4  


def market_id(ticker: str) -> int:
    """``XTP_MARKET_TYPE`` for an order (SH_A=2, SZ_A=1); BSE unsupported."""
    if ticker.startswith(BSE_PREFIXES):
        raise ValueError(f"BSE ticker {ticker} not supported by XTP order API")
    return XTP_MKT_SH_A if ticker.startswith(("6", "5", "9")) else XTP_MKT_SZ_A


def plan_to_order(plan: Dict[str, Any], side: int, cash: float, price: float) -> dict:
    """Convert one ``TraderPlan`` (from ``Pnc.generate_plan``) to an XTP ``insertOrder`` dict.

    ``side`` is ``XTP_SIDE_BUY``/``XTP_SIDE_SELL``; for buys ``size==0`` so the
    quantity is derived from ``weight * cash / price`` (rounded to 100-share lots),
    for sells it is ``plan["size"]``.

    Price type: buy = market-like (``XTP_PRICE_BEST5_OR_CANCEL``, immediate fill);
    sell = limit (``XTP_PRICE_LIMIT``) so the order stays in the book until close —
    no lower-limit detection / pending re-issue needed, the exchange keeps it active
    and fills when 跌停 opens (or leaves it unfilled at close).

    The returned dict is sent through the ZMQ chain:
    ``ZmqClient.call("trader.insert_order", {"req": order})`` → bt_playgroud
    ``ZmqServer`` → ``TraderService.insert_order`` → ``vnxtptrader.insertOrder``.
    """
    ticker = sid_to_str(plan["sid"])
    if side == XTP_SIDE_BUY:
        qty = int(plan["weight"] * cash / price) if price > 0 else 0
        qty = (qty // 100) * 100  # A股整手
    else:
        qty = int(plan["size"] or 0)
    return {
        "ticker": ticker,
        "market": market_id(ticker),
        "price": price,
        "quantity": qty,
        "price_type": XTP_PRICE_BEST5_OR_CANCEL if side == XTP_SIDE_BUY else XTP_PRICE_LIMIT,
        "side": side,
    }


def reconcile_fills(pnc, trades: List[dict]) -> None:
    """Aggregate XTP sell trade reports (``side==2``) per sid and feed ``Pnc.on_execute``.

    Only sells decrement ``pending_sells``; buy fills are ignored here.
    """
    executed: Dict[bytes, int] = {}
    for t in trades:
        if int(t.get("side") or 0) != XTP_SIDE_SELL:
            continue
        ticker = t.get("ticker") or ""
        sid = sid_to_bytes(ticker)
        qty = int(t.get("quantity") or 0)
        executed[sid] = executed.get(sid, 0) + qty
    pnc.on_execute(executed)


def fill_frame(trades: List[dict], day: int) -> pl.DataFrame:
    """XTP trade reports → actual-fill frame ``{sid, isbuy, day, price}`` for ``calibrate()``.
    """
    rows = []
    for t in trades:
        ticker = t.get("ticker") or ""
        if not ticker:
            continue
        rows.append({
            "sid": sid_to_bytes(ticker),
            "isbuy": int(t.get("side") or 0) == XTP_SIDE_BUY,
            "day": int(day),
            "price": float(t.get("price") or 0.0),
        })
    if not rows:
        return pl.DataFrame(
            schema={"sid": pl.Binary, "isbuy": pl.Boolean, "day": pl.Int64, "price": pl.Float64}
        )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Execution-mode decoupling: daily_plan is mode-agnostic; only the executor
# differs (manual = emit for human, auto = submit via ZmqClient.insert_order).
# ---------------------------------------------------------------------------

class ExecMode:
    MANUAL = "manual"  
    AUTO = "auto"      


def _materialize(plans: dict, cash: float, prices: Dict[str, float]) -> List[dict]:
    """``TraderPlans`` → ordered list of XTP ``insertOrder`` dicts (shared by all executors)."""
    orders: List[dict] = []
    for plan in plans.get(b"buy", []):
        orders.append(plan_to_order(plan, XTP_SIDE_BUY, cash, prices.get(sid_to_str(plan["sid"]), 0.0)))
    for plan in plans.get(b"sell", []):
        orders.append(plan_to_order(plan, XTP_SIDE_SELL, cash, prices.get(sid_to_str(plan["sid"]), 0.0)))
    return orders


class ManualExecutor:
    """Emit orders for human review; no automatic submission (手动下单模式)."""

    def __init__(self, sink: Optional[Callable[[str, Any], None]] = None):
        self.sink = sink or log.info

    async def submit(self, plans: dict, cash: float, prices: Dict[str, float]) -> None:
        for order in _materialize(plans, cash, prices):
            self.sink("MANUAL ORDER: %s", order)


class AutoExecutor:
    """Submit orders via ``ZmqClient.call("trader.insert_order")`` """

    def __init__(self, client):
        self.client = client

    async def submit(self, plans: dict, cash: float, prices: Dict[str, float]) -> None:
        for order in _materialize(plans, cash, prices):
            await self.client.call("trader.insert_order", {"req": order})


def make_executor(mode: str, client=None, auto_allowed: bool = False):
    """Factory: pick the plan executor by mode. ``daily_plan`` stays mode-agnostic.

    Auto-trading is deferred until the XTP scale threshold is met: ``mode == auto``
    additionally requires ``auto_allowed=True`` (``common_config["auto_allowed"]`` /
    ``BT_LIVE_AUTO_ALLOWED=1``), so the current manual-only rollout cannot flip
    into auto submission by setting the mode alone.
    """
    if mode == ExecMode.AUTO:
        if not auto_allowed:
            raise RuntimeError(
                "auto exec deferred: scale threshold not met "
                "(set common_config['auto_allowed']=True to enable)"
            )
        return AutoExecutor(client)
    if mode == ExecMode.MANUAL:
        return ManualExecutor()
    raise ValueError(f"unknown exec mode: {mode!r}")
