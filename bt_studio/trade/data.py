#! /usr/bin/env python3

from __future__ import annotations

import datetime
from typing import Any, Dict, List

import polars as pl
from bt_protocol._protocol import AccountBody, PositionBody, SnapshotBody

from bt_studio.utils.sid import sid_to_bytes, sid_to_str


def snapshot_to_bars(snapshot: Dict[str, Any], day) -> pl.DataFrame:
    """Convert ``TopkQuoteCache.snapshot_bars()`` columnar dict to a long bars frame.

    Parameters
    ----------
    snapshot:
        dict with ``sid`` (list[bytes]) and ``bar_idx/open/high/low/close/
        volume/amount`` (``array`` of int/float), the exact shape
        ``TopkQuoteCache.snapshot_bars()`` returns.
    day:
        ``datetime.date`` or ``int`` YYYYMMDD trading day.

    Returns a frame ``day(pl.Date), sid, bar_idx + BASE_MARKET_COLS``.
    """
    if not isinstance(day, (datetime.date, datetime.datetime)):
        day = datetime.datetime.strptime(str(int(day)), "%Y%m%d").date()
    n = len(snapshot["sid"])
    return pl.DataFrame({
        "day": pl.Series("day", [day] * n, dtype=pl.Date),
        "sid": snapshot["sid"],
        "bar_idx": list(snapshot["bar_idx"]),
        "open": list(snapshot["open"]),
        "high": list(snapshot["high"]),
        "low": list(snapshot["low"]),
        "close": list(snapshot["close"]),
        "volume": list(snapshot["volume"]),
        "amount": list(snapshot["amount"]),
    })


def daily_hfq_scales(factors: Dict[Any, Any], asof: int) -> Dict[str, float]:
    """Cumulative backward-adjustment scale per sid, matching ``bt_core`` ``apply_factor``.

    Parameters
    ----------
    factors:
        ``{sid: FactorResult}`` — the raw output of ``fetch_hfq_factors_safe`` /
        ``mdapi.get_factor``. Only ``FactorResult.raw_factors`` (per-event ratio
        ``{ex_date: r}``) is consumed; ``adj_factors`` (already-cumulative) is
        deliberately NOT used, to stay bit-comparable with ``bt_core``.
    asof:
        ``int`` YYYYMMDD cutoff (ex-dates ``<= asof`` count).

    Returns ``{ticker(str): scale}`` where ``scale = prod(1/r)`` — exactly the
    scale ``TopkQuoteCache`` multiplies prices by (and divides volume by).
    """
    out: Dict[str, float] = {}
    for sid, factor_obj in factors.items():
        raw = getattr(factor_obj, "raw_factors", None)
        if raw is None:
            raise TypeError(
                f"{sid!r}: expected FactorResult with 'raw_factors', got {type(factor_obj).__name__}"
            )
        scale = 1.0
        for ex_date, r in raw.items():
            if int(ex_date) <= asof:
                scale /= float(r)
        ticker = sid_to_str(sid)
        out[ticker] = scale
    return out


def build_snapshot(
    asset: dict,
    positions: List[dict],
    created_dt: Dict[str, int],
) -> SnapshotBody:
    """Build a Pnc-ready snapshot from XTP query results.

    Parameters
    ----------
    asset:
        XTP ``query_asset`` frame (keys ``buying_power``, ``total_asset``).
    positions:
        XTP ``query_position`` frames (keys ``ticker``, ``total_qty``,
        ``sellable_qty``, ``avg_price``, ``unrealized_pnl``).
    created_dt:
        ``{ticker: unix_ts}`` entry timestamps (XTP does not expose them);
        required for every held position — a missing entry makes Pnc's
        ``days_held`` huge and liquidates the position immediately.

    ``pnl_ratio`` follows bt_core ``Position._update_pnl_ratio``:
    ``(unrealized_pnl + realized_pnl) / (size * cost_basis)``, else 0.0.
    """
    pos_list: List[PositionBody] = []
    total_mv = 0.0
    total_pnl = 0.0

    for p in positions:
        ticker = p.get("ticker") or ""
        sid = sid_to_bytes(ticker)
        size = int(p.get("total_qty") or 0)
        available = int(p.get("sellable_qty") or 0)
        cost_basis = float(p.get("avg_price") or 0.0)
        pnl = float(p.get("unrealized_pnl") or 0.0)
        if size > 0 and ticker not in created_dt:
            raise ValueError(
                f"missing created_dt for held position {ticker!r} (size={size}); "
                f"Pnc days_held would be wrong and force immediate liquidation"
            )
        total_cost = size * cost_basis
        total_mv += total_cost + pnl
        total_pnl += pnl
        pos_list.append(PositionBody(
            sid=sid,
            datetime=0,
            size=size,
            available=available,
            cost_basis=cost_basis,
            pnl=pnl,
            created_dt=int(created_dt.get(ticker, 0) or 0),
            experiment_id=b"",
            pnl_ratio=(pnl / total_cost) if total_cost > 0 else 0.0,
            realized_pnl=0.0,
        ))

    cash = float(asset.get("buying_power") or 0.0)
    acc = AccountBody(
        datetime=0,
        portfolio_value=cash + total_mv,
        cash=cash,
        pnl=total_pnl,
        leverage=0.0,
        margin=0.0,
        experiment_id=b"",
    )
    return SnapshotBody(account=acc, positions=pos_list)
