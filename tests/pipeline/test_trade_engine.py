#! /usr/bin/env python3

import asyncio
import datetime
from array import array

import polars as pl
import pytest

from bt_studio.trade import (
    AutoExecutor,
    ExecMode,
    TradeEngine,
    ManualExecutor,
    XTP_SIDE_BUY,
    XTP_SIDE_SELL,
    build_topk,
    make_executor,
    make_sizer,
)


class _MockClient:
    def __init__(self, asset=None, positions=None, trades=()):
        self._asset = asset or {"buying_power": 100000.0}
        self._positions = positions or []
        self._trades = list(trades)
        self.insert_calls = []

    async def call(self, method, params):
        if method == "trader.query_asset":
            return {"event": "onQueryAsset", "data": self._asset, "last": True}
        if method == "trader.insert_order":
            self.insert_calls.append(params["req"])
            return {"order_xtp_id": 1}
        return None

    async def subscribe(self, method, params, idle_timeout=None):
        if method == "trader.query_position":
            for p in self._positions:
                yield {"event": "onQueryPosition", "data": p, "last": False}
            return
        if method == "trader.subscribe_events":
            for t in self._trades:
                yield t
            # never yields EOF; the watch task is cancelled by the caller


class _MockCache:
    def __init__(self, ticker="600000", preclose=10.0, close=10.5):
        self.sid_to_idx = {ticker: 0}
        self.preclose = array("d", [preclose])
        self._ticker = ticker
        self._close = close

    def snapshot_bars(self, up_to_bar_idx=239):
        cut = up_to_bar_idx + 1
        return {
            "sid": [self._ticker.encode()] * cut,
            "bar_idx": array("i", range(cut)),
            "open": array("d", [10.0] * cut),
            "high": array("d", [10.6] * cut),
            "low": array("d", [9.9] * cut),
            "close": array("d", [self._close] * cut),
            "volume": array("d", [1000.0] * cut),
            "amount": array("d", [10000.0] * cut),
        }


def _engine(client, **cc_overrides):
    cc = {
        "T1_rets": {"exit_5m": 5, "exit_15m": 15, "exit_30m": 30},
        "decay_minutes": 15,
        "stake": 0.9,
        "max_single_weight": 0.5,
        "topk": 5,
        "days_held": 1,
        "dd": 0.25,
        "max_positions": 10,
        "bars_per_day": 240,
        "exclude_bars": 10,
        "ranking_window": 5,
        **cc_overrides,
    }
    return TradeEngine({"feature_col": "close", "ast_recipe": None, "config": {"downsample": 1}}, cc, client)


def test_snapshot_queries_asset_and_positions():
    client = _MockClient(
        asset={"buying_power": 500000.0},
        positions=[{"ticker": "600000", "total_qty": 1000, "sellable_qty": 800,
                    "avg_price": 10.0, "unrealized_pnl": 500.0}],
    )
    eng = _engine(client)
    eng._created_dt["600000"] = 1704159060

    snap = asyncio.run(eng.snapshot())
    assert snap.account.cash == 500000.0
    assert snap.positions[0].sid == b"600000"
    assert snap.positions[0].size == 1000
    assert snap.positions[0].created_dt == 1704159060


def test_auto_executor_submits_orders():
    client = _MockClient()
    plans = {
        b"buy": [{"sid": b"600000", "weight": 0.5, "size": 0}],
        b"sell": [{"sid": b"300750", "weight": 1.0, "size": 300}],
    }
    asyncio.run(AutoExecutor(client).submit(plans, cash=1_000_000.0, prices={"600000": 10.0, "300750": 0.0}))

    assert len(client.insert_calls) == 2
    buy, sell = client.insert_calls
    assert buy["ticker"] == "600000" and buy["side"] == XTP_SIDE_BUY and buy["quantity"] == 50000
    assert sell["ticker"] == "300750" and sell["side"] == XTP_SIDE_SELL and sell["quantity"] == 300


def test_manual_executor_emits_but_does_not_submit():
    emitted = []

    class StubClient:
        async def call(self, method, params):
            raise AssertionError("manual mode must not submit")

    plans = {b"buy": [{"sid": b"600000", "weight": 0.5, "size": 0}]}
    asyncio.run(ManualExecutor(lambda fmt, *a: emitted.append(a)).submit(
        plans, cash=1_000_000.0, prices={"600000": 10.0},
    ))
    assert len(emitted) == 1
    order = emitted[0][0]
    assert order["ticker"] == "600000" and order["quantity"] == 50000


def test_make_executor_selects_by_mode():
    assert isinstance(make_executor(ExecMode.MANUAL), ManualExecutor)
    assert isinstance(make_executor(ExecMode.AUTO, client=object(), auto_allowed=True), AutoExecutor)
    # auto 规模阈值守卫: 未显式放行时 mode=auto 直接拒绝 (当前仅手动下单)
    with pytest.raises(RuntimeError, match="scale threshold"):
        make_executor(ExecMode.AUTO, client=object())
    with pytest.raises(ValueError, match="unknown exec mode"):
        make_executor("bogus")


def test_bars_quotes_extracts_preclose_and_entry_close():
    eng = _engine(_MockClient())
    cache = _MockCache(preclose=10.0, close=10.5)
    bars, quotes = eng._bars_quotes(cache, datetime.date(2026, 11, 20))

    assert bars.height == 231  # entry_idx=230 -> up_to_bar_idx=230 -> 231 bars
    assert quotes["600000"] == {"preclose": 10.0, "entry_close": 10.5}


def test_watch_trades_records_created_dt_and_reconciles(monkeypatch):
    client = _MockClient(trades=[
        {"event": "onTradeEvent", "data": {"ticker": "600000", "side": 2, "quantity": 200}},
    ])
    eng = _engine(client)

    reconciled = {}

    def fake_reconcile(trades):
        reconciled["trades"] = trades

    monkeypatch.setattr(eng, "reconcile", fake_reconcile)

    # run one iteration of the watch loop manually by driving the client stream
    async def drive():
        watch = asyncio.create_task(eng._watch_trades())
        await asyncio.sleep(0.05)
        watch.cancel()
        try:
            await watch
        except asyncio.CancelledError:
            pass

    asyncio.run(drive())
    assert reconciled["trades"] == [{"ticker": "600000", "side": 2, "quantity": 200}]


def test_snapshot_prunes_created_dt_for_closed_positions():
    client = _MockClient(
        asset={"buying_power": 100000.0},
        positions=[{"ticker": "600000", "total_qty": 1000, "sellable_qty": 1000,
                    "avg_price": 10.0, "unrealized_pnl": 0.0}],
    )
    eng = _engine(client)
    eng._created_dt = {"600000": 1704159060, "000001": 1704000000}  # 000001 已清仓

    asyncio.run(eng.snapshot())
    assert eng._created_dt == {"600000": 1704159060}  # 000001 被剪枝


def test_make_sizer_derives_exit_weights_from_t1_rets():
    cc = {
        "T1_rets": {"exit_5m": 5, "exit_15m": 15, "exit_30m": 30},
        "decay_minutes": 15,
        "stake": 0.9,
        "max_single_weight": 0.5,
    }
    sizer = make_sizer(cc)
    assert sizer.stake == 0.9
    assert sizer.max_single_weight == 0.5
    assert abs(sum(sizer.exit_weights) - 1.0) < 1e-9
    assert sizer.exit_weights[0] > sizer.exit_weights[1] > sizer.exit_weights[2]


def test_build_topk_filters_limit_up_and_topk():
    scored = pl.DataFrame({
        "day": [datetime.date(2026, 11, 20)] * 4,
        "sid": ["600000", "300750", "000001", "688111"],
        "distance": [1.0, 1.0, 1.0, 1.0],
        "macro_state": [1, 1, 1, 1],
        "expected_return": [0.01, 0.02, 0.03, 0.04],
        "fsm_score": [1.0, 2.0, 3.0, -1.0],  # 688111 负分应被过滤
    })
    quotes = {
        "600000": {"preclose": 10.0, "entry_close": 11.0},   # +10% 涨停 -> 排除
        "300750": {"preclose": 10.0, "entry_close": 10.5},
        "000001": {"preclose": 10.0, "entry_close": 10.2},
        "688111": {"preclose": 10.0, "entry_close": 10.1},
    }
    topk = build_topk(scored, quotes, topk=5)
    assert list(topk.keys()) == [b"000001", b"300750"]
    assert topk[b"000001"]["expected_return"] == 0.03


def test_build_topk_backfills_rejected_candidates():
    """被涨停拒绝的候选由次优候选补位, topk 池不缩水 (替补候选语义)。"""
    scored = pl.DataFrame({
        "day": [datetime.date(2026, 11, 20)] * 3,
        "sid": ["600000", "300750", "000001"],
        "distance": [1.0] * 3,
        "macro_state": [1] * 3,
        "expected_return": [0.01, 0.02, 0.03],
        "fsm_score": [3.0, 2.0, 1.0],  # 600000 分最高但涨停
    })
    quotes = {
        "600000": {"preclose": 10.0, "entry_close": 11.0},  # +10% 涨停 -> 拒绝
        "300750": {"preclose": 10.0, "entry_close": 10.5},
        "000001": {"preclose": 10.0, "entry_close": 10.2},
    }
    topk = build_topk(scored, quotes, topk=2)
    # 600000 被拒后 000001 顶上, 池仍满 topk=2 (旧 .head(topk) 先截断会只剩 1 个)
    assert list(topk.keys()) == [b"300750", b"000001"]


def test_trade_engine_daily_plan_scores_once_then_drift_then_plan():
    """打分只做一次: 同一份 scored 先喂漂移哨兵 (只观察), 再生成计划。"""
    eng = _engine(_MockClient())
    calls = []
    eng.strategy.score = lambda bars: calls.append("score") or "SCORED"
    eng.drift.check = lambda scored, day: calls.append(("drift", scored, day)) or []
    eng.strategy.plan_from_scored = (
        lambda scored, quotes, snap, day: calls.append(("plan", scored, quotes, snap, day)) or {"sell": [], "buy": []}
    )
    quotes, snap = {"600000": {}}, object()

    plans = eng.daily_plan("BARS", quotes, snap, 20261124)

    assert plans == {"sell": [], "buy": []}
    assert calls == ["score", ("drift", "SCORED", 20261124), ("plan", "SCORED", quotes, snap, 20261124)]
