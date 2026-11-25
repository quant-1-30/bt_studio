#! /usr/bin/env python3

import pytest

from bt_studio.trade import (
    XTP_SIDE_BUY,
    XTP_SIDE_SELL,
    market_id,
    plan_to_order,
    reconcile_fills,
)
from bt_studio.trade.execution import (
    XTP_MKT_SH_A,
    XTP_MKT_SZ_A,
    XTP_PRICE_BEST5_OR_CANCEL,
    fill_frame,
)


def test_market_id_orders_are_inverse_of_quote_exchange():
    # 订单 market 用 XTP_MARKET_TYPE: 沪A=2, 深A=1 (与 quote exchange_id 相反)
    assert market_id("600519") == XTP_MKT_SH_A
    assert market_id("000001") == XTP_MKT_SZ_A
    assert market_id("300750") == XTP_MKT_SZ_A
    with pytest.raises(ValueError, match="BSE"):
        market_id("430047")


def test_plan_to_order_buy_derives_quantity_from_weight():
    order = plan_to_order(
        {"sid": b"600000", "weight": 0.5, "size": 0},
        XTP_SIDE_BUY, cash=1_000_000.0, price=10.0,
    )
    assert order["ticker"] == "600000"
    assert order["market"] == XTP_MKT_SH_A
    assert order["side"] == XTP_SIDE_BUY
    assert order["price_type"] == XTP_PRICE_BEST5_OR_CANCEL  # market-like, matches PseudoFiller
    assert order["quantity"] == 50000  # 0.5 * 1e6 / 10, already a 100-lot


def test_plan_to_order_sell_uses_plan_size():
    order = plan_to_order(
        {"sid": b"300750", "weight": 1.0, "size": 300},
        XTP_SIDE_SELL, cash=0.0, price=0.0,
    )
    assert order["market"] == XTP_MKT_SZ_A
    assert order["side"] == XTP_SIDE_SELL
    assert order["quantity"] == 300


def test_reconcile_fills_only_sells_feed_pnc():
    calls = {}

    class StubPnc:
        def on_execute(self, mtrades):
            calls["mtrades"] = mtrades

    trades = [
        {"ticker": "600000", "side": 2, "quantity": 200},  # sell
        {"ticker": "600000", "side": 2, "quantity": 100},  # sell (aggregate)
        {"ticker": "000001", "side": 1, "quantity": 500},  # buy -> ignored
    ]
    reconcile_fills(StubPnc(), trades)
    assert calls["mtrades"] == {b"600000": 300}


def test_fill_frame_maps_trades_to_calibrate_schema():
    """实盘成交 → {sid, isbuy, day, price} 校准帧 (与回测成交同 schema)。"""
    trades = [
        {"ticker": "600000", "side": XTP_SIDE_BUY, "quantity": 100, "price": 10.1},
        {"ticker": "600000", "side": XTP_SIDE_SELL, "quantity": 100, "price": 10.2},
        {"ticker": "300750", "side": XTP_SIDE_BUY, "quantity": 200, "price": 20.0},
        {"ticker": "", "side": XTP_SIDE_BUY, "price": 1.0},  # 无 ticker → 跳过
    ]
    df = fill_frame(trades, 20261120)
    assert list(df.columns) == ["sid", "isbuy", "day", "price"]
    assert df["sid"].to_list() == [b"600000", b"600000", b"300750"]
    assert df["isbuy"].to_list() == [True, False, True]
    assert df["price"].to_list() == [10.1, 10.2, 20.0]
    assert (df["day"] == 20261120).all()


def test_fill_frame_empty_keeps_schema():
    df = fill_frame([], 20261120)
    assert df.height == 0
    assert list(df.columns) == ["sid", "isbuy", "day", "price"]
