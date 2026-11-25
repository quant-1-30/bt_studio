#! /usr/bin/env python3

import pytest

from bt_studio.trade import build_snapshot


def test_build_snapshot_maps_xtp_fields_and_pnl_ratio():
    asset = {"buying_power": 500000.0, "total_asset": 600000.0}
    positions = [{
        "ticker": "600000", "total_qty": 1000, "sellable_qty": 800,
        "avg_price": 10.0, "unrealized_pnl": 500.0,
    }]
    created = {"600000": 1704159060}

    snap = build_snapshot(asset, positions, created)

    acc = snap.account
    assert acc.cash == 500000.0
    assert acc.portfolio_value == 500000.0 + (1000 * 10.0 + 500.0)
    assert acc.pnl == 500.0

    pos = snap.positions[0]
    assert pos.sid == b"600000"
    assert pos.size == 1000 and pos.available == 800
    assert pos.cost_basis == 10.0 and pos.pnl == 500.0
    assert pos.created_dt == 1704159060
    assert abs(pos.pnl_ratio - 500.0 / (1000 * 10.0)) < 1e-12


def test_build_snapshot_zero_cost_guards_pnl_ratio():
    snap = build_snapshot(
        {"buying_power": 100.0},
        [{"ticker": "600000", "total_qty": 0, "sellable_qty": 0,
          "avg_price": 0.0, "unrealized_pnl": 0.0}],
        {},  # size==0 的仓位无需 created_dt
    )
    assert snap.positions[0].pnl_ratio == 0.0
    assert snap.account.portfolio_value == 100.0


def test_build_snapshot_raises_when_held_position_lacks_created_dt():
    # 缺失 created_dt 会让 Pnc days_held 巨大 -> 次日全清仓, 必须显式报错
    with pytest.raises(ValueError, match="missing created_dt"):
        build_snapshot(
            {"buying_power": 100.0},
            [{"ticker": "600000", "total_qty": 1000, "sellable_qty": 1000,
              "avg_price": 10.0, "unrealized_pnl": 0.0}],
            {},
        )
