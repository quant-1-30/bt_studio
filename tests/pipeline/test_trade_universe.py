#! /usr/bin/env python3

from bt_studio.trade import current_universe, make_pnc


def test_current_universe_reuses_offline_monthly_pit(monkeypatch):
    def fake_macro(common_config):
        return {"universe_sids": {202401: [b"600000", b"000001"], 202402: [b"300750"]}}

    monkeypatch.setattr("bt_studio.tune.node_prepare_macro", fake_macro)
    assert current_universe({}, 202401) == [b"600000", b"000001"]
    assert current_universe({}, 202402) == [b"300750"]


def test_make_pnc_builds_backtest_equivalent(monkeypatch):
    class StubSizer:
        pass

    captured = {}

    def fake_pnc(sizer, **kw):
        captured.update(kw)
        return object()

    monkeypatch.setattr("bt_core.pnc.Pnc", fake_pnc)

    make_pnc({"days_held": 2, "dd": 0.2, "stop_loss": -0.15, "max_positions": 8}, StubSizer())
    assert captured == {
        "days_held": 2, "dd": 0.2, "stop_loss": -0.15,
        "max_positions": 8, "exit_on_risk": True,
    }
