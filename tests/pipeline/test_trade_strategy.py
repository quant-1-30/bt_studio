#! /usr/bin/env python3

from bt_studio.trade import FsmStrategy, exit_stage_from_minutes


def test_exit_stage_from_minutes_mapping():
    offsets = (5, 15, 30)
    assert exit_stage_from_minutes(offsets, 0) == 0   
    assert exit_stage_from_minutes(offsets, 5) == 0   # 09:35 → stage 0
    assert exit_stage_from_minutes(offsets, 15) == 1  # 09:45 → stage 1
    assert exit_stage_from_minutes(offsets, 30) == 2  # 10:00 → stage 2
    assert exit_stage_from_minutes(offsets, 100) == 2  


def test_fsm_strategy_set_exit_stage_delegates(monkeypatch):
    import bt_studio.trade.strategy as S

    class StubSizer:
        def __init__(self):
            self.stage = -1

        def set_exit_stage(self, stage):
            self.stage = stage

    sizer = StubSizer()
    monkeypatch.setattr(S, "make_sizer", lambda cc: sizer)
    monkeypatch.setattr(S, "make_pnc", lambda cc, s: object())

    strat = S.FsmStrategy({"feature_col": "x"}, {"T1_rets": {"exit_5m": 5, "exit_15m": 15, "exit_30m": 30}})
    strat.set_exit_stage(1)
    assert sizer.stage == 1
    assert strat.exit_offsets == (5, 15, 30)


def test_fsm_strategy_plan_from_scored_wiring(monkeypatch):
    import bt_studio.trade.strategy as S

    captured = {}

    class StubPnc:
        def generate_plan(self, day, topk, snapshot):
            captured["day"] = day
            captured["topk"] = topk
            captured["snapshot"] = snapshot
            return {"sell": [], "buy": []}

    monkeypatch.setattr(S, "make_sizer", lambda cc: object())
    monkeypatch.setattr(S, "make_pnc", lambda cc, s: StubPnc())
    monkeypatch.setattr(S, "build_topk", lambda scored, quotes, n: {b"600000": {"fsm_score": 1.0}})

    strat = S.FsmStrategy(
        {"feature_col": "x"},
        {"candidate_multiplier": 1.5, "max_positions": 10, "T1_rets": {"exit_5m": 5}},
    )
    snap = object()
    plans = strat.plan_from_scored("SCORED", {"600000": {}}, snap, 20261120)

    assert plans == {"sell": [], "buy": []}
    assert captured["day"] == 20261120
    assert captured["topk"] == {b"600000": {"fsm_score": 1.0}}
    assert captured["snapshot"] is snap


def test_candidate_multiplier_sizes_topk_pool(monkeypatch):
    import bt_studio.trade.strategy as S

    monkeypatch.setattr(S, "make_sizer", lambda cc: object())
    monkeypatch.setattr(S, "make_pnc", lambda cc, s: object())
    monkeypatch.setattr(S, "score_live", lambda ck, bars, cc: "SCORED")

    captured = {}

    def fake_topk(scored, quotes, n):
        captured["n"] = n
        return {}

    monkeypatch.setattr(S, "build_topk", fake_topk)

    strat = S.FsmStrategy(
        {"feature_col": "x"},
        {"max_positions": 10, "candidate_multiplier": 1.5, "T1_rets": {"exit_5m": 5}},
    )
    strat.topk_from_scored("SCORED", {})
    assert captured["n"] == 15  # 1.5 × max_positions
