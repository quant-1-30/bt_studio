#! /usr/bin/env python3

import datetime

import numpy as np
import polars as pl

from bt_studio.trade import score_live


def _bars(days: list[datetime.date], sids: list[str]) -> pl.DataFrame:
    parts = []
    for day in days:
        for sid in sids:
            close = np.linspace(10.0, 11.0, 240, dtype=np.float64)
            parts.append(pl.DataFrame({
                "day": [day] * 240,
                "sid": [sid] * 240,
                "bar_idx": np.arange(240, dtype=np.int32),
                "open": close, "high": close * 1.001,
                "low": close * 0.999, "close": close,
                "volume": np.full(240, 1000.0), "amount": close * 1000.0,
            }))
    return pl.concat(parts)


def test_score_live_builds_lag0_panel_and_scores(monkeypatch):
    from bt_studio.trade import scorer as S

    ckpt = {
        "feature_col": "close",
        "ast_recipe": None,
        "config": {"downsample": 1},
        "motif": np.zeros((1, 10)),
        "fsm_matrix": {},  # stubbed predictor never reads it
    }
    common = {"ranking_window": 2, "exclude_bars": 10}

    seen = {}

    class StubPredictor:
        def __init__(self, ck, cc):
            seen["ckpt"] = ck
            seen["cc"] = cc

        def predict(self, panel_lf):
            panel = panel_lf.collect()
            seen["panel_cols"] = panel.columns
            seen["panel_rows"] = panel.height
            return pl.DataFrame({
                "day": [datetime.date(2024, 1, 3)],
                "sid": ["600000"],
                "distance": [1.0],
                "macro_state": [1],
                "expected_return": [0.004],
                "fsm_score": [1.5],
            })

    monkeypatch.setattr(S, "FSMPredictor", StubPredictor)

    bars = _bars(
        [datetime.date(2024, 1, 2), datetime.date(2024, 1, 3)],
        ["600000", "688001"],
    )
    out = score_live(ckpt, bars, common)

    # feature_col is threaded from ckpt into common_config
    assert seen["cc"]["feature_col"] == "close"
    # panel handed to the predictor carries a fixed-length lag_0 curve per (day, sid)
    assert "lag_0" in seen["panel_cols"]
    assert seen["panel_rows"] == 4  # 2 days x 2 sids
    # predictor output is returned verbatim with the scores schema
    assert list(out.columns) == ["day", "sid", "distance", "macro_state", "expected_return", "fsm_score"]
    assert out["fsm_score"][0] == 1.5
