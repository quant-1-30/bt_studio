import numpy as np
import polars as pl
import pytest
from datetime import date, timedelta

from bt_studio.pipeline.features import build_ofi
from bt_studio.pipeline.preprocess import build_fsm_panel
from bt_studio.pipeline.patterns import prepare_curves
from bt_studio.pipeline.patterns.fsm import evaluate_and_build_fsm
from bt_studio.pipeline.inference import FSMPredictor
from bt_studio.pipeline.metrics import find_pareto_front, select_best_model_from_pareto
from bt_studio.utils.common import calculate_decay_weights


def _trading_days(start: date, n: int):
    days = []
    d = start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(1)
    return days


def _make_minute_bars(n_days: int = 2, sids: list[str] = None, seed: int = 0):
    if sids is None:
        sids = ["000001"]
    rng = np.random.default_rng(seed)
    rows = []
    for d in _trading_days(date(2020, 1, 1), n_days):
        for sid in sids:
            close = 10.0
            for m in range(240):
                close += rng.normal() * 0.01
                high = close + abs(rng.normal() * 0.01)
                low = close - abs(rng.normal() * 0.01)
                amount = abs(rng.normal()) * 1e6
                rows.append(
                    {
                        "day": d.strftime("%Y%m%d"),
                        "sid": sid,
                        "minute_idx": m,
                        "open": close,
                        "close": close,
                        "high": high,
                        "low": low,
                        "amount": amount,
                        "volume": amount / 10,
                    }
                )
    df = pl.DataFrame(rows)
    return df.with_columns(pl.col("day").str.to_date("%Y%m%d"))


def _make_daily(n_days: int = 2, sids: list[str] = None, seed: int = 1):
    if sids is None:
        sids = ["000001"]
    rng = np.random.default_rng(seed)
    rows = []
    for d in _trading_days(date(2020, 1, 1), n_days):
        for sid in sids:
            rows.append({"day": d.strftime("%Y%m%d"), "sid": sid, "close": 10.0 + rng.normal() * 0.1})
    df = pl.DataFrame(rows)
    return df.with_columns(pl.col("day").str.to_date("%Y%m%d"))


@pytest.fixture
def common_config():
    return {
        "exclude_bars": 10,
        "ranking_window": 5,
        "ranking_ratio": 0.25,
        "decay": 1.0,
        "stats_windows": [1, 2, 3],
        "alternative": "greater",
        "win_rate": 0.5,
        "dtw_window_frac": 0.1,
    }


def test_calculate_decay_weights():
    w = calculate_decay_weights([1, 2, 3], half_life=1.0)
    assert set(w.keys()) == {1, 2, 3}
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert w[1] > w[2] > w[3]


def test_build_ofi():
    raw = _make_minute_bars(n_days=2, sids=["000001", "000002"])
    feat = build_ofi(raw.lazy(), {}).collect()
    assert "ofi_ratio" in feat.columns
    assert "bar_idx" in feat.columns
    # 2 days * 2 sids * 240 minutes
    assert feat.height == 2 * 2 * 240
    # ofi_ratio [-1, 1]
    ratios = feat["ofi_ratio"].to_numpy()
    assert np.all(np.isfinite(ratios))
    assert ratios.min() >= -1.0 - 1e-6
    assert ratios.max() <= 1.0 + 1e-6


def test_build_fsm_panel(common_config):
    raw = _make_minute_bars(n_days=5, sids=["000001", "000002"])
    daily = _make_daily(n_days=5, sids=["000001", "000002"])
    feat = build_ofi(raw.lazy(), common_config)
    tune = {"downsample": 2, "cross_days": 1}
    panel = build_fsm_panel(feat, daily.lazy(), tune, common_config, is_train=True).collect()
    assert "lag_0" in panel.columns
    assert "fwd_ret_1" in panel.columns
    assert "fwd_ret_2" in panel.columns
    assert "fwd_ret_3" in panel.columns
    # 240//2 = 120 bars per day
    curve_len = panel["curve_len"].unique().to_list()
    assert curve_len == [120]


def test_build_fsm_panel_oss_left_join(common_config):
    """OOS left join"""
    raw = _make_minute_bars(n_days=3, sids=["000001"])
    daily = _make_daily(n_days=2, sids=["000001"])  # 少一天
    feat = build_ofi(raw.lazy(), common_config)
    tune = {"downsample": 1, "cross_days": 1}
    panel = build_fsm_panel(feat, daily.lazy(), tune, common_config, is_train=False).collect()
    assert panel.height > 0


def test_prepare_curves(common_config):
    raw = _make_minute_bars(n_days=3, sids=["000001"])
    daily = _make_daily(n_days=3, sids=["000001"])
    feat = build_ofi(raw.lazy(), common_config)
    tune = {"downsample": 2, "cross_days": 1}
    panel = build_fsm_panel(feat, daily.lazy(), tune, common_config, is_train=True).collect()
    curves = prepare_curves(panel, tune, common_config)
    bars_per_day = 240 // tune["downsample"]
    assert curves.shape == (panel.height, bars_per_day)
    # exclude_bars//downsample
    exclude = common_config["exclude_bars"] // tune["downsample"]
    assert np.all(np.isnan(curves[:, -exclude:]))


def test_evaluate_and_build_fsm_smoke(common_config):
    """构造一个能被 motif 匹配且触发组收益更高的面板，验证 evaluate 可跑出 success。"""
    L = 30
    m = 5
    motif = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    days = _trading_days(date(2020, 1, 1), 40)
    sids = ["000001", "000002", "000003", "000004", "000005"]
    rows = []
    rng = np.random.default_rng(7)
    for d in days:
        for j, sid in enumerate(sids):
            curve = rng.normal(size=L) * 0.3
            if j == 0:
                curve[10 : 10 + m] = motif
                fr1, fr2, fr3 = 0.02, 0.03, 0.04
            else:
                fr1, fr2, fr3 = rng.normal() * 0.005, rng.normal() * 0.005, rng.normal() * 0.005
            rows.append(
                {
                    "day": d,
                    "sid": sid,
                    "lag_0": curve.tolist(),
                    "fwd_ret_1": fr1,
                    "fwd_ret_2": fr2,
                    "fwd_ret_3": fr3,
                }
            )
    panel = pl.DataFrame(rows)
    common_config = {**common_config, "exclude_bars": 5}
    tune = {"downsample": 1, "cross_days": 1, "motif_minutes": m, "threshold_r": 0.95}
    tune["m"] = int(tune["motif_minutes"] // tune["downsample"])
    tune["threshold_d"] = float(np.sqrt(2 * tune["m"] * (1.0 - tune["threshold_r"])))
    curves = prepare_curves(panel, tune, common_config)
    result = evaluate_and_build_fsm(panel, curves, motif, tune, common_config)
    assert result["status"] == "success", result.get("reason")
    assert "fsm_matrix" in result
    assert "bin_weights" in result["fsm_matrix"]
    assert result["trigger_count"] >= 10


def test_fsmpredictor_smoke(common_config):
    """FSMPredictor"""
    L = 30
    m = 5
    motif = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    tune = {"downsample": 1, "cross_days": 1, "m": m, "threshold_d": 0.1}
    fsm_matrix = {
        "P(T1|Macro)": [[0.5, 0.3, 0.1, 0.1], [0.1, 0.3, 0.5, 0.1], [0.1, 0.1, 0.3, 0.5]],
        "P(T2|T1)": [[0.4, 0.3, 0.2, 0.1], [0.1, 0.4, 0.3, 0.2], [0.2, 0.1, 0.4, 0.3], [0.1, 0.2, 0.3, 0.4]],
        "P(T3|T2)": [[0.4, 0.3, 0.2, 0.1], [0.1, 0.4, 0.3, 0.2], [0.2, 0.1, 0.4, 0.3], [0.1, 0.2, 0.3, 0.4]],
        "bin_weights": {
            1: [0.01, 0.005, -0.005, -0.01],
            2: [0.015, 0.007, -0.007, -0.015],
            3: [0.02, 0.01, -0.01, -0.02],
        },
    }
    ckpt = {"config": tune, "motif": motif, "fsm_matrix": fsm_matrix}

    days = _trading_days(date(2020, 1, 1), 10)
    rows = []
    for d in days:
        curve = np.random.randn(L)
        curve[:m] = motif
        rows.append({"day": d, "sid": "000001", "lag_0": curve.tolist()})
    panel = pl.DataFrame(rows)

    predictor = FSMPredictor(ckpt, common_config)
    scored = predictor.predict(panel.lazy())
    assert scored.height > 0
    assert "fsm_score" in scored.columns
    assert scored["fsm_score"].max() > 0


def test_find_pareto_front():
    df = pl.DataFrame(
        {
            "config/downsample": [1, 1, 1, 1],
            "config/cross_days": [2, 1, 1, 2],
            "config/motif_minutes": [50, 30, 20, 30],
            "config/threshold_r": [0.8, 0.8, 0.8, 0.7],
            "metrics_score": [2.0, 1.5, 1.0, 0.5],
        }
    )
    common = {"dtw_window_frac": 0.1}
    front = find_pareto_front(df, common)
    assert front.height == 3
    scores = set(front["metrics_score"].to_list())
    assert scores == {2.0, 1.5, 1.0}


def test_select_best_model_from_pareto():
    df = pl.DataFrame(
        {
            "config/downsample": [1, 1],
            "config/cross_days": [1, 2],
            "config/motif_minutes": [30, 50],
            "config/threshold_r": [0.8, 0.7],
            "metrics_score": [-10.0, -5.0],
        }
    )
    common = {"dtw_window_frac": 0.1}
    # complexity row0 = 1*30*0.1*0.2 = 0.6; row1 = 2*50*0.1*0.3 = 3.0
    front = find_pareto_front(df, common)
    assert front.height == 2
    # efficiency = score * complexity: -10*0.6 = -6.0, -5*3.0 = -15.0 -> 选 -10.0
    best = select_best_model_from_pareto(front)
    assert best is not None
    assert best["metrics_score"] == -10.0
