#! /usr/bin/env python3

import datetime
import json

import polars as pl
import pytest

from bt_studio.trade.drift import DriftMonitor, breaches, drift_alerts, observed_median, signal_stats

DAY = datetime.date(2026, 11, 2)
DAY_INT = 20261102
CKPT = {"train_end_month": 202610, "feature_col": "f_abc"}


def _cross_section(n_trigger=10, score=1.0, ret=0.004, day=DAY):
    """One day's ``score_live``-shaped DTW triggers (``macro_state`` is a per-day value)."""
    return pl.DataFrame({
        "day": [day] * n_trigger,
        "sid": [f"{k:06d}" for k in range(n_trigger)],
        "distance": [0.5] * n_trigger,
        "macro_state": [1] * n_trigger,
        "expected_return": [ret] * n_trigger,
        "fsm_score": [score] * n_trigger,
    })


def _write_baseline(score_dir, counts):
    """Offline ``scores_{model_id}.parquet``: one cross-section per day with the given trigger counts."""
    score_dir.mkdir(parents=True)
    if not counts:
        empty = pl.DataFrame({
            "day": pl.Series([], dtype=pl.Date),
            "sid": pl.Series([], dtype=pl.Utf8),
            "distance": pl.Series([], dtype=pl.Float64),
            "macro_state": pl.Series([], dtype=pl.Int32),
            "expected_return": pl.Series([], dtype=pl.Float64),
            "fsm_score": pl.Series([], dtype=pl.Float64),
        })
        empty.write_parquet(score_dir / "scores_202610.parquet")
        return
    frames = [
        _cross_section(n_trigger=n).with_columns(pl.col("day") + pl.duration(days=i))
        for i, n in enumerate(counts)
    ]
    pl.concat(frames).write_parquet(score_dir / "scores_202610.parquet")


def _cc(tmp_path):
    return {"tune_score_dir": str(tmp_path / "scores"), "tune_collapse_dir": str(tmp_path / "collapse")}


def _monitor(tmp_path, counts=None, rel=None):
    cc = _cc(tmp_path)
    _write_baseline(tmp_path / "scores", counts if counts is not None else [10] * 25)
    if rel is not None:
        cc["drift_rel"] = rel
    return DriftMonitor(cc, CKPT, topk=5)


def _nov(d):
    """11 月 d 日 → ``(date, YYYYMMDD)``。"""
    dt = datetime.date(2026, 11, d)
    return dt, int(dt.strftime("%Y%m%d"))


# ---------------------------------------------------------------- signal_stats
def test_signal_stats_counts_triggers_and_ranks_topk_by_score():
    df = pl.DataFrame({
        "day": [DAY] * 4,
        "sid": ["a", "b", "c", "d"],
        "distance": [0.5] * 4,
        "macro_state": [2] * 4,
        "expected_return": [0.001, 0.003, 0.009, 0.005],
        "fsm_score": [1.0, -2.0, 4.0, 3.0],  # b 负分: 不可交易
    })
    row = signal_stats(df.lazy(), topk=2).row(0, named=True)
    assert row["n_trigger"] == 4  # 含负分触发
    assert row["macro_state"] == 2
    assert row["topk_score"] == pytest.approx(3.5)  # top-2 正分: (4.0 + 3.0) / 2
    assert row["topk_ret"] == pytest.approx(0.007)  # 按分数 (非行序) 取 expected_return: (0.009 + 0.005) / 2


def test_signal_stats_null_conviction_when_regime_untradeable():
    row = signal_stats(_cross_section(score=-1.0).lazy(), topk=5).row(0, named=True)
    assert row["n_trigger"] == 10 and row["topk_score"] is None and row["topk_ret"] is None


# ---------------------------------------------------------------- median / breaches / alerts
def test_observed_median_over_all_days():
    stats = pl.DataFrame({
        "day": [DAY, DAY + datetime.timedelta(days=1), DAY + datetime.timedelta(days=2)],
        "n_trigger": [10, 30, 20], "topk_score": [1.0, 1.0, 1.0], "topk_ret": [0.004, 0.004, 0.004],
    })
    assert observed_median(stats) == {"n_trigger": 20.0, "topk_score": 1.0, "topk_ret": 0.004}


def test_breaches_flags_relative_median_deviation():
    med = {"n_trigger": 20.0, "topk_score": None, "topk_ret": 0.004}
    assert breaches({"n_trigger": 25, "topk_score": 1.0, "topk_ret": 0.004}, med, 0.5) == []
    assert breaches({"n_trigger": 31, "topk_score": None, "topk_ret": 0.001}, med, 0.5) == ["n_trigger", "topk_ret"]
    # 中位数为 0 或 null 的指标不比较
    assert breaches({"n_trigger": 5}, {"n_trigger": 0.0}, 0.5) == []


def test_drift_alerts_shape():
    record = {"day": DAY_INT, "drifted": ["n_trigger"], "n_trigger": 50}
    [a] = drift_alerts(record, {"n_trigger": 10.0}, 202610)
    assert a["code"] == "model_drift_n_trigger" and a["level"] == "warn"
    assert "deviates from OOS median 10" in a["message"] and "+400.0%" in a["message"]
    assert a["meta"] == {"model_id": 202610, "day": DAY_INT, "metric": "n_trigger",
                         "value": 50, "median": 10.0}


# ---------------------------------------------------------------- DriftMonitor
def test_monitor_healthy_day_records_and_persists(tmp_path):
    mon = _monitor(tmp_path)
    assert mon.med == {"n_trigger": 10.0, "topk_score": 1.0, "topk_ret": 0.004}

    assert mon.check(_cross_section(), DAY_INT) == []

    record = mon.history[-1]
    assert (record["day"], record["n_trigger"], record["macro_state"]) == (DAY_INT, 10, 1)
    assert record["topk_score"] == pytest.approx(1.0) and record["topk_ret"] == pytest.approx(0.004)
    assert record["drifted"] == []
    saved = json.loads((tmp_path / "collapse" / "drift_202610_f_abc.json").read_text())
    assert saved == {"model_id": 202610, "median": mon.med, "history": mon.history}


def test_monitor_breach_alerts_warn(tmp_path):
    mon = _monitor(tmp_path)
    [a] = mon.check(_cross_section(n_trigger=50), DAY_INT)  # 触发数偏离中位数 400%
    assert a["code"] == "model_drift_n_trigger" and a["level"] == "warn"
    assert a["meta"]["value"] == 50


def test_monitor_same_day_refire_is_idempotent(tmp_path):
    mon = _monitor(tmp_path)
    mon.check(_cross_section(n_trigger=50), DAY_INT)
    mon.check(_cross_section(n_trigger=60), DAY_INT)  # 重启后同日重触发 → 原地替换
    assert len(mon.history) == 1 and mon.history[0]["n_trigger"] == 60


def test_monitor_resumes_history_from_disk(tmp_path):
    cc = _cc(tmp_path)
    _write_baseline(tmp_path / "scores", [10] * 25)
    DriftMonitor(cc, CKPT, 5).check(_cross_section(n_trigger=50), DAY_INT)
    mon = DriftMonitor(cc, CKPT, 5)  # 新进程读回 history
    assert [r["day"] for r in mon.history] == [DAY_INT]


def test_monitor_history_appends_across_days(tmp_path):
    mon = _monitor(tmp_path)
    for d in range(2, 7):
        day, day_int = _nov(d)
        mon.check(_cross_section(day=day), day_int)
    assert [r["day"] for r in mon.history] == [20261102, 20261103, 20261104, 20261105, 20261106]


def test_monitor_scores_only_the_requested_day(tmp_path):
    mon = _monitor(tmp_path)
    other = _cross_section(n_trigger=80, day=DAY + datetime.timedelta(days=1))
    assert mon.check(pl.concat([_cross_section(), other]), 20261102) == []
    assert mon.check(pl.concat([_cross_section(), other]), 20261103)[0]["meta"]["value"] == 80


def test_rel_invalid_env_falls_back_to_default(tmp_path, monkeypatch):
    """回归: 非法 BT_DRIFT_REL 不得令 DriftMonitor 构造崩溃, 回落 config 默认。"""
    monkeypatch.setenv("BT_DRIFT_REL", "abc")
    mon = _monitor(tmp_path)
    assert mon.rel == 0.5  # PRODUCTION_COMMON_PARAMS 默认


def test_rel_overridable_via_env(tmp_path, monkeypatch):
    cc = _cc(tmp_path)
    _write_baseline(tmp_path / "scores", range(10, 35))  # 触发数中位数 22
    low = _cross_section(n_trigger=8)

    assert DriftMonitor(cc, CKPT, 5).check(low, DAY_INT)[0]["meta"]["value"] == 8  # |8-22|/22=0.64 > 0.5 → 越界

    monkeypatch.setenv("BT_DRIFT_REL", "0.7")
    assert DriftMonitor(cc, CKPT, 5).check(low, DAY_INT) == []  # 0.64 < 0.7 → 不越界


@pytest.mark.parametrize("setup", ["no_dirs", "no_scores", "empty_scores", "corrupt_scores", "no_model_id"])
def test_monitor_disabled_without_usable_baseline(tmp_path, setup):
    cc, ckpt = _cc(tmp_path), dict(CKPT)
    if setup == "no_dirs":
        cc = {}
    elif setup == "empty_scores":
        _write_baseline(tmp_path / "scores", [])
    elif setup == "corrupt_scores":
        _write_baseline(tmp_path / "scores", [10])
        (tmp_path / "scores" / "scores_202610.parquet").write_bytes(b"not a parquet")
    elif setup == "no_model_id":
        _write_baseline(tmp_path / "scores", [10] * 25)
        ckpt.pop("train_end_month")

    mon = DriftMonitor(cc, ckpt, 5)

    assert mon.med is None and mon._path is None
    assert mon.check(_cross_section(n_trigger=50), DAY_INT) == []


def test_monitor_failure_is_isolated_and_empty_cross_section_alerts(tmp_path):
    mon = _monitor(tmp_path)

    assert mon.check(object(), DAY_INT) == []  # 哨兵自身出错 → 不抛, 不阻塞计划
    assert mon.history == []
    [a] = mon.check(pl.DataFrame(), DAY_INT)  # predictor 无触发时返回无列空帧
    assert a["code"] == "model_drift_n_trigger" and a["meta"]["value"] == 0
