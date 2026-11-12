#!/usr/bin/env python3
"""raw_pit 月包缓存行为锁定: 写时验证/增量合并/读侧过滤/原子性/元数据。

mock 边界 = mdapi.subscribe (记录请求) + _collect_stream_sync (按 day_counts
产对齐后 DataFrame dict); prepare_tick 内部的订阅/验证/合并为被测实现。
"""

import datetime as dt
import json
import os
import sys

import polars as pl
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import bt_studio.pipeline.preprocess.prepare as pr


def _mk_tick_stream(sid: bytes, days: int) -> pl.DataFrame:
    """原始 tick 流形状 (含 tick 秒戳, UTC) — 走生产 align_skeleton 真实对齐。

    09:31..11:30 + 13:01..15:00 (END-label), UTC = 北京-8h。
    """
    import datetime as _d
    rows = []
    base = _d.datetime(2024, 1, 1, 0, 0, tzinfo=_d.timezone.utc)
    for d in range(days):
        bj = _d.date(2024, 1, d + 1)
        # 北京 09:31..11:30 (120 分钟) + 13:01..15:00 (120 分钟) → UTC 各 -8h
        for minutes_bj in (list(range(9 * 60 + 31, 11 * 60 + 31))
                           + list(range(13 * 60 + 1, 15 * 60 + 1))):
            ts = int(_d.datetime(bj.year, bj.month, bj.day,
                                 minutes_bj // 60, minutes_bj % 60,
                                 tzinfo=_d.timezone.utc).timestamp()) - 8 * 3600
            px = 10.0 + minutes_bj * 1e-4
            rows.append((ts, px, px * 1.01, px * 0.99, px, 100.0, 1000.0))
    return pl.DataFrame(rows, schema={
        "tick": pl.Int64, "open": pl.Float64, "high": pl.Float64,
        "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64,
        "amount": pl.Float64}, orient="row")


def _mk_df(sid: bytes, days: int):
    rows = []
    for d in range(days):
        day = dt.date(2024, 1, d + 1)
        for b in range(240):
            rows.append((sid, day, b, 10.0, 10.1, 9.9, 10.0, 100.0, 1000.0))
    return pl.DataFrame(rows, schema=pr._BARS_SCHEMA, orient="row")


class _FakeMdapi:
    """subscribe 只记录请求 sid 集; 数据由 fake _collect_stream_sync 产出。"""

    def __init__(self):
        self.last_sids: list = []

    def subscribe(self, body, topic):
        self.last_sids = list(body.sid)

        class _Obs:
            pass
        return _Obs()


class _ctx:
    def __init__(self, api):
        self._api = api

    def __enter__(self):
        return self._api

    def __exit__(self, *a):
        return False


@pytest.fixture()
def pit_env(tmp_path, monkeypatch):
    root = str(tmp_path / "bars")
    monkeypatch.setattr(pr, "RAW_PIT_DIR", root)
    fetch_log: list = []
    api = _FakeMdapi()
    monkeypatch.setattr(pr, "external_mdapi_context", lambda: _ctx(api))

    def make_fetch(day_counts: dict):
        def fake_collect(obs, timeout=600.0):
            fetch_log.append([s.decode() for s in api.last_sids])
            return {sid: _mk_tick_stream(sid, day_counts.get(sid, 0))
                    for sid in api.last_sids if day_counts.get(sid, 0)}
        monkeypatch.setattr(pr, "_collect_stream_sync", fake_collect)

    return root, fetch_log, make_fetch


def _run(monkeypatch, **kw):
    return pr.prepare_tick(**kw)


def test_first_fetch_writes_pack_with_meta(pit_env):
    root, log, mk = pit_env
    mk({b"600001": 3, b"600002": 2})
    lf = _run(None, start_date=20240101, end_date=20240131,
              sids=[b"600001", b"600002"])
    assert lf.select(pl.len()).collect().item() == 5 * 240
    meta = json.load(open(os.path.join(root, "ym=202401.meta.json")))
    assert set(meta["covered_sids"]) == {"600001", "600002"}


def test_second_call_incremental_no_refetch(pit_env):
    root, log, mk = pit_env
    mk({b"600001": 3})
    for _ in range(2):
        _run(None, start_date=20240101, end_date=20240131,
             sids=[b"600001"])
    assert len(log) == 1  # 第二次全命中, 零拉取


def test_incremental_merge_new_sid_appends(pit_env):
    root, log, mk = pit_env
    mk({b"600001": 3})
    _run(None, start_date=20240101, end_date=20240131,
         sids=[b"600001"])
    mk({b"600002": 2})
    lf = _run(None, start_date=20240101, end_date=20240131,
              sids=[b"600001", b"600002"])
    # 增量: 只拉 600002, 旧段保留
    assert [e for e in log[-1]] == ["600002"]
    assert lf.filter(pl.col("sid") == b"600001").select(pl.len()).collect().item() == 3 * 240
    assert lf.filter(pl.col("sid") == b"600002").select(pl.len()).collect().item() == 2 * 240


def test_poisoned_sid_not_cached(pit_env):
    root, log, mk = pit_env
    # bad: 源返回 0 天 (空/无有效行情) → 不入 covered, 记入 empty_sids 负缓存防重复 RPC; good 正常
    mk({b"bad": 0, b"good": 3})
    _run(None, start_date=20240101, end_date=20240131,
         sids=[b"bad", b"good"])
    meta = json.load(open(os.path.join(root, "ym=202401.meta.json")))
    assert "bad" not in meta["covered_sids"] and "good" in meta["covered_sids"]
    assert "bad" in meta["empty_sids"]
    # 第二次再次请求 [bad, good] 时, bad 命中 empty_sids 负缓存, 零 RPC 重拉!
    _run(None, start_date=20240101, end_date=20240131,
         sids=[b"bad", b"good"])
    assert len(log) == 1


def test_readside_filters_request_range(pit_env):
    root, log, mk = pit_env
    mk({b"600001": 3})
    lf = _run(None, start_date=20240102, end_date=20240102,
              sids=[b"600001"])
    days = lf.select("day").unique().sort("day").collect()["day"].to_list()
    assert days == [dt.date(2024, 1, 2)]


def test_orphan_pack_without_meta_refetches(pit_env):
    """meta 丢失 (孤儿包) → covered=空 → 全量重拉自愈。"""
    root, log, mk = pit_env
    mk({b"600001": 3})
    _run(None, start_date=20240101, end_date=20240131,
         sids=[b"600001"])
    os.remove(os.path.join(root, "ym=202401.meta.json"))
    mk({b"600001": 3})
    _run(None, start_date=20240101, end_date=20240131,
         sids=[b"600001"])
    assert len(log) == 2
