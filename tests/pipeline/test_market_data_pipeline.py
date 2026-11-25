#! /usr/bin/env python3

from __future__ import annotations

import asyncio
import datetime
import socket

import polars as pl
import pytest
import zmq
import zmq.asyncio

from bt_studio.plugins.zmq_client import ZmqClient
from bt_studio.plugins.zmq_client.codec import EOF, decode_payload, encode_response
from bt_studio.plugins.xtp import TopkQuoteCache
from bt_studio.trade import current_universe
from bt_studio.trade.data import snapshot_to_bars
from bt_studio.utils.math_func import entry_bar_idx

YM = 202611
SIDS = [f"60{i:04d}" for i in range(10)]  
PRECLOSE = 10.0
HFQ_SCALES = {"600000": 2.0}  


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _session_minutes() -> list[int]:
    """240 09:31..11:30 + 13:01..15:00"""
    mins = list(range(9 * 60 + 31, 11 * 60 + 31))
    mins += list(range(13 * 60 + 1, 15 * 60 + 1))
    assert len(mins) == 240
    return mins


def _build_ticks(sids: list[str], day: int) -> list[dict]:
    ticks = []
    for b, m in enumerate(_session_minutes()):
        hh, mm = divmod(m, 60)
        dt = day * 1_000_000_000 + hh * 10_000_000 + mm * 100_000
        for sid in sids:
            ticks.append({
                "ticker": sid,
                "last_price": PRECLOSE + (b + 1) * 0.01,
                "pre_close_price": PRECLOSE,
                "upper_limit_price": PRECLOSE * 1.1,
                "lower_limit_price": PRECLOSE * 0.9,
                "qty": float((b + 1) * 1000),
                "turnover": float((b + 1) * 10000),
                "data_time": dt,
            })
    return ticks


async def _serve(port: int, ticks: list[dict]) -> None:
    """XTP ROUTER subscribe_events / subscribe_market_data / ping"""
    ctx = zmq.asyncio.Context()
    router = ctx.socket(zmq.ROUTER)
    router.bind(f"tcp://127.0.0.1:{port}")
    try:
        while True:
            ident, rid, payload = await router.recv_multipart()
            method = decode_payload(payload)["method"]
            if method == "quote.subscribe_events":
                await router.send_multipart([ident, rid, encode_response({"subscribed": True})])
                for t in ticks:
                    await router.send_multipart(
                        [ident, rid, encode_response({"event": "onDepthMarketData", "data": t})]
                    )
                await router.send_multipart([ident, rid, EOF])
            elif method == "quote.subscribe_market_data":
                await router.send_multipart([ident, rid, encode_response({"ret": 0})])
                await router.send_multipart([ident, rid, EOF])
            elif method == "ping":
                await router.send_multipart([ident, rid, encode_response({"pong": True})])
                await router.send_multipart([ident, rid, EOF])
    finally:
        router.close()
        ctx.term()


async def _run(tmp_path) -> None:
    ticks = _build_ticks(SIDS, day=20261120)
    n_ticks = len(ticks)  # 2400

    port = _free_port()
    srv = asyncio.create_task(_serve(port, ticks))

    client = ZmqClient(f"tcp://127.0.0.1:{port}", timeout=10.0, rcvhwm=65536)
    cache = TopkQuoteCache(client, SIDS, hfq_scales=HFQ_SCALES)
    try:
        ack = await cache.start()
        assert ack["result"]["ret"] == 0, "quote.subscribe_market_data 未应答"

        for _ in range(400):
            if cache.ticks_ingested >= n_ticks:
                break
            await asyncio.sleep(0.01)
        assert cache.ticks_ingested == n_ticks, f"丢 tick: {cache.ticks_ingested}/{n_ticks}"

        entry = entry_bar_idx()
        assert entry == 230
        assert len(cache.snapshot_bars(up_to_bar_idx=229)["sid"]) == len(SIDS) * 230
        snap = cache.snapshot_bars(up_to_bar_idx=entry)
        assert len(snap["sid"]) == len(SIDS) * (entry + 1), "快照 bar 数不符"

        bars = snapshot_to_bars(snap, datetime.date(2026, 11, 20))
        assert bars.height == len(SIDS) * (entry + 1)
        assert set(bars.columns) == {"day", "sid", "bar_idx", "open", "high", "low", "close", "volume", "amount"}

        # 600000 (hfq=2.0) close = (preclose + (b+1)*0.01) * 2.0
        row = bars.filter((pl.col("sid") == b"600000") & (pl.col("bar_idx") == entry))
        assert abs(row["close"][0] - (PRECLOSE + (entry + 1) * 0.01) * 2.0) < 1e-9
        # (600001) close = preclose + (b+1)*0.01
        row2 = bars.filter((pl.col("sid") == b"600001") & (pl.col("bar_idx") == entry))
        assert abs(row2["close"][0] - (PRECLOSE + (entry + 1) * 0.01)) < 1e-9

        out = tmp_path / "market_snapshot.parquet"
        bars.write_parquet(out)
        back = pl.read_parquet(out)
        assert back.height == bars.height
        assert set(back.columns) == set(bars.columns)
        assert back.filter((pl.col("sid") == b"600000") & (pl.col("bar_idx") == entry))["close"][0] == pytest.approx(
            (PRECLOSE + (entry + 1) * 0.01) * 2.0
        )
    finally:
        await cache.stop()
        await client.close()
        srv.cancel()
        try:
            await srv
        except asyncio.CancelledError:
            pass


def test_market_data_pipeline_lossless_to_parquet(tmp_path, monkeypatch):
    """10% universe → zmq_client → XTP 行情 → parquet"""
    monkeypatch.setattr(
        "bt_studio.tune.node_prepare_macro",
        lambda cc: {"universe_sids": {YM: [s.encode() for s in SIDS]}},
    )
    sids = current_universe({"min_amount_rank": 0.90}, YM)
    assert sids == [s.encode() for s in SIDS]

    # 2. 全链路: zmq_client → TopkQuoteCache → snapshot → parquet
    asyncio.run(_run(tmp_path))
