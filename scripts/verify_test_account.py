#! /usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import datetime
import socket

import zmq
import zmq.asyncio

from bt_studio.plugins.zmq_client import ZmqClient
from bt_studio.plugins.zmq_client.codec import EOF, decode_payload, encode_response
from bt_studio.plugins.xtp import TopkQuoteCache
from bt_studio.trade import ExecMode, make_executor, snapshot_to_bars

DEFAULT_TICKERS = ["600519", "000001", "300750", "601318", "600000"]


async def _run(endpoint: str, tickers: list[str], seconds: int, out_parquet: str) -> int:
    client = ZmqClient(endpoint, timeout=15.0, rcvhwm=65536)
    cache = TopkQuoteCache(client, tickers, hfq_scales={})
    try:
        await client.connect()

        # 1. ping —— ZMQ 连通 + XTP 登录状态
        res = await client.call("ping", {})
        info = (res or {}).get("result") or {}
        print(f"[1] ping: {info}")
        if not info.get("pong"):
            print("✗ ping 失败 —— 检查容器 run_server.py 是否启动、端口是否映射")
            return 1
        if not info.get("quote_started"):
            print("⚠ quote_started=False —— 行情未登录, 检查 XTP_QUOTE_USER/PASSWORD 环境变量")

        # 2. 订阅行情 + 摄取 N 秒
        ack = await cache.start()
        print(f"[2] 订阅行情: {(ack or {}).get('result')}")
        print(f"    等待 {seconds}s 摄取实时行情 ...")
        await asyncio.sleep(seconds)
        print(f"    已摄取 {cache.ticks_ingested} ticks / {cache.frames_received} 帧")

        # 3. 快照 → parquet (截至当前最新 bar)
        snap = cache.snapshot_bars(up_to_bar_idx=239)
        if not snap["sid"]:
            print("[3] ⚠ 快照为空 (可能非交易时段 / quote 未登录), 跳过落盘")
        else:
            bars = snapshot_to_bars(snap, datetime.date.today())
            bars.write_parquet(out_parquet)
            print(f"[3] 快照落盘 {out_parquet}: {bars.height} 行 x {bars.width} 列")

        # 4. manual executor: 合成买单物化并打印 (不下真实单)
        executor = make_executor(ExecMode.MANUAL)
        sample = {b"buy": [{"sid": tickers[0].encode(), "weight": 0.1, "size": 0}], b"sell": []}
        await executor.submit(sample, cash=1_000_000.0, prices={tickers[0]: 10.0})
        print("[4] manual executor 已打印 MANUAL ORDER (见上方日志)")

        return 0
    finally:
        await cache.stop()
        await client.close()


def _session_minutes() -> list[int]:
    """240 根 1 分钟 bar 的会话分钟 (09:31..11:30 + 13:01..15:00)。"""
    mins = list(range(9 * 60 + 31, 11 * 60 + 31))
    mins += list(range(13 * 60 + 1, 15 * 60 + 1))
    assert len(mins) == 240
    return mins


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _stress_tickers(n: int) -> list[str]:
    """生成 n 只真实形态 A 股代码 (沪 60xxxx + 深 00xxxx 各半)。"""
    half = n // 2
    sh = [f"60{i:04d}" for i in range(half)]
    sz = [f"00{i:04d}" for i in range(1, n - half + 1)]
    return sh + sz


def _build_stress_ticks(tickers: list[str], day: int) -> list[dict]:
    """每标的每 bar 1 笔 tick, close 随 bar 单调递增 (便于无损校验)。"""
    ticks = []
    for b, m in enumerate(_session_minutes()):
        hh, mm = divmod(m, 60)
        dt = day * 1_000_000_000 + hh * 10_000_000 + mm * 100_000
        px = 10.0 + (b + 1) * 0.01
        for sid in tickers:
            ticks.append({
                "ticker": sid,
                "last_price": px,
                "pre_close_price": 10.0,
                "upper_limit_price": 11.0,
                "lower_limit_price": 9.0,
                "qty": float((b + 1) * 1000),
                "turnover": float((b + 1) * 10000),
                "data_time": dt,
            })
    return ticks


async def _serve(port: int, ticks: list[dict]) -> None:
    """最小 mock 行情源 (ROUTER): 应答 ping / subscribe_events / subscribe_market_data。

    时序与真实 XTP 一致: subscribe_events 先回 ``subscribed``(不 EOF),
    等 subscribe_market_data 回 ack 后, 再在 events 流上推 tick + EOF。
    """
    ctx = zmq.asyncio.Context()
    router = ctx.socket(zmq.ROUTER)
    router.setsockopt(zmq.SNDHWM, 0)  # mock 无界发送, 避免 ROUTER 突发丢帧
    router.bind(f"tcp://127.0.0.1:{port}")
    ev_ident = ev_rid = None
    try:
        while True:
            ident, rid, payload = await router.recv_multipart()
            method = decode_payload(payload)["method"]
            if method == "ping":
                await router.send_multipart([ident, rid, encode_response({"pong": True, "trader_started": True, "quote_started": True})])
                await router.send_multipart([ident, rid, EOF])
            elif method == "quote.subscribe_events":
                # 只回 subscribed, 记住流句柄, 等 subscribe_market_data 后再推 tick
                await router.send_multipart([ident, rid, encode_response({"subscribed": True})])
                ev_ident, ev_rid = ident, rid
            elif method == "quote.subscribe_market_data":
                # 先回 ack (让 client.start() 立即返回, 不阻塞在整批 tick 上)
                await router.send_multipart([ident, rid, encode_response({"ret": 0})])
                await router.send_multipart([ident, rid, EOF])
                # 再在 events 流上推 tick + EOF
                if ev_ident is not None:
                    for t in ticks:
                        await router.send_multipart([ev_ident, ev_rid, encode_response({"event": "onDepthMarketData", "data": t})])
                    await router.send_multipart([ev_ident, ev_rid, EOF])
                    ev_ident = ev_rid = None
    finally:
        router.close()
        ctx.term()


async def _stress_run(n: int, out_parquet: str) -> int:
    """N 标的压测 (自包含 mock 行情源): 订阅 N 只, 摄取 N×240 tick, 验证无损快照。"""
    tickers = _stress_tickers(n)
    ticks = _build_stress_ticks(tickers, day=20261123)
    total = len(ticks)  # n * 240
    print(f"[压测] 标的 {n} 只 (沪 {n // 2} + 深 {n - n // 2}), tick {total} 笔")

    port = _free_port()
    srv = asyncio.create_task(_serve(port, ticks))
    client = ZmqClient(f"tcp://127.0.0.1:{port}", timeout=60.0, rcvhwm=max(65536, total + 4096))
    cache = TopkQuoteCache(client, tickers, hfq_scales={})
    try:
        await client.connect()
        ack = await cache.start()
        print(f"[压测] 订阅: {(ack or {}).get('result')}")

        # 等待全部 tick 摄取 (无损 = 0 丢帧)
        for _ in range(1200):  # 最长 ~60s
            if cache.ticks_ingested >= total:
                break
            await asyncio.sleep(0.05)
        assert cache.ticks_ingested == total, (
            f"丢 tick: {cache.ticks_ingested}/{total} (frames_received={cache.frames_received})"
        )
        print(f"[压测] 摄取 {cache.ticks_ingested}/{total} ticks (0 丢帧, frames_received={cache.frames_received})")

        snap = cache.snapshot_bars(up_to_bar_idx=239)
        assert len(snap["sid"]) == total, f"快照行数 {len(snap['sid'])} != {total}"
        bars = snapshot_to_bars(snap, datetime.date(2026, 11, 23))
        bars.write_parquet(out_parquet)
        print(f"[压测] 快照 {bars.height} 行 x {bars.width} 列 → {out_parquet}")
        print(f"[压测] ✓ {n} 标的无损通过 (订阅→摄取→快照→parquet)")
        return 0
    finally:
        await cache.stop()
        await client.close()
        srv.cancel()
        try:
            await srv
        except asyncio.CancelledError:
            pass


def main() -> int:
    p = argparse.ArgumentParser(description="验证测试账号 XTP 数据链路 / 压测")
    p.add_argument("--endpoint", default="tcp://127.0.0.1:5570")
    p.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    p.add_argument("--seconds", type=int, default=5)
    p.add_argument("--out", default="market_snapshot.parquet")
    p.add_argument("--stress", type=int, default=0, help="N 标的压测 (自包含 mock 行情源, 如 --stress 1000)")
    args = p.parse_args()
    if args.stress > 0:
        return asyncio.run(_stress_run(args.stress, args.out))
    return asyncio.run(_run(args.endpoint, [t.strip() for t in args.tickers.split(",") if t.strip()], args.seconds, args.out))


if __name__ == "__main__":
    raise SystemExit(main())
