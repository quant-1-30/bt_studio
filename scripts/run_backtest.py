#!/usr/bin/env python3
"""Stage 3 Functional Entry: bt_core Portfolio Backtest Execution aligned with common_config."""

from __future__ import annotations

import argparse
import datetime
import faulthandler
import glob
import os
import uuid
import warnings
from pathlib import Path

import bt_core as bt
import polars as pl
from bt_core.cerebro import Cerebro
from bt_core.feeds import ParquetPatch
from bt_core.pnc import Pnc
from dotenv import load_dotenv

from bt_studio.constant import BEIJING_UTC_OFFSET_SEC, SESSION_OPEN_MINUTE, TUNE_BASE_DIR, TUNE_SCORE_DIR
from bt_studio.default_config import PRODUCTION_COMMON_PARAMS
from bt_studio.trade import FSMSizer, exit_stage_from_minutes, sizer_params
from bt_studio.utils.math_func import entry_bar_clock

faulthandler.enable()
warnings.filterwarnings("ignore")


class FsmBacktestStrategy(bt.Strategy):
    params = (
        ("name", "fsm"),
        ("exit_minutes", (5, 15, 30)),
    )

    def __init__(self):
        self._exit_offsets = sorted(int(m) for m in self.p.exit_minutes)

    def on_risk(self, current_dts: int):
        snapshot = self.get_snapshot()
        current_day = bt.utils.ts2intdt(current_dts)
        bj_sec = (int(current_dts) + BEIJING_UTC_OFFSET_SEC) % 86400
        mins_from_open = max(0, (bj_sec - SESSION_OPEN_MINUTE * 60) // 60)
        # 出场分批 stage 映射与实盘共用同一函数 (FsmStrategy.exit_stage_from_minutes)
        stage_idx = exit_stage_from_minutes(self._exit_offsets, mins_from_open)
        self.env.sizer.set_exit_stage(stage_idx)
        sell_plans = self.pnc.on_risk(snapshot, self.stats, current_day, filler=b"default")
        if sell_plans:
            self.sell(sell_plans)

    def on_trade(self, current_dts: int):
        current_day = bt.utils.ts2intdt(current_dts)
        topk = self.datas[-1].get_topk(current_day)
        snapshot = self.get_snapshot()
        # 默认 PseudoFiller(收盘价成交)而非 TWAP:TWAP/VWAP 需实盘 XTP 切片支持,
        # market 模式(收盘价)下回测与实盘成交价偏差最小。
        plan = self.pnc.generate_plan(current_day, topk, snapshot, filler=b"default")
        self.sell(plan[b"sell"])
        self.buy(plan[b"buy"])


def _resolve_scores_dir(explicit_dir: str | None = None, task_id: str | None = None) -> str:
    if explicit_dir:
        return explicit_dir
    if task_id:
        return os.path.join(TUNE_BASE_DIR, task_id, "scores")
    if glob.glob(os.path.join(TUNE_SCORE_DIR, "scores_*.parquet")):
        return TUNE_SCORE_DIR
    candidates = sorted(
        glob.glob(os.path.join(TUNE_BASE_DIR, "*", "scores")),
        key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0.0,
        reverse=True,
    )
    for cand in candidates:
        if glob.glob(os.path.join(cand, "scores_*.parquet")):
            return cand
    return TUNE_SCORE_DIR


def _discover_backtest_scope(scores_dir: str) -> tuple[list[bytes], int, int]:
    """Discover covered sids and date bounds from scores_*.parquet via streaming aggregation."""
    files = sorted(glob.glob(os.path.join(scores_dir, "scores_*.parquet")))
    if not files:
        return (
            [],
            int(os.environ.get("FROM_DATE", PRODUCTION_COMMON_PARAMS["start_date"])),
            int(os.environ.get("TO_DATE", PRODUCTION_COMMON_PARAMS["end_date"])),
        )
    row = (
        pl.scan_parquet(files)
        .select([
            pl.col("sid").unique().implode().alias("sids"),
            pl.col("day").min().alias("d_min"),
            pl.col("day").max().alias("d_max"),
        ])
        .collect(engine="streaming")
        .row(0, named=True)
    )
    sids = sorted({
        s if isinstance(s, bytes) else str(s).encode()
        for s in (row["sids"] or [])
    })
    d_min, d_max = row["d_min"], row["d_max"]
    fromdate = int(d_min.strftime("%Y%m%d")) if hasattr(d_min, "strftime") else int(str(d_min).replace("-", "")[:8])
    todate = int(d_max.strftime("%Y%m%d")) if hasattr(d_max, "strftime") else int(str(d_max).replace("-", "")[:8])
    return (
        sids,
        int(os.environ.get("FROM_DATE", fromdate)),
        int(os.environ.get("TO_DATE", todate)),
    )


def main() -> None:
    load_dotenv()

    p = argparse.ArgumentParser(description="Run bt_core Portfolio Backtest")
    p.add_argument("--task-id", default=os.environ.get("BT_STUDIO_TASK_ID"), help="Task ID bucket under result/artifacts/tune/<task_id>/scores")
    p.add_argument("--score-dir", default=os.environ.get("BT_STUDIO_SCORE_DIR"), help="Explicit directory containing scores_*.parquet")
    args = p.parse_args()

    common = dict(PRODUCTION_COMMON_PARAMS)
    # 回测与实盘共用同一套 sizer/pnc 配置(env 覆盖), 最小化回测↔实盘漂移。
    common["days_held"] = int(os.environ.get("BT_DAYS_HELD", "1"))
    common["max_positions"] = int(os.environ.get("BT_MAX_POSITIONS", "10"))
    common["max_single_weight"] = float(os.environ.get("BT_MAX_SINGLE_WEIGHT", "0.5"))
    common["stake"] = float(os.environ.get("BT_STAKE", "0.9"))
    common["dd"] = float(os.environ.get("BT_DD", "0.25"))
    common["stop_loss"] = float(os.environ.get("BT_STOP_LOSS", "-0.1"))

    t1_rets: dict[str, int] = dict(common["T1_rets"])
    exit_minutes = tuple(sorted(int(m) for m in t1_rets.values()))
    entry_hh, entry_mm = entry_bar_clock(common)  # e.g. bar 230 -> 14:51 close

    cerebro = Cerebro(
        client_id=uuid.UUID("e9f8cd38-e73c-453f-8a47-55beda640ae6").bytes,
        fmt="parquet",
    )

    cerebro.addstore("local")
    # sizer 参数与实盘 make_sizer 同源 (sizer_params), 单一构造路径零漂移。
    cerebro.addsizer(FSMSizer, **sizer_params(common))
    cerebro.addpnc(Pnc, days_held=common["days_held"], dd=common["dd"], stop_loss=common["stop_loss"],
                   max_positions=common["max_positions"], exit_on_risk=True)

    # Register T+1 morning tranche exit timers strictly matching T1_rets (e.g. +5m, +15m, +30m)
    for m_off in exit_minutes:
        cerebro.add_timer(
            when=bt.timer.Session.SESSION_START,
            offset=datetime.timedelta(minutes=m_off),
            weekdays=[1, 2, 3, 4, 5],
            weekcarry=False,
            event_type=bt.timer.TimerEvent.RISK,
        )

    # Register T afternoon entry timer strictly matching entry_bar_idx(common_config):
    # the TRADE timer fires on the entry bar's END-labeled close (bar 230 -> 14:51),
    # so the entry price equals close @ bar 230 exactly as the T+1 labels assume.
    cerebro.add_timer(
        when=datetime.time(entry_hh, entry_mm),
        weekdays=[1, 2, 3, 4, 5],
        weekcarry=False,
        event_type=bt.timer.TimerEvent.TRADE,
    )

    cerebro.resampledata(timeframe=bt.TimeFrame.Days, adjbartime=False)
    cerebro.resampledata(timeframe=bt.TimeFrame.Weeks, adjbartime=False)

    scores_dir = _resolve_scores_dir(args.score_dir, args.task_id)
    cerebro.adddata(ParquetPatch(parquet_path=Path(scores_dir).expanduser()))
    cerebro.addstrategy(FsmBacktestStrategy, exit_minutes=exit_minutes)

    sids, fromdate, todate = _discover_backtest_scope(scores_dir)
    benchmark = [str(common["benchmark"]).encode()]

    try:
        cerebro.run(
            cash=float(os.environ.get("BT_CASH", "1000000")),
            sid=sids,
            fromdate=fromdate,
            todate=todate,
            benchmark=benchmark,
        )
    except Exception as e:
        print(f"Backtest error: {e}")
        if hasattr(cerebro, "_shutdown"):
            cerebro._shutdown()


if __name__ == "__main__":
    main()
