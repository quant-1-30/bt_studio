#!/usr/bin/env python3

from __future__ import annotations

import datetime
import faulthandler
import glob
import os
import uuid
import warnings
import argparse
from pathlib import Path

import bt_core as bt
import polars as pl
from bt_core.cerebro import Cerebro
from bt_core.feeds import ParquetPatch
from bt_core.pnc import Pnc
from dotenv import load_dotenv

from bt_studio.constant import TUNE_BASE_DIR, TUNE_SCORE_DIR
from bt_studio.default_config import PRODUCTION_COMMON_PARAMS
from bt_studio.plugins.backtest.sizers import FSMSizer

faulthandler.enable()
warnings.filterwarnings("ignore")


class FsmStrategy(bt.Strategy):
    params = (("name", "fsm"),)


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

    cerebro = Cerebro(
        client_id=uuid.UUID("e9f8cd38-e73c-453f-8a47-55beda640ae6").bytes,
        fmt="parquet",
    )

    days_held = int(os.environ.get("BT_DAYS_HELD", "1"))
    cerebro.addstore("local")
    cerebro.addsizer(FSMSizer, stake=0.9)
    cerebro.addpnc(Pnc, days_held=days_held, stake=0.9, dd=0.25, max_positions=5)

    cerebro.add_timer(
        when=bt.timer.Session.SESSION_START,
        offset=datetime.timedelta(minutes=0),
        weekdays=[1, 2, 3, 4, 5],
        weekcarry=False,
        event_type=bt.timer.TimerEvent.RISK,
    )
    cerebro.add_timer(
        when=bt.timer.Session.SESSION_END,
        offset=datetime.timedelta(minutes=-10),
        weekdays=[1, 2, 3, 4, 5],
        weekcarry=False,
        event_type=bt.timer.TimerEvent.TRADE,
    )

    cerebro.resampledata(timeframe=bt.TimeFrame.Days, adjbartime=False)
    cerebro.resampledata(timeframe=bt.TimeFrame.Weeks, adjbartime=False)

    scores_dir = _resolve_scores_dir(args.score_dir, args.task_id)
    cerebro.adddata(ParquetPatch(parquet_path=Path(scores_dir).expanduser()))
    cerebro.addstrategy(FsmStrategy)

    sids, fromdate, todate = _discover_backtest_scope(scores_dir)
    benchmark = [str(PRODUCTION_COMMON_PARAMS["benchmark"]).encode()]

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
