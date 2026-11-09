#!/usr/bin/env python3
"""Backtest runner consuming OOS score parquets produced by bt_studio."""

import datetime
import faulthandler
import glob
import os
import uuid
import warnings
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

import bt_core as bt
from bt_core.cerebro import Cerebro
from bt_core.feeds import ParquetPatch
from bt_core.pnc import Pnc
from bt_studio.constant import TUNE_SCORE_DIR
from bt_studio.default_config import PRODUCTION_COMMON_PARAMS
from bt_studio.plugins.backtest.sizers import FSMSizer

faulthandler.enable()
warnings.filterwarnings("ignore")


class FsmStrategy(bt.Strategy):
    params = (("name", "fsm"),)


def _discover_backtest_scope(scores_dir: str) -> tuple[list[bytes], int, int]:
    """Discover covered sids and date bounds from scores_*.parquet files."""
    files = sorted(glob.glob(os.path.join(scores_dir, "scores_*.parquet")))
    if not files:
        return (
            [],
            int(os.environ.get("FROM_DATE", PRODUCTION_COMMON_PARAMS["start_date"])),
            int(os.environ.get("TO_DATE", PRODUCTION_COMMON_PARAMS["end_date"])),
        )
    df = pl.scan_parquet(files).select(["sid", "day"]).collect()
    sids = sorted({
        s if isinstance(s, bytes) else str(s).encode()
        for s in df["sid"].unique().to_list()
    })
    d_min = df["day"].min()
    d_max = df["day"].max()
    fromdate = int(d_min.strftime("%Y%m%d")) if hasattr(d_min, "strftime") else int(str(d_min).replace("-", "")[:8])
    todate = int(d_max.strftime("%Y%m%d")) if hasattr(d_max, "strftime") else int(str(d_max).replace("-", "")[:8])
    return (
        sids,
        int(os.environ.get("FROM_DATE", fromdate)),
        int(os.environ.get("TO_DATE", todate)),
    )


if __name__ == "__main__":
    load_dotenv()
    cerebro = Cerebro(
        client_id=uuid.UUID("e9f8cd38-e73c-453f-8a47-55beda640ae6").bytes,
        fmt="parquet",
    )

    # Align holding period (days_held=1) with T entry -> T+1 exit label basis
    days_held = int(os.environ.get("BT_DAYS_HELD", "1"))
    cerebro.addstore("local")
    cerebro.addsizer(FSMSizer, stake=0.9)
    cerebro.addpnc(Pnc, days_held=days_held, stake=0.9, dd=0.25, max_positions=5)

    # T+1 session start risk & exit check
    cerebro.add_timer(
        when=bt.timer.Session.SESSION_START,
        offset=datetime.timedelta(minutes=0),
        weekdays=[1, 2, 3, 4, 5],
        weekcarry=False,
        event_type=bt.timer.TimerEvent.RISK,
    )

    # T session end - 10min (14:50) signal entry plan (matching exclude_bars=10)
    cerebro.add_timer(
        when=bt.timer.Session.SESSION_END,
        offset=datetime.timedelta(minutes=-10),
        weekdays=[1, 2, 3, 4, 5],
        weekcarry=False,
        event_type=bt.timer.TimerEvent.TRADE,
    )

    cerebro.resampledata(timeframe=bt.TimeFrame.Days, adjbartime=False)
    cerebro.resampledata(timeframe=bt.TimeFrame.Weeks, adjbartime=False)

    scores_dir = os.environ.get("BT_STUDIO_SCORE_DIR", TUNE_SCORE_DIR)
    patch_data = ParquetPatch(parquet_path=Path(scores_dir).expanduser())
    cerebro.adddata(patch_data)
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
