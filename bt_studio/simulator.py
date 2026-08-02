# Import the backtrader platform
import faulthandler
faulthandler.enable()

import os
import uuid
import datetime
import warnings
import numpy as np
import polars as pl
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

import bt_core as bt
import bt_core.indicators as btind

from bt_core.cerebro import Cerebro
from bt_core.feeds import *
from bt_core.brokers import *
from bt_core.pnc import Pnc
from bt_core.utils.dateintern import ts2intdt
from bt_protocol._protocol import SnapshotBody
from bt_studio.bt_trade.sizers import FixedSize


warnings.filterwarnings('ignore')

os.environ["GRPC_POLL_STRATEGY"] = "poll"


class FsmStrategy(bt.Strategy):

    params = (
        ("filler", "oco"),
    )


if __name__ == '__main__':

    load_dotenv()
    cerebro = Cerebro(client_id=uuid.UUID("e9f8cd38-e73c-453f-8a47-55beda640ae6").bytes, fmt="parquet")

    # store / size / pnc
    cerebro.addstore("local")
    cerebro.addsizer(FixedSize)
    cerebro.addpnc(Pnc, days_held=5, stake=0.9, dd=0.25)

    # timer
    cerebro.add_timer(
        when=bt.timer.Session.SESSION_END,
        offset=datetime.timedelta(minutes=-10),
        weekdays=[1, 2, 3, 4, 5],
        weekcarry=False,
        event_type=bt.timer.TimerEvent.TRADE
    )

    # resample
    ddata = cerebro.resampledata(timeframe=bt.TimeFrame.Days, adjbartime=False)
    wdata = cerebro.resampledata(timeframe=bt.TimeFrame.Weeks, adjbartime=False)

    # signal
    cerebro.add_signal(bt.SIGNAL_LONG, MACDSignal, ddata)

    # add parquet
    from bt_core.feeds import ParquetPatch
    patch_data = ParquetPatch(parquet_path=Path("~/startup/bt_studio/result/fsm/scores").expanduser())
    cerebro.adddata(patch_data)
    cerebro.addstrategy(FsmStrategy)

    try:
        cerebro.run(cash=100000, sid=[b"300308"], fromdate=20040101, todate=20260531, benchmark=[b"1A0001"], filler=b"default")
    except Exception as e:
        print(f"运行报错: {e}")
        if hasattr(cerebro, '_shutdown'):
            cerebro._shutdown()