# Import the backtrader platform
import faulthandler
faulthandler.enable()

import os
import uuid
import datetime
import warnings
import numpy as np
import polars as pl
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


class WeekPriceSignal(btind.Indicator):
    lines = ('signal',)
    params = (("period", 10),)

    def __init__(self):
        self.sma_weekly = btind.SMA(self.data1.close, period=self.p.period)

    def next(self):
        if len(self.sma_weekly) == 0 or len(self.data0) == 0:
            return

        # =============================================================
        # Look-ahead Bias
        # =============================================================
        if self.data1.datetime.datetime() == self.data0.datetime.datetime():
            if len(self.sma_weekly) < 2:
                return
            last_week_sma = self.sma_weekly[-1]
        else:
            last_week_sma = self.sma_weekly[0]

        self.lines.signal[0] = last_week_sma / self.data0.close[-1] - 1.0
        if self.lines.signal[0] > 10.0:
            raise ValueError(f"WeekPriceSignal corrupted {self.lines.signal[0]}")


class DailyPriceSignal(btind.Indicator):

    lines = ('signal',)
    params = (("period", 120),)

    def __init__(self):
        low_ind = btind.Lowest(self.data0.close, period=self.p.period)
        self.lines.signal = self.data0.close / low_ind  - 2.0

    def next(self):
        signal = self.lines.signal[0]
        if signal > 10.0:
           print("DailyPriceSignal ", signal)
           raise


class MACDSignal(btind.Indicator):

    lines = ('signal',)
    params = (('period_me1', 12), ('period_me2', 26), ('period_signal', 9),)

    def __init__(self):
        macd = btind.MACD(self.data0.close,
                            fast=self.p.period_me1,
                            slow=self.p.period_me2,
                            period=self.p.period_signal)
        self.lines.signal = macd

    def next(self):
        signal = self.lines.signal[0]
        if not np.isnan(signal):
           print("MacdSignal :", signal)


class VolSignal(btind.Indicator):

    lines = ("signal",)
    params = (("period", 10), ("thres", 1.1))

    def __init__(self):
        vsma = btind.SMA(self.data0.volume, period=self.p.period)
        self.lines.signal = vsma / (self.data0.volume * self.p.thres) - 1.0

    def next(self):
        signal = self.lines.signal[0]
        if signal > 30.0:
           print("VolSignal ", signal)


class SellSignal(btind.Indicator):

    lines = ("signal",)
    params = (("period", 10), ("thres", 0.85))

    def __init__(self):
        high_ind = btind.Highest(self.data0.close, period=self.p.period)
        self.lines.signal = self.data0.close / (high_ind * self.p.thres) - 1.0

    def next(self):
        signal = self.lines.signal[0]
        if signal > 10.0:
            raise


class DrawDownSignal(btind.Indicator):

    lines = ('signal',)
    params = (("thres", 0.25),)

    def __init__(self):
        self.thres = self.p.thres
        self.stats = self._owner.stats

    def next(self):
        dd = self.stats["drawdown"].maxdd
        signal = self.thres - dd
        self.lines.signal[0] = 0.0 if np.isnan(signal) else signal


class PanelRanker(bt.Indicator):
    lines = ('dummy',)

    params = (
        ("parquet_path", None),
        ('thres', 0.0),
        ('top_k', 6),
    )

    def __init__(self):
        self.context_info = defaultdict(dict)

        lf = pl.scan_parquet(self.p.parquet_path)

        # ==========================================
        # group by day and struct columns
        # ==========================================
        agg_df = (
            lf
            .with_columns(
                pl.col("sid").cast(pl.Binary)
            )
            .filter(pl.col("score") > self.p.thres)
            .group_by("day")
            .agg(
                pl.struct(["sid", "score", "distance", "macro_state"])
                .sort_by("score", descending=True)
                .head(self.p.top_k)
                .alias("topk_info")
            )
        ).collect()

        for row in agg_df.iter_rows(named=True):
            day = row["day"]
            self.context_info[day] = {
                item["sid"]: item for item in row["topk_info"]
            }
        self._current_metadata = None

    def next(self):
        current_day = ts2intdt(self.data.datetime[0])

        if current_day in self.context_info:
            info = self.context_info[current_day]
            self._current_metadata = info


class FsmStrategy(bt.Strategy):

    params = (
        ("parquet_path", "./data/fsm/*"),
    )

    def __init__(self):
        self.pr = PanelRanker(parquet_path=self.p.parquet_path)

    def next(self):
        current_tick = self.data.datetime[0]
        current_day = ts2intdt(current_tick)
        print("FsmStrategy current_day ", current_day)
        seconds_in_day = int(current_tick) % 86400 # utc 28800
        snapshot = self.get_snapshot()
        psids = [p.sid for p in snapshot.positions]

        pending_sells = self.pnc.get_pending_sells()

        # =========================================================
        # stage1 09:30 —— Pending Sells
        # =========================================================
        if seconds_in_day == 34200:
            # mode eager
            self.sell(pending_sells.values())

        # =========================================================
        # stage2 14:55 —— FSM
        # =========================================================
        elif seconds_in_day == 53700:
            topk_info = self.pr._current_metadata
            if not topk_info: return
            current_prices = self.store.getdata(psids, int(current_tick))
            plan = self.pnc.generate_plan(topk_info, current_prices, snapshot, self.stats)

            # 【卖出指令生成】
            self.sell(plan["sell"])

            # 【买入指令生成】 可以重复建仓
            buy_sids = [plan.core["sid"] for plan in plan["buy"]]
            if len(buy_sids) != len(set(buy_sids)):
                print(f"🚨 严重警告: {current_tick} 这分钟内, 同一个标的被买入多次! 计划列表: {plan['buy']}")
                raise ValueError("重复买入")

            self.buy(plan["buy"])


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
    from bt_core.feeds import SignalPatch, ParquetPatch
    patch_data = SignalPatch(sid=b"300308")
    cerebro.adddata(patch_data)
    cerebro.addstrategy(FsmStrategy)

    try:
        cerebro.run(cash=100000, sid=[b"300308"], fromdate=20040101, todate=20260531, benchmark=[b"1A0001"], filler=b"default")
    except Exception as e:
        print(f"运行报错: {e}")
        if hasattr(cerebro, '_shutdown'):
            cerebro._shutdown()