# Import the backtrader platform
import faulthandler
faulthandler.enable()

import os
import uuid
import datetime
import warnings
import numpy as np

from dotenv import load_dotenv
from typing import Dict, Any

import bt_core as bt
import bt_core.indicators as btind

from bt_core.cerebro import Cerebro
from bt_core.feeds import *
from bt_core.brokers import *
from bt_core.pnc import Pnc
from bt_protocol._protocol import SnapshotBody

warnings.filterwarnings('ignore')


class FixedSize(bt.Sizer):
    '''
    This sizer simply returns a fixed size for any operation.
    Size can be controlled by number of tranches that a system
    wishes to use to scale into trades by specifying the ``tranches``
    parameter.
    '''
    def __init__(self, *args, **kwargs):
        self.stake = kwargs.get("stake", 1.0)

    def _getsizing(self, topk_info: Dict[bytes, Any], snapshot: SnapshotBody, isbuy: bool):
        if isbuy:
            ratio = self.stake / len(topk_info)
            _sizer = {sid: ratio for sid in topk_info.keys()}
        else:
            _sizer = {p.sid: 1.0 for p in snapshot.positions if p.size > 0}
        return _sizer


class WeekPriceSignal(btind.Indicator): 
    lines = ('signal',)
    params = (("period", 10),) 

    def __init__(self):
        # self.data0 ddata), self.data1 wdata)
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
    params = (('period_me1', 12), ('period_me2', 26), ('period_signal', 9),) # daily

    def __init__(self):
        macd = btind.MACD(self.data0.close, 
                            fast=self.p.period_me1, 
                            slow=self.p.period_me2, 
                            period=self.p.period_signal) 
        self.lines.signal = macd 

    def next(self):
        signal = self.lines.signal[0] # macd', 'signal', 'histo'
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
    params = (("period", 10), ("thres", 0.85)) # daily

    def __init__(self): 
        high_ind = btind.Highest(self.data0.close, period=self.p.period) # inherit from PeriodN(addminperiod(self.p.period)) 
        self.lines.signal = self.data0.close / (high_ind * self.p.thres) - 1.0
    
    def next(self):
        signal = self.lines.signal[0]
        # print("SellSignal ", signal)
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
        self.lines.signal[0] = 0.0 if np.isnan(signal) else signal # np.nan_to_num(signal) used for array not scalar


# class TestStrategy(bt.Strategy):

#     def log(self, txt, dt=None):
#         ''' Logging function for this strategy'''
#         dt = dt or self.datas[0].datetime.datetime()
#         print('%s, %s' % (dt.isoformat(), txt))

#     def __init__(self):
#         # Keep a reference to the "close" line in the data[0] dataseries
#         self.dataclose = self.datas[0].close

#     def next(self):
#         # Simply log the closing price of the series from the reference
#         print('Close, %.2f' % self.dataclose[0])


class TestSignalStrategy(bt.SignalStrategy):

    params = (
        ("name", "test_signal_strategy"),
        ('_accumulate', True),
    )


if __name__ == '__main__':

    load_dotenv()
    cerebro = Cerebro(client_id=uuid.UUID("e9f8cd38-e73c-453f-8a47-55beda640ae6").bytes, fmt="parquet") 

    # store / size  / pnc
    cerebro.addstore("local")
    cerebro.addsizer(FixedSize)
    cerebro.addpnc(Pnc, days_held=5, stake=0.9, dd=0.25, max_positions=5)

    # RISK timer - max_positions 
    cerebro.add_timer(
        when=bt.timer.Session.SESSION_START,
        offset=datetime.timedelta(minutes=0),  
        weekdays=[1, 2, 3, 4, 5],
        weekcarry=False,
        event_type=bt.timer.TimerEvent.RISK
    )

    # TRADE timer - Plan
    cerebro.add_timer(
        # when=datetime.time(14, 50, 0),   
        when=bt.timer.Session.SESSION_END,
        offset=datetime.timedelta(minutes=-10),
        # repeat=datetime.timedelta(minutes=15), # intended for intraday
        weekdays=[1, 2, 3, 4, 5],
        weekcarry=False,
        event_type=bt.timer.TimerEvent.TRADE
    )

    # resample 
    ddata = cerebro.resampledata(timeframe=bt.TimeFrame.Days, adjbartime=False)
    wdata = cerebro.resampledata(timeframe=bt.TimeFrame.Weeks, adjbartime=False)
    # mdata = cerebro.resampledata(timeframe=bt.TimeFrame.Months, adjbartime=False, compression=1)
    # ydata = cerebro.resampledata(timeframe=bt.TimeFrame.Years, adjbartime=False, compression=1)
    
    cerebro.add_signalStrategy(TestSignalStrategy)
    
    # signal
    cerebro.add_signal(bt.SIGNAL_LONG, WeekPriceSignal, ddata, wdata)
    cerebro.add_signal(bt.SIGNAL_LONG_INV, DailyPriceSignal, ddata)
    cerebro.add_signal(bt.SIGNAL_LONG, MACDSignal, ddata)
    cerebro.add_signal(bt.SIGNAL_LONG, VolSignal, ddata)
    cerebro.add_signal(bt.SIGNAL_SHORT, SellSignal, ddata) 
    cerebro.add_signal(bt.SIGNAL_SHORT, DrawDownSignal)
    
    # add parquet
    from bt_core.feeds import SignalPatch
    patch_data = SignalPatch(sid=b"300308")
    cerebro.adddata(patch_data)

    try:
        cerebro.run(cash=100000, sid=[b"300308"], fromdate=20040101, todate=20260531, benchmark=[b"1A0001"])
    except Exception as e:
        print(f"运行报错: {e}")
        if hasattr(cerebro, '_shutdown'):
            cerebro._shutdown()
