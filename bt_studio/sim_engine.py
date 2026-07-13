# Import the backtrader platform
import faulthandler
faulthandler.enable()

import os
import uuid
import datetime
import warnings
import numpy as np
import polars as pl

from dotenv import load_dotenv
from typing import Dict, Any

import bt_core as bt
import bt_core.indicators as btind

from bt_core.cerebro import Cerebro
from bt_core.feeds import *
from bt_core.brokers import *
from bt_core.pnc import Pnc
from bt_protocol._protocol import SnapshotBody
from bt_studio.bt_trade.sizers import FixedSize


warnings.filterwarnings('ignore')

os.environ["GRPC_POLL_STRATEGY"] = "poll"


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


class TestStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.datetime()
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

    def next(self):
        # Simply log the closing price of the series from the reference
        print('Close, %.2f' % self.dataclose[0])


if __name__ == '__main__':

    load_dotenv()
    cerebro = Cerebro(client_id=uuid.UUID("e9f8cd38-e73c-453f-8a47-55beda640ae6").bytes, fmt="parquet") 

    # store / size  / pnc
    cerebro.addstore("local")
    cerebro.addsizer(FixedSize)
    cerebro.addpnc(Pnc, days_held=5, stake=0.9, dd=0.25)

    # timer
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
    
    # signal
    # cerebro.add_signal(bt.SIGNAL_LONG, WeekPriceSignal, ddata, wdata)
    # cerebro.add_signal(bt.SIGNAL_LONG_INV, DailyPriceSignal, ddata)
    cerebro.add_signal(bt.SIGNAL_LONG, MACDSignal, ddata)
    # cerebro.add_signal(bt.SIGNAL_LONG, VolSignal, ddata)
    # cerebro.add_signal(bt.SIGNAL_SHORT, SellSignal, ddata) 
    # cerebro.add_signal(bt.SIGNAL_SHORT, DrawDownSignal)
    
    # add parquet
    from bt_core.feeds import SignalPatch, ParquetPatch
    patch_data = SignalPatch(sid=b"300308")
    cerebro.adddata(patch_data)

    try:
        cerebro.run(cash=100000, sid=[b"300308"], fromdate=20040101, todate=20260531, benchmark=[b"1A0001"], filler=b"default")
    except Exception as e:
        print(f"运行报错: {e}")
        if hasattr(cerebro, '_shutdown'):
            cerebro._shutdown()


# class BacktestEngine:
#     def __init__(self, dret_path: str, score_dir: str, exp_config: dict):
#         self.score_dir = score_dir
#         self.common_config = exp_config["run_params"]
        
#         # 💡 使用 Polars LazyFrame 扫描全局日线数据，按需 collect，内存占用极低
#         self.mkt_scan_lf = pl.scan_parquet(dret_path)
        
#         # --- 状态寄存器 (State Registers) ---
#         self.current_model_id = None      # 当前正在使用的模型/信号周期 ID
#         self.is_current_period_valid = False # 当前周期是否有合法信号文件
#         self.was_prev_day_valid = True     # 前一日信号是否有效（用于边缘检测）
#         self.cached_today_signals = None  # 内存缓存的“今日信号集”
        
#         # --- 账户与交易管理 ---
#         self.active_positions = {}        # 当前持仓: {sid: qty}
#         self.cash = 10000000.0            # 初始现金 1000 万
#         self.liquidation_queue = set()     # 待强制平仓队列 (处理停牌)

#     def _get_expected_model_id(self, current_day: datetime.date) -> int:
#         """
#         💡 纯数学时间对齐：动态计算当前日期应归属的 HPO 周期起点
#         支持任意 update_freq (如 1, 3, 6, 12 个月)
#         """
#         step = self.common_config["update_freq"] # 比如 6
#         # 将当前月份对齐到最近的下边界 (例如 4月 在 step=6 时对齐到 1月)
#         boundary_month = ((current_day.month - 1) // step) * step + 1
#         expected_model_id = current_day.year * 100 + boundary_month
#         return expected_model_id

#     def next(self, current_day: datetime.date):
#         """
#         每日流式交易主循环 (100% 杜绝前视误差)
#         """
#         # 1. 动态确定当前所属的模型周期
#         expected_model_id = self._get_expected_model_id(current_day)
        
#         # =====================================================================
#         # 💡 核心机制 A：周期边界交叉检测 (Boundary Crossover Detection)
#         # =====================================================================
#         if expected_model_id != self.current_model_id:
#             # 跨越了 6 个月的周期边界
#             self.current_model_id = expected_model_id
#             score_path = f"{self.score_dir}/scores_{expected_model_id}.parquet"
            
#             if os.path.exists(score_path):
#                 # 寻优成功，整建制载入新周期信号到内存缓存，物理 I/O 在 6 个月内仅发生一次
#                 self.cached_signals_df = pl.read_parquet(score_path)
#                 self.is_current_period_valid = True
#                 print(f"🌲 [Engine] {current_day} | 成功装载新周期信号: scores_{expected_model_id}.parquet")
#             else:
#                 # 寻优失败，进入黑暗期
#                 self.cached_signals_df = None
#                 self.is_current_period_valid = False
#                 print(f"🚨 [Engine] {current_day} | 警告: 未找到 scores_{expected_model_id}.parquet，系统切入无信号保护状态！")

#         # 2. 提取今日的市场行情切片 (用于获取平仓价格、判断停牌)
#         today_mkt = self.mkt_scan_lf.filter(pl.col("date") == current_day).collect()

#         # =====================================================================
#         # 💡 核心机制 B：【边缘触发】从 有信号 跨入 无信号 的瞬间，执行全量强制清仓 [4]
#         # =====================================================================
#         if (not self.is_current_period_valid) and self.was_prev_day_valid:
#             print(f"🧹 [Boundary] {current_day} 跨入无信号空洞起点！强行清仓当前持仓并锁定...")
#             for sid in list(self.active_positions.keys()):
#                 self.liquidation_queue.add(sid)

#         # 更新前一日状态寄存器
#         self.was_prev_day_valid = self.is_current_period_valid

#         # =====================================================================
#         # 💡 核心机制 C：【待平仓队列处理】流式处理停牌股延迟复牌清仓 [1]
#         # =====================================================================
#         if len(self.liquidation_queue) > 0:
#             for sid in list(self.liquidation_queue):
#                 mkt_row = today_mkt.filter(pl.col("sid") == sid)
                
#                 # 物理规则约束：只有今天不停牌（有成交量）才允许平仓！ [1]
#                 if mkt_row.height > 0 and mkt_row["volume"][0] > 0:
#                     exec_price = mkt_row["close"][0]
#                     qty = self.active_positions[sid]
                    
#                     # 释放资金，物理平仓
#                     self.cash += qty * exec_price * (1.0 - 0.0015) # 扣税
#                     del self.active_positions[sid]
#                     self.liquidation_queue.remove(sid)
#                     print(f"🧹 [Liquidation] {sid} 复牌，成功强制清仓释放资金 | 价格: {exec_price}")
#                 else:
#                     print(f"🔒 [Settle Lock] 停牌锁定：{sid} 处于停牌中，强制清仓指令顺延...")

#         # =====================================================================
#         # 💡 核心机制 D：【空仓交易锁】
#         # =====================================================================
#         if not self.is_current_period_valid:
#             # 拒绝任何买入指令，保持 100% 现金空仓（已停牌锁定的残余仓位除外）
#             return

#         # =====================================================================
#         # 💡 正常交易流（仅在 is_current_period_valid = True 时执行）
#         # =====================================================================
#         # 从内存缓存中快速检索今日的信号
#         today_signals = self.cached_signals_df.filter(pl.col("day") == current_day)
        
#         self.execute_normal_fsm_trading(today_signals, today_mkt)

#     def execute_normal_fsm_trading(self, today_signals: pl.DataFrame, today_mkt: pl.DataFrame):
#         """在这里安全地实现你原来的 FSM 仓位调仓逻辑"""
#         pass

