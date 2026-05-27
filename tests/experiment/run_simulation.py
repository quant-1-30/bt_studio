import uuid
import numpy as np
import polars as pl
import datetime
from functools import wraps
from typing import Dict, Any
from collections import defaultdict
from dotenv import load_dotenv

import bt_core as bt
import bt_core.indicators as btind
from bt_core.utils.dateintern import ts2intdt


def consume_time(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print('{} cost {}s'.format(func.__name__, round(end_time - start_time, 3)))
        return result
    return wrapper


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
        # self.lines.dummy[0] = 0.0 # dummy line to trigger next

        if current_day in self.context_info:
            info = self.context_info[current_day]
            self._current_metadata = info


class FsmStrategy(bt.Strategy):

    params = (
        ("parquet_path", "./data/fsm/*"),
    )

    def __init__(self):
        self.pr = PanelRanker(parquet_path=self.p.parquet_path)

    def on_dt_over(self, dts):

        super().on_dt_over(dts)

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
            # print("plan ", plan)

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
    cerebro = bt.Cerebro(client_id=uuid.UUID("e9f8cd38-e73c-453f-8a47-55beda640ae6").bytes, fmt="parquet") 
    cerebro.addstore() 
    cerebro.addpnc("fixed", days_held=5, stake=0.9)

    # ddata = cerebro.resampledata(timeframe=bt.TimeFrame.Days, adjbartime=False)
    # wdata = cerebro.resampledata(timeframe=bt.TimeFrame.Weeks, adjbartime=False)
    
    cerebro.addstrategy(FsmStrategy)

    # 600036/ 300308
    try:
        cerebro.run(cash=100000, sid=[b"1A0001"], fromdate=20100101, todate=20121231, benchmark=[b"1A0001"])
    except Exception as e:
        print(f"运行报错: {e}")
        if hasattr(cerebro, '_shutdown'):
            cerebro._shutdown()
