# Import the backtrader platform

import warnings
import uuid
import numpy as np
from dotenv import load_dotenv

import bt_core as bt
import bt_core.indicators as btind

warnings.filterwarnings('ignore')


# class TestStrategy(bt.Strategy):
#     params = dict(period=10)

#     def __init__(self):
#         sma0 = btind.SMA(self.data.close, period=25)  # 15 days sma
#         sma1 = btind.SMA(sma0, period=5)  
#         sma2 = btind.SMA(sma1, period=5) 
#         sma3 = btind.SMA(sma2, period=10) 
#         ema = btind.EMA(sma2, period=5)
#         self.macd = btind.MACDHisto(self.data.close)
#         self.buysig = bt.logic.Cmp(sma1, ema)
    
#     def next(self):
#         if self.buysig[0] < 0.0 and self.macd.histo[0] > 0:
#             print("buysig: ", self.buysig[0])
#             self.buy(plimit=2, execType=1, filler="trend") # default 
#         elif self.macd.histo[0] < 0 and self.buysig[0] > 0:
#             print("sellsig: ", self.buysig[0])
#             self.sell(plimit=2, execType=1, filler="trend")


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
    cerebro = bt.Cerebro(client_id=uuid.UUID("e9f8cd38-e73c-453f-8a47-55beda640ae6").bytes) 
    cerebro.addstore() 
    cerebro.addpnc("fixed", days_held=5, stake=0.9, dd=0.25)

    ddata = cerebro.resampledata(timeframe=bt.TimeFrame.Days, adjbartime=False)
    wdata = cerebro.resampledata(timeframe=bt.TimeFrame.Weeks, adjbartime=False)

    cerebro.addstrategy(TestStrategy, ddata)

    cerebro.run(cash=100000, sid=[b"600036"], fromdate=20040301, todate=20260201, benchmark=[b"1A0001"])
