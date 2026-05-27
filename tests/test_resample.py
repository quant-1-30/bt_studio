# Import the backtrader platform
import uuid
import warnings
import numpy as np
import datetime
from dotenv import load_dotenv

from bt_sdk.core.protocol import *
import bt_core as bt
import bt_core.indicators as btind

warnings.filterwarnings('ignore')


class TestResample(bt.Strategy):

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt, txt))

    def __init__(self):

        # data0 is a daily data
        sma0 = btind.SMA(self.data.close, period=25)  # 15 days sma


if __name__ == '__main__':
    
    load_dotenv()

    # 2>/dev/null
    cerebro = bt.Cerebro(client_id=uuid.UUID("e9f8cd38-e73c-453f-8a47-55beda640ae6").bytes, stdstats=False) 
    cerebro.addstore()

    cerebro.addpnc("fixed", days_held=5, stake=0.9, dd=0.25)

    cerebro.add_timer(
        when=bt.timer.Session.SESSION_START, 
        offset=datetime.timedelta(minutes=10), 
        repeat=datetime.timedelta(minutes=15)
    ) 

    data1 = cerebro.resampledata(timeframe=bt.TimeFrame.Days, adjbartime=False)

    data2 = cerebro.resampledata(timeframe=bt.TimeFrame.Weeks, adjbartime=False, compression=1)

    data3 = cerebro.resampledata(timeframe=bt.TimeFrame.Months, adjbartime=False, compression=1)

    data4 = cerebro.resampledata(timeframe=bt.TimeFrame.Years, adjbartime=False, compression=1)
    
    # Add a strategy
    cerebro.addstrategy(TestResample)

    # data1 = data0.clone()
    # data1.addfilter(bt.filters.Renko, **fkwargs) 
    # data.addfilter(btfilters.SessionFiller, fill_vol=args.fvol)

    try:
        cerebro.run(cash=10000, sid=[b"603676"], fromdate=20200101, todate=20210101, benchmark=[b"1A0001"]) # localhost
    except Exception as e:
        print(f"运行报错: {e}")
        if hasattr(cerebro, '_shutdown'):
            cerebro._shutdown()
 