import backtest as bt
import backtest.indicators as btind


class WeekPriceSignal(btind.Indicator): 

    lines = ('signal',)
    params = (("period", 10),) 

    def __init__(self):
        sma = btind.SMA(self.data0.close, period=self.p.period)
        self.lines.signal = sma / self.data1.close - 1.0

    def next(self):
        signal = self.lines.signal[0]
        # if signal > 10.0:
        #     print("WeekPriceSignal ", signal)


class DailyPriceSignal(btind.Indicator): 

    lines = ('signal',)
    params = (("period", 120),) 

    def __init__(self):
        low_ind = btind.Lowest(self.data0.close, period=self.p.period) 
        self.lines.signal = self.data0.close / low_ind  - 2.0
    
    def next(self):
        signal = self.lines.signal[0]
        # if signal > 10.0:
        #     print("DailyPriceSignal ", signal)


class MACDSignal(btind.Indicator): 

    lines = ('signal',)
    params = (('period_me1', 12), ('period_me2', 26), ('period_signal', 9),) # daily

    def __init__(self):
        macd = btind.MACDHisto(self.data0.close, 
                            period_me1=self.p.period_me1, 
                            period_me2=self.p.period_me2, 
                            period_signal=self.p.period_signal) 
        self.lines.signal = macd 

    # def next(self):
    #     signal = self.lines.signal[0] # histo
    #     if not np.isnan(signal):
    #         print("MacdSignal :", signal)


class VolSignal(btind.Indicator):

    lines = ("signal",)
    params = (("period", 10), ("thres", 1.1)) 

    def __init__(self):
        vsma = btind.SMA(self.data0.volume, period=self.p.period)
        self.lines.signal = vsma / (self.data0.volume * self.p.thres) - 1.0

    def next(self):
        signal = self.lines.signal[0]
        # if signal > 10.0:
        #     print("VolSignal ", signal)


class SellSignal(btind.Indicator): 

    lines = ("signal",)
    params = (("period", 10), ("thres", 0.85)) # daily

    def __init__(self): 
        high_ind = btind.Highest(self.data0.close, period=self.p.period) # inherit from PeriodN(addminperiod(self.p.period)) 
        self.lines.signal = self.data0.close / (high_ind * self.p.thres) - 1.0
    
    def next(self):
        signal = self.lines.signal[0]
        # if signal > 10.0:
        #     print("SellSignal ", signal)


class DrawDownSignal(btind.Indicator): 

    lines = ('signal',)
    params = (("thres", 0.25),)

    def next(self):
        obs = self._owner.stats.getbyname("drawdown") # lowercase
        signal = self.p.thres - obs.lines.drawdown[0]
        self.lines.signal[0] = signal
