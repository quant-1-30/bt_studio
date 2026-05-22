import backtrader as bt

from backtest.utils.wrapper import register


@register
class SignalStrategy(bt.SignalStrategy):
    def __init__(self):
        # 创建信号
        self.signal_add(bt.SIGNAL_LONG, self.sma1 > self.sma2)
        self.signal_add(bt.SIGNAL_SHORT, self.sma1 < self.sma2)
        
        # 计算移动平均线
        self.sma1 = bt.indicators.SMA(period=10)
        self.sma2 = bt.indicators.SMA(period=30)

# 使用示例
cerebro = bt.Cerebro()
cerebro.addstrategy(SignalStrategy)
cerebro.run()