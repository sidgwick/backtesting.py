from collections import OrderedDict
import pandas as pd

from .backtesting import Backtest, Strategy, Data, GreatWallBroker


def SMA(values, n):
    """
    Return simple moving average of `values`, at
    each step taking into account `n` previous values.
    """
    a1 = pd.Series(values).rolling(n).mean()
    a2 = pd.Series(values).rolling(n * 2).mean()
    return pd.DataFrame([a1, a2])


class SmaCross(Strategy):
    def init(self):
        print("strategy init")

    def next(self):
        data = self.get_data("GOOG")
        _data = data.aspect(-1)
        print(_data)
        print("strategy next tick")


def run():
    from backtesting.test import GOOG

    data = Data(OrderedDict(GOOG=GOOG))
    broker = GreatWallBroker(cash=10_000)
    strategy = SmaCross()

    bt = Backtest(data, strategy, broker)
    stats = bt.run()
    print(stats)
