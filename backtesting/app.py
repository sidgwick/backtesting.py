import pandas as pd

from .backtesting import Backtest
from .backtesting import Strategy
from .lib import crossover


def SMA(values, n):
    """
    Return simple moving average of `values`, at
    each step taking into account `n` previous values.
    """
    a1 = pd.Series(values).rolling(n).mean()
    a2 = pd.Series(values).rolling(n * 2).mean()
    return pd.DataFrame([a1, a2])


class SmaCross(Strategy):
    # Define the two MA lags as *class variables*
    # for later optimization
    n1 = 10
    n2 = 20

    def init(self):
        # Precompute the two moving averages
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)

    def next(self):
        # If sma1 crosses above sma2, close any existing
        # short trades, and buy the asset
        if crossover(self.sma1, self.sma2):
            self.position.close()
            bo = self.buy(sl=123)
            bo.cancel()
            bo._replace(a=123)

        # Else, if sma1 crosses below sma2, close any existing
        # long trades, and sell the asset
        elif crossover(self.sma2, self.sma1):
            self.position.close()
            self.sell()


def run():
    from .test import GOOG

    bt = Backtest(GOOG, SmaCross, cash=10_000, commission=0.002)
    stats = bt.run()
    bt.plot()
    print(stats)
