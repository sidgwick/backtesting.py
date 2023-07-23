from math import copysign
from collections import OrderedDict
from typing import Any, Callable, List, Dict, Tuple, Type, Optional, TypeAlias
from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np

from .exception import OutOfMoneyError
from .util import _Data, Data, Indicator
from .stats import compute_stats


class Strategy(metaclass=ABCMeta):
    def __init__(self):
        pass

    def set_backtesting(self, bt: "Backtest"):
        self._backtest = bt

    def get_data(self, name: str) -> _Data:
        return self._backtest.get_data(name)

    def I(self, code: str, name: str, func: Callable[[Any], Indicator], **kwargs):
        i: Indicator = func(**kwargs)
        self.get_data(code).set_indicator(name, i)

    @abstractmethod
    def init(self):
        """策略初始化方法"""

    @abstractmethod
    def next(self):
        """执行具体的 bar"""

    def _next(self):
        self.next()


class Trade:
    """
    When an `Order` is filled, it results in an active `Trade`.
    Find active trades in `Strategy.trades` and closed, settled trades in `Strategy.closed_trades`.
    """

    def __init__(self, broker: "Broker", size: int, entry_price: float, entry_bar, tag):
        self.broker = broker
        self.size = size
        self.entry_price = entry_price
        self.exit_price: Optional[float] = None
        self.entry_bar: int = entry_bar
        self.exit_bar: Optional[int] = None
        self.sl_order: Optional[Order] = None
        self.tp_order: Optional[Order] = None
        self.tag = tag

    def __repr__(self):
        return (
            f'<Trade size={self.size} time={self.entry_bar}-{self.exit_bar or ""} '
            f'price={self.entry_price}-{self.exit_price or ""} pl={self.pl:.0f}'
            f'{" tag="+str(self.tag) if self.tag is not None else ""}>'
        )

    def close(self, portion: float = 1.0):
        """Place new `Order` to close `portion` of the trade at next market price."""
        assert 0 < portion <= 1, "portion must be a fraction between 0 and 1"
        size = copysign(max(1, round(abs(self.size) * portion)), -self.size)
        order = Order(self.broker, size, parent_trade=self, tag=self.tag)
        self.broker.orders.insert(0, order)


class Order:
    def __init(
        self,
        symbol: str,
        broker: "Broker",
        size: float,
        limit_price: Optional[float] = None,  # limit_price 是限价单的价格
        stop_price: Optional[float] = None,  # stop_price 是止损单的价格
        sl_price: Optional[float] = None,  # sl_price 是止损限价单的价格
        tp_price: Optional[float] = None,  # tp_price 是止盈单的价格
        parent_trade: Optional["Trade"] = None,  # parent_trade 是与订单相关联的交易
        tag: object = None,  # tag 是一个标签，用于标识订单
    ):
        self.symbol = symbol
        self.broker = broker
        self.size = size
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.sl_price = sl_price
        self.tp_price = tp_price
        self.parent_trade = parent_trade
        self.tag = tag

    def can(self, data: pd.Series) -> bool:
        """判断 data 表示的 OHLC 条件的时候, 订单能否成交"""
        if data.low <= self.limit_price <= data.high:
            return True

        return False

    def fill(self, data: pd.Series) -> Trade:
        """填充订单"""
        assert self.can(data)

        trade = Trade(
            broker=self.broker,
            size=self.size,
            entry_price=self.limit_price,
            entry_bar=data.name,
            tag="",
        )

        return trade


class Broker(metaclass=ABCMeta):
    def __init__(self, *, cash: float = 10_000):
        self.cash: float = cash
        self._backtest: "Backtest" = None
        self.trades: List[Trade] = []
        self.orders: List[Order] = []

    def set_backtesting(self, bt: "Backtest"):
        self._backtest = bt

    @property
    def equity(self):
        return self.cash + sum([t.size * t.entry_price for t in self.trades])

    @abstractmethod
    def fee(self, order: Order) -> float:
        """计算如果完成 Order 订单的话, 过程中需要的费用"""

    @abstractmethod
    def next(self):
        """一根新 K 线到来需要做的事情"""

    def _next(self):
        self.next()

    def get_data(self, name: str) -> _Data:
        return self._backtest.get_data()


class GreatWallBroker(Broker):
    """模拟 A 股券商"""

    def fee(self, order: Order) -> float:
        return 5

    def next(self):
        """在这里撮合订单"""
        for order in self.orders:
            name = order.symbol
            data = self.get_data(name)
            _data = data.aspect(-1)

            # 上一个 bar 发出的交易指令, 在这个 bar 里面撮合
            if not order.can(_data):
                continue

            # 撮合订单成为 Trade
            trade = order.fill()
            self.trades.append(trade)
            self.orders.remove(order)


class Backtest:
    def __init__(
        self,
        data: Data,
        strategy: Strategy,
        broker: Broker,
        benchmark: str = None,
    ):
        self._data: Data = data
        self._broker = broker
        self._strategy = strategy
        self._results: Optional[pd.Series] = None
        self._benchmark = data

    def get_data(self, name: str) -> _Data:
        return self._data.get(name)

    def get_indicator(self, name: str) -> Indicator:
        pass

    def run(self, **kwargs) -> pd.Series:
        data = self._data
        broker = self._broker
        strategy = self._strategy

        broker.set_backtesting(self)
        strategy.set_backtesting(self)
        strategy.init()

        # Disable "invalid value encountered in ..." warnings. Comparison
        # np.nan >= 3 is not invalid; it's False.
        with np.errstate(invalid="ignore"):
            # 以 self._data 里面的第一个数据表作为时间 tick 序列
            _, first = self._data.first()

            for i in range(0, len(first)):
                data.set_length(i)

                try:
                    broker._next()
                except OutOfMoneyError:
                    break

                # Next tick, a moment before bar close
                strategy._next()
            else:
                # Close any remaining open trades so they produce some stats
                for trade in broker.trades:
                    trade.close()

                broker._next()

            equity = pd.Series(broker.equity).bfill().fillna(broker.cash).values
            self._results = compute_stats(
                trades=broker.closed_trades,
                equity=equity,
                ohlc_data=self._data,
                risk_free_rate=0.0,
                strategy_instance=strategy,
            )

        return self._results
