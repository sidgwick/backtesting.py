"""
Collection of common building blocks, helper auxiliary functions and
composable strategy classes for reuse.

Intended for simple missing-link procedures, not reinventing
of better-suited, state-of-the-art, fast libraries,
such as TA-Lib, Tulipy, PyAlgoTrade, NumPy, SciPy ...

Please raise ideas for additions to this collection on the [issue tracker].

[issue tracker]: https://github.com/kernc/backtesting.py
"""

from collections import OrderedDict
from inspect import currentframe
from itertools import compress
from numbers import Number
from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ._plotting import plot_heatmaps as _plot_heatmaps
from ._stats import compute_stats as _compute_stats
from ._util import _Array, _as_str
from .backtesting import Strategy

__pdoc__ = {}


OHLCV_AGG = OrderedDict((
    ('Open', 'first'),
    ('High', 'max'),
    ('Low', 'min'),
    ('Close', 'last'),
    ('Volume', 'sum'),
))
"""OHLCV_AGG 用来在 pandas 重新采样数据的时候当做采样标准参数, 重新采样之后数据含义是,
在指定的时间范围数据条目(有序)集合中:

- Open 使用第一条数据的 Open 值
- High 使用集合中 High 指标最大的那个值
- Low 使用集合中 Low 指标最小的那个值
- Close 使用集合中最后一条的 Close 值
- Volume 使用集合中所有条目 Volume 的和值

Dictionary of rules for aggregating resampled OHLCV data frames,
e.g.

    df.resample('4H', label='right').agg(OHLCV_AGG).dropna()
"""

TRADES_AGG = OrderedDict((
    ('Size', 'sum'),
    ('EntryBar', 'first'),
    ('ExitBar', 'last'),
    ('EntryPrice', 'mean'),
    ('ExitPrice', 'mean'),
    ('PnL', 'sum'),
    ('ReturnPct', 'mean'),
    ('EntryTime', 'first'),
    ('ExitTime', 'last'),
    ('Duration', 'sum'),
))
"""关于重采样的说明参考 OHLCV_AGG 的说明. 这里每个指标的含义, ChatGPT 给出的解释如下(后续读代码的时候, 注意核实对不对):

- Size: 交易量的总和 (The total size of the trade)
- EntryBar: 进入交易的第一个条目的索引 (The index of the first entry bar)
- ExitBar: 退出交易的最后一个条目的索引 (The index of the last exit bar)
- EntryPrice: 进入交易的平均价格 (The average entry price)
- ExitPrice: 退出交易的平均价格 (The average exit price)
- PnL: 交易的总利润和损失 (The total profit and loss of the trade)
- ReturnPct: 交易的平均收益率 (The average return percentage of the trade)
- EntryTime: 进入交易的时间戳 (The timestamp of the first entry)
- ExitTime: 退出交易的时间戳 (The timestamp of the last exit)
- Duration: 交易的总持续时间 (The total duration of the trade).

Dictionary of rules for aggregating resampled trades data,
e.g.

    stats['_trades'].resample('1D', on='ExitTime',
                              label='right').agg(TRADES_AGG)
"""

_EQUITY_AGG = {
    'Equity': 'last',
    'DrawdownPct': 'max',
    'DrawdownDuration': 'max',
}
"""字段含义
- Equity: 最后的资产净值 (The last equity value)
- DrawdownPct: 最大回撤百分比 (The maximum drawdown percentage)
- DrawdownDuration: 最大回撤持续时间 (The maximum drawdown duration)
"""


def barssince(condition: Sequence[bool], default=np.inf) -> int:
    """
    Return the number of bars since `condition` sequence was last `True`,
    or if never, return `default`.

    知识点
    - compress 函数的使用 - 接受两个序列 (A, B), 根据 B 序列中的真值情况, 保留 A 中的元素
    - next 函数如果不提供 default 值, 迭代到头之后会抛出 StopInteration 异常

        >>> barssince(self.data.Close > self.data.Open)
        3
    """
    return next(compress(range(len(condition)), reversed(condition)), default)


def cross(series1: Sequence, series2: Sequence) -> bool:
    """
    Return `True` if `series1` and `series2` just crossed
    (above or below) each other.

    这个函数检查第一个序列和第二个序列是否产生了相交.

        >>> cross(self.data.Close, self.sma)
        True
    """
    return crossover(series1, series2) or crossover(series2, series1)


def crossover(series1: Sequence, series2: Sequence) -> bool:
    """
    Return `True` if `series1` just crossed over (above)
    `series2`.

    这个函数检查, 第一个序列是否上穿了第二个序列. 原理如下:

    1. 整理数据
        1.1 如果是 pandas.Serise 数据, 直接使用它的 values 值序列
        1.2 如果是 numbers.Number 数据, #### 这块有点没太看懂为啥这么做 ####
        1.3 如果不是上面这两种, 就直接使用原始的序列数据.

        >>> crossover(self.data.Close, self.sma)
        True
    """
    series1 = (
        series1.values if isinstance(series1, pd.Series) else
        (series1, series1) if isinstance(series1, Number) else
        series1)
    series2 = (
        series2.values if isinstance(series2, pd.Series) else
        (series2, series2) if isinstance(series2, Number) else
        series2)
    try:
        # 下面是交叉的判断, 图上画一下就明白了这个式子
        return series1[-2] < series2[-2] and series1[-1] > series2[-1]
    except IndexError:
        return False


def plot_heatmaps(heatmap: pd.Series,
                  agg: Union[str, Callable] = 'max',
                  *,
                  ncols: int = 3,
                  plot_width: int = 1200,
                  filename: str = '',
                  open_browser: bool = True):
    """
    这个函数只是绘图库函数的包装, 等分析到那边的时候在处理具体细节.
    Plots a grid of heatmaps, one for every pair of parameters in `heatmap`.

    `heatmap` is a Series as returned by
    `backtesting.backtesting.Backtest.optimize` when its parameter
    `return_heatmap=True`.

    When projecting the n-dimensional heatmap onto 2D, the values are
    aggregated by 'max' function by default. This can be tweaked
    with `agg` parameter, which accepts any argument pandas knows
    how to aggregate by.

    .. todo::
        Lay heatmaps out lower-triangular instead of in a simple grid.
        Like [`skopt.plots.plot_objective()`][plot_objective] does.

    [plot_objective]: \
        https://scikit-optimize.github.io/stable/modules/plots.html#plot-objective
    """
    return _plot_heatmaps(heatmap, agg, ncols, filename, plot_width, open_browser)


def quantile(series: Sequence, quantile: Union[None, float] = None):
    """
    这个函数用来计算分位数(quantile = 分位数, 这个函数说的分位数实际上是百分位数的意思).

    分位数的概念: 简单的理解平常用到的百分数实际上是百分位数, 中值实际上是二分位数, 常用的还有 4 分位
    分位数指的就是连续分布函数中的一个点, 这个点对应概率 p.
    若概率 0 < p < 1, 随机变量 X 或它的概率分布的分位数 Za, 是指满足条件 p(X<=Za) = a 的实数.

    If `quantile` is `None`, return the quantile _rank_ of the last
    value of `series` wrt former series values.

    `quantile` 是空的时候, 计算最后一个元素在序列中的百分位数.

    If `quantile` is a value between 0 and 1, return the _value_ of
    `series` at this quantile. If used to working with percentiles, just
    divide your percentile amount with 100 to obtain quantiles.

    `quantile` 不是空的时候(0-1 之间), 计算 `quantile` 指定的分位数在 series 里面的数值

        >>> quantile(self.data.Close[-20:], .1)
        162.130
        >>> quantile(self.data.Close)
        0.13
    """
    if quantile is None:
        try:
            last, series = series[-1], series[:-1]
            return np.mean(series < last)
        except IndexError:
            return np.nan
    assert 0 <= quantile <= 1, "quantile must be within [0, 1]"
    return np.nanpercentile(series, quantile * 100)


def compute_stats(
        *,
        stats: pd.Series,
        data: pd.DataFrame,
        trades: pd.DataFrame = None,
        risk_free_rate: float = 0.) -> pd.Series:
    """
    (Re-)compute strategy performance metrics.

    本函数计算策略性能指标.
    pandas 的 iloc 可以用来根据 index 获取特定维度上的数据.
    df.iloc[0] 表示的是获取 df 第一个维度(shape 的第一个元素对应的那个维度)上索引值为 0 的数据.
    df.iloc[x:y] 表示的是获取 df 第一个维度从 x 到 y 的数据.
    df.iloc[x, y] 表示的是获取 df 第一个维度 x 处, 第二个维度 y 处的数据.

    `stats` is the statistics series as returned by `backtesting.backtesting.Backtest.run()`.
    `data` is OHLC data as passed to the `backtesting.backtesting.Backtest`
    the `stats` were obtained in.
    `trades` can be a dataframe subset of `stats._trades` (e.g. only long trades).
    You can also tune `risk_free_rate`, used in calculation of Sharpe and Sortino ratios.

        >>> stats = Backtest(GOOG, MyStrategy).run()
        >>> only_long_trades = stats._trades[stats._trades.Size > 0]
        >>> long_stats = compute_stats(stats=stats, trades=only_long_trades,
        ...                            data=GOOG, risk_free_rate=.02)
    """
    equity = stats._equity_curve.Equity
    if trades is None:
        trades = stats._trades
    else:
        # XXX: Is this buggy?
        equity = equity.copy()
        equity[:] = stats._equity_curve.Equity.iloc[0]
        for t in trades.itertuples(index=False):
            equity.iloc[t.EntryBar:] += t.PnL
    return _compute_stats(trades=trades, equity=equity, ohlc_data=data,
                          risk_free_rate=risk_free_rate, strategy_instance=stats._strategy)


def resample_apply(rule: str,
                   func: Optional[Callable[..., Sequence]],
                   series: Union[pd.Series, pd.DataFrame, _Array],
                   *args,
                   agg: Optional[Union[str, dict]] = None,
                   **kwargs):
    """
    Apply `func` (such as an indicator) to `series`, resampled to
    a time frame specified by `rule`. When called from inside
    `backtesting.backtesting.Strategy.init`,
    the result (returned) series will be automatically wrapped in
    `backtesting.backtesting.Strategy.I`
    wrapper method.

    `rule` is a valid [Pandas offset string] indicating
    a time frame to resample `series` to.

    [Pandas offset string]: \
http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

    `func` is the indicator function to apply on the resampled series.

    `series` is a data series (or array), such as any of the
    `backtesting.backtesting.Strategy.data` series. Due to pandas
    resampling limitations, this only works when input series
    has a datetime index.

    `agg` is the aggregation function to use on resampled groups of data.
    Valid values are anything accepted by `pandas/resample/.agg()`.
    Default value for dataframe input is `OHLCV_AGG` dictionary.
    Default value for series input is the appropriate entry from `OHLCV_AGG`
    if series has a matching name, or otherwise the value `"last"`,
    which is suitable for closing prices,
    but you might prefer another (e.g. `"max"` for peaks, or similar).

    Finally, any `*args` and `**kwargs` that are not already eaten by
    implicit `backtesting.backtesting.Strategy.I` call
    are passed to `func`.

    For example, if we have a typical moving average function
    `SMA(values, lookback_period)`, _hourly_ data source, and need to
    apply the moving average MA(10) on a _daily_ time frame,
    but don't want to plot the resulting indicator, we can do:

        class System(Strategy):
            def init(self):
                self.sma = resample_apply(
                    'D', SMA, self.data.Close, 10, plot=False)

    The above short snippet is roughly equivalent to:

        class System(Strategy):
            def init(self):
                # Strategy exposes `self.data` as raw NumPy arrays.
                # Let's convert closing prices back to pandas Series.
                close = self.data.Close.s

                # Resample to daily resolution. Aggregate groups
                # using their last value (i.e. closing price at the end
                # of the day). Notice `label='right'`. If it were set to
                # 'left' (default), the strategy would exhibit
                # look-ahead bias.
                daily = close.resample('D', label='right').agg('last')

                # We apply SMA(10) to daily close prices,
                # then reindex it back to original hourly index,
                # forward-filling the missing values in each day.
                # We make a separate function that returns the final
                # indicator array.
                def SMA(series, n):
                    from backtesting.test import SMA
                    return SMA(series, n).reindex(close.index).ffill()

                # The result equivalent to the short example above:
                self.sma = self.I(SMA, daily, 10, plot=False)

    """
    if func is None:
        def func(x, *_, **__):
            return x

    if not isinstance(series, (pd.Series, pd.DataFrame)):
        assert isinstance(series, _Array), \
            'resample_apply() takes either a `pd.Series`, `pd.DataFrame`, ' \
            'or a `Strategy.data.*` array'
        series = series.s

    if agg is None:
        # getattr 首先获取 serise 的 name 属性, 然后根据这个属性去 OHLCV 里面找对应的 aggr 函数配置
        # 一般来说我们是把 df.Open/df.Close 之类的东西作为 serise 传入函数中来, 这样得到的 attr 就是 Open/Close
        agg = OHLCV_AGG.get(getattr(series, 'name', ''), 'last')
        if isinstance(series, pd.DataFrame):
            agg = {column: OHLCV_AGG.get(column, 'last')
                   for column in series.columns}

    # 对数据重新采样, 重新采样之后的数据可以作为指标计算的源数据
    # resample 的 label=right 参数, 指定采样区间的右边值作为采样后数据(时间序列)的索引
    # 比如数据 [19,19,20,21,22] 按照 'D' 规则采样完成之后数据索引变成了 [20, 21, 22, 23]
    # 如果按照 left 采样, 则计算完成之后, 采样数据是 [19, 20, 21, 22]
    # 量化策略执行过程中, 我们在 20 日才知道 19 日的具体数据, 因此这里选用的 label 是 right
    resampled = series.resample(rule, label='right').agg(agg).dropna()
    resampled.name = _as_str(series) + '[' + rule + ']'

    # 检查是不是在 Strategy.init 方法中调用进来这里的, 如果是, 就是用 Strategy.I 函数
    # 在接下来生成数组(多数时候是计算指标)过程中对数据进行处理
    # 关于 Strategy.I 函数的说明读到了再细致了解
    # Check first few stack frames if we are being called from
    # inside Strategy.init, and if so, extract Strategy.I wrapper.
    frame, level = currentframe(), 0
    while frame and level <= 3:
        frame = frame.f_back
        level += 1
        if isinstance(frame.f_locals.get('self'), Strategy):  # type: ignore
            strategy_I = frame.f_locals['self'].I             # type: ignore
            break
    else:
        def strategy_I(func, *args, **kwargs):
            return func(*args, **kwargs)

    # 下面这个函数主要是作为给 Strategy.I 的入参使用的
    # TODO: 具体怎么使用这个函数, 需要在 Strategy.I 函数中了解
    def wrap_func(resampled, *args, **kwargs):
        # 通过 func 处理重新采样之后的 resampled 数据, 计算得到指标数据结果
        result = func(resampled, *args, **kwargs)

        # 检查返回指标是否是 pdndas.DataFrame 或者 Series,
        # 如果不是就需要把数据组织成 DataFrame 或者 Series 的形式, 方便以后的计算
        if not isinstance(result, pd.DataFrame) and not isinstance(result, pd.Series):
            result = np.asarray(result)
            if result.ndim == 1:
                result = pd.Series(result, name=resampled.name)
            elif result.ndim == 2:
                result = pd.DataFrame(result.T)

        # Resample back to data index
        # 如果返回的数据还是时间序列, 那么更新结果的索引和计算指标输入一样
        if not isinstance(result.index, pd.DatetimeIndex):
            result.index = resampled.index

        # 把数据指标的索引更新成原始输入 serise 的索引
        # ffill 是在数据数量不匹配的时候, 向前填充 NaN 操作, 因为一般指标都是需要前面 N
        # 天的数据才可以计算, 那么 N 天以前的数据就用到了这个操作, 当时的指标就都是 NaN
        # series.index.union 用于取并集, 这里是 series.index 和 resampled.index 的并集
        # 第一次 reindex 完成之后, 无论指标结果是否是时间序列, 现在它都是时间序列了
        result = result.reindex(index=series.index.union(resampled.index),
                                method='ffill').reindex(series.index)
        return result

    wrap_func.__name__ = func.__name__

    array = strategy_I(wrap_func, resampled, *args, **kwargs)
    return array


def random_ohlc_data(example_data: pd.DataFrame, *,
                     frac=1., random_state: Optional[int] = None) -> pd.DataFrame:
    """
    这个函数是用来生成 OHLC 随机数的.

    OHLC data generator. The generated OHLC data has basic
    [descriptive statistics](https://en.wikipedia.org/wiki/Descriptive_statistics)
    similar to the provided `example_data`.

    `frac` is a fraction of data to sample (with replacement). Values greater
    than 1 result in oversampling.

    Such random data can be effectively used for stress testing trading
    strategy robustness, Monte Carlo simulations, significance testing, etc.

    >>> from backtesting.test import EURUSD
    >>> ohlc_generator = random_ohlc_data(EURUSD)
    >>> next(ohlc_generator)  # returns new random data
    ...
    >>> next(ohlc_generator)  # returns new random data
    ...
    """
    def shuffle(x):
        return x.sample(frac=frac, replace=frac > 1, random_state=random_state)

    if len(example_data.columns.intersection({'Open', 'High', 'Low', 'Close'})) != 4:
        raise ValueError("`data` must be a pandas.DataFrame with columns "
                         "'Open', 'High', 'Low', 'Close'")
    while True:
        df = shuffle(example_data)
        df.index = example_data.index
        padding = df.Close - df.Open.shift(-1)
        gaps = shuffle(example_data.Open.shift(-1) - example_data.Close)
        deltas = (padding + gaps).shift(1).fillna(0).cumsum()
        for key in ('Open', 'High', 'Low', 'Close'):
            df[key] += deltas
        yield df


class SignalStrategy(Strategy):
    """
    这是信号策略的实现示例代码.

    A simple helper strategy that operates on position entry/exit signals.
    This makes the backtest of the strategy simulate a [vectorized backtest].
    See [tutorials] for usage examples.

    [vectorized backtest]: https://www.google.com/search?q=vectorized+backtest
    [tutorials]: index.html#tutorials

    To use this helper strategy, subclass it, override its
    `backtesting.backtesting.Strategy.init` method,
    and set the signal vector by calling
    `backtesting.lib.SignalStrategy.set_signal` method from within it.

        class ExampleStrategy(SignalStrategy):
            def init(self):
                super().init()
                self.set_signal(sma1 > sma2, sma1 < sma2)

    Remember to call `super().init()` and `super().next()` in your
    overridden methods.
    """
    __entry_signal = (0,)
    __exit_signal = (False,)

    def set_signal(self, entry_size: Sequence[float],
                   exit_portion: Optional[Sequence[float]] = None,
                   *,
                   plot: bool = True):
        """
        Set entry/exit signal vectors (arrays).

        A long entry signal is considered present wherever `entry_size`
        is greater than zero, and a short signal wherever `entry_size`
        is less than zero, following `backtesting.backtesting.Order.size` semantics.

        If `exit_portion` is provided, a nonzero value closes portion the position
        (see `backtesting.backtesting.Trade.close()`) in the respective direction
        (positive values close long trades, negative short).

        If `plot` is `True`, the signal entry/exit indicators are plotted when
        `backtesting.backtesting.Backtest.plot` is called.
        """
        # replace(0, np.nan) 表示使用 0 替换掉 NaN 值
        self.__entry_signal = self.I(  # type: ignore
            lambda: pd.Series(entry_size, dtype=float).replace(0, np.nan),
            name='entry size', plot=plot, overlay=False, scatter=True, color='black')

        if exit_portion is not None:
            self.__exit_signal = self.I(  # type: ignore
                lambda: pd.Series(exit_portion, dtype=float).replace(0, np.nan),
                name='exit portion', plot=plot, overlay=False, scatter=True, color='black')

    def next(self):
        super().next()

        # 离场信号, 如果大于 0 表示平多头仓位, 小于 0 表示平空头仓位
        exit_portion = self.__exit_signal[-1]
        if exit_portion > 0:
            for trade in self.trades:
                if trade.is_long:
                    trade.close(exit_portion)
        elif exit_portion < 0:
            for trade in self.trades:
                if trade.is_short:
                    trade.close(-exit_portion)

        # 入场信号, 大于 0 表示开多头仓位, 小于 0 表示开空头仓位
        entry_size = self.__entry_signal[-1]
        if entry_size > 0:
            self.buy(size=entry_size)
        elif entry_size < 0:
            self.sell(size=-entry_size)


class TrailingStrategy(Strategy):
    """
    ATR 跟踪止损策略.

    ATR 计算方法: max(high-low, abs(p_close - high), abs(p_close - low)) 得到的, 其中:

    - p_close 表示昨日收盘价
    - high 表示今日最高价
    - low 表示今日最低价

    A strategy with automatic trailing stop-loss, trailing the current
    price at distance of some multiple of average true range (ATR). Call
    `TrailingStrategy.set_trailing_sl()` to set said multiple
    (`6` by default). See [tutorials] for usage examples.

    [tutorials]: index.html#tutorials

    Remember to call `super().init()` and `super().next()` in your
    overridden methods.
    """
    __n_atr = 6.
    __atr = None

    def init(self):
        super().init()
        self.set_atr_periods()

    def set_atr_periods(self, periods: int = 100):
        """
        设置 ATR 的周期
        Set the lookback period for computing ATR. The default value
        of 100 ensures a _stable_ ATR.
        """
        hi, lo, c_prev = self.data.High, self.data.Low, pd.Series(self.data.Close).shift(1)
        tr = np.max([hi - lo, (c_prev - hi).abs(), (c_prev - lo).abs()], axis=0)

        # the rolling() method is called on the Pandas Series object with the argument periods,
        # which specifies the number of periods to use for the rolling window.
        # The mean() method is then called on the resulting rolling object to calculate
        # the mean of the true range values over the rolling window.
        # Finally, the bfill() method is called to backfill any missing values with the
        # last valid value, and the values attribute is used to return the ATR values as
        # a NumPy array and assign it to the atr variable.
        atr = pd.Series(tr).rolling(periods).mean().bfill().values
        self.__atr = atr

    def set_trailing_sl(self, n_atr: float = 6):
        """
        Sets the future trailing stop-loss as some multiple (`n_atr`)
        average true bar ranges away from the current price.
        """
        self.__n_atr = n_atr

    def next(self):
        super().next()
        # sl = Stop Loss Level
        # 止损位置是收盘价的 n 倍 atr 值
        # Can't use index=-1 because self.__atr is not an Indicator type
        index = len(self.data)-1
        for trade in self.trades:
            if trade.is_long:
                trade.sl = max(trade.sl or -np.inf,
                               self.data.Close[index] - self.__atr[index] * self.__n_atr)
            else:
                trade.sl = min(trade.sl or np.inf,
                               self.data.Close[index] + self.__atr[index] * self.__n_atr)


# Prevent pdoc3 documenting __init__ signature of Strategy subclasses
for cls in list(globals().values()):
    if isinstance(cls, type) and issubclass(cls, Strategy):
        __pdoc__[f'{cls.__name__}.__init__'] = False


# NOTE: Don't put anything below this __all__ list

__all__ = [getattr(v, '__name__', k)
           for k, v in globals().items()                        # export
           if ((callable(v) and v.__module__ == __name__ or     # callables from this module
                k.isupper()) and                                # or CONSTANTS
               not getattr(v, '__name__', k).startswith('_'))]  # neither marked internal

# NOTE: Don't put anything below here. See above.
