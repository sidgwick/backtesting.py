from typing import TYPE_CHECKING, List, Union

import numpy as np
import pandas as pd

from ._util import _data_period

if TYPE_CHECKING:
    from .backtesting import Strategy, Trade


def compute_drawdown_duration_peaks(dd: pd.Series):
    '''函数的入参是资产的 `1 - equity / np.maximum.accumulate(equity)` 结果(回撤率),
    所以值为 0 的位置, 就是时间序列中曾经资产值最多的那些点.
    这个函数并不是用来计算最大回撤的, 他会把所有的回撤计算出来.
    '''
    # (dd == 0).values.nonzero()[0] 这个表达式用于找到 dd 中值不为 0 的元素索引
    # len(dd) - 1 是 dd 中最后一个元素的索引
    # np.r_[a, b] 可以将 a 和 b 两个数组连接起来
    # 以上操作完成之后, 就得到了 dd 中全部非零元素或者最后一个元素的索引值序列
    # 然后使用 unique 函数将索引去重
    iloc = np.unique(np.r_[(dd == 0).values.nonzero()[0], len(dd) - 1])

    # 将 np.array 类型的 iloc, 转化为 pandas Series 类型, 其索引是 dd 序列中的索引
    iloc = pd.Series(iloc, index=dd.index[iloc])

    # 下面将 iloc 转化为 pandas DataFrame, 指定 frame 的列就是 'iloc', 索引是 iloc 的索引
    # assign 语句继续操作刚刚生成的 frame 为他增加一个新的叫做 prev 的列,
    # 这个列的值, 是 iloc 元素都往后移动 1 个位置, 第一个位置被填充为 NaN
    df = iloc.to_frame('iloc').assign(prev=iloc.shift())

    # 找到 ft 里面, iloc 列 大于 prev 列加 1 的那些行, 让后把行里面的值全都强制转换为 int 类型
    df = df[df['iloc'] > df['prev'] + 1].astype(int)

    # If no drawdown since no trade, avoid below for pandas sake and return nan series
    if not len(df):
        return (dd.replace(0, np.nan),) * 2

    # 下面是回撤周期和回撤数值
    df['duration'] = df['iloc'].map(dd.index.__getitem__) - df['prev'].map(dd.index.__getitem__)
    df['peak_dd'] = df.apply(lambda row: dd.iloc[row['prev']:row['iloc'] + 1].max(), axis=1)
    df = df.reindex(dd.index)
    return df['duration'], df['peak_dd']


def geometric_mean(returns: pd.Series) -> float:
    '''计算几何平均数'''
    returns = returns.fillna(0) + 1
    if np.any(returns <= 0):
        return 0
    return np.exp(np.log(returns).sum() / (len(returns) or np.nan)) - 1


def compute_stats(
        trades: Union[List['Trade'], pd.DataFrame],
        equity: np.ndarray,
        ohlc_data: pd.DataFrame,
        strategy_instance: 'Strategy',
        risk_free_rate: float = 0,
) -> pd.Series:
    '''本函数用于计算策略的统计值
    参考:
    - [年化收益/波动/夏普率计算](https://blog.csdn.net/qq_41281698/article/details/125546451)
    '''

    # 无风险利率, 用来计算夏普比率和索提诺比率
    assert -1 < risk_free_rate < 1

    index = ohlc_data.index

    # maximum of an array. It takes an array as input and returns an array of the same
    # shape, where each element is the maximum value encountered so far in the input array.
    #
    # For example, if we have an input array [3, 1, 4, 1, 5, 9, 2, 6, 5], the output
    # of np.maximum.accumulate would be [3, 3, 4, 4, 5, 9, 9, 9, 9].
    dd = 1 - equity / np.maximum.accumulate(equity)
    dd_dur, dd_peaks = compute_drawdown_duration_peaks(pd.Series(dd, index=index))

    equity_df = pd.DataFrame({
        'Equity': equity,
        'DrawdownPct': dd, # 回撤率
        'DrawdownDuration': dd_dur}, # 回撤区间
        index=index)

    # 下面转化 trades 的数据类型, 方便后续的分析处理
    if isinstance(trades, pd.DataFrame):
        trades_df: pd.DataFrame = trades
    else:
        # Came straight from Backtest.run()
        trades_df = pd.DataFrame({
            'Size': [t.size for t in trades],
            'EntryBar': [t.entry_bar for t in trades],
            'ExitBar': [t.exit_bar for t in trades],
            'EntryPrice': [t.entry_price for t in trades],
            'ExitPrice': [t.exit_price for t in trades],
            'PnL': [t.pl for t in trades], # The total profit and loss of the trade
            'ReturnPct': [t.pl_pct for t in trades],
            'EntryTime': [t.entry_time for t in trades],
            'ExitTime': [t.exit_time for t in trades],
            'Tag': [t.tag for t in trades],
        })

        # 交易的持有时间
        trades_df['Duration'] = trades_df['ExitTime'] - trades_df['EntryTime']

    del trades

    pl = trades_df['PnL']
    returns = trades_df['ReturnPct']
    durations = trades_df['Duration']

    def _round_timedelta(value, _period=_data_period(index)):
        if not isinstance(value, pd.Timedelta):
            return value
        resolution = getattr(_period, 'resolution_string', None) or _period.resolution
        return value.ceil(resolution)

    s = pd.Series(dtype=object)
    s.loc['Start'] = index[0]
    s.loc['End'] = index[-1]
    s.loc['Duration'] = s.End - s.Start

    # 是否有持仓
    have_position = np.repeat(0, len(index))
    for t in trades_df.itertuples(index=False):
        have_position[t.EntryBar:t.ExitBar + 1] = 1

    # 持仓时间占比
    s.loc['Exposure Time [%]'] = have_position.mean() * 100  # In "n bars" time, not index time
    s.loc['Equity Final [$]'] = equity[-1] # 期末资产
    s.loc['Equity Peak [$]'] = equity.max() # 历史最大资产
    s.loc['Return [%]'] = (equity[-1] - equity[0]) / equity[0] * 100 # 回报率

    # 计算从一开始就买入, 中间无任何操作, 最后卖出的回报率
    c = ohlc_data.Close.values
    s.loc['Buy & Hold Return [%]'] = (c[-1] - c[0]) / c[0] * 100  # long-only return

    # 几何平均数
    gmean_day_return: float = 0
    day_returns = np.array(np.nan) # 每天相比于上一天的收益率
    annual_trading_days = np.nan # 交易日历
    if isinstance(index, pd.DatetimeIndex):
        day_returns = equity_df['Equity'].resample('D').last().dropna().pct_change()
        gmean_day_return = geometric_mean(day_returns)
        # 交易日历的计算, 这里没有考虑假期因素.
        # 大概就是先判断一下有没有可能周末开市, 如果有则每一天都是交易日, 否则只有工作日是交易日
        annual_trading_days = float(
            365 if index.dayofweek.to_series().between(5, 6).mean() > 2/7 * .6 else
            252)

    # empyrical 是一个风险指标模块

    # Annualized return and risk metrics are computed based on the (mostly correct)
    # assumption that the returns are compounded(复利). See: https://dx.doi.org/10.2139/ssrn.3054517
    # Our annualized return matches `empyrical.annual_return(day_returns)` whereas
    # our risk doesn't; they use the simpler approach below.
    annualized_return = (1 + gmean_day_return)**annual_trading_days - 1
    s.loc['Return (Ann.) [%]'] = annualized_return * 100
    # 波动率计算, TODO: 再看看这块公式
    s.loc['Volatility (Ann.) [%]'] = np.sqrt((day_returns.var(ddof=int(bool(day_returns.shape))) + (1 + gmean_day_return)**2)**annual_trading_days - (1 + gmean_day_return)**(2*annual_trading_days)) * 100  # noqa: E501
    # s.loc['Return (Ann.) [%]'] = gmean_day_return * annual_trading_days * 100
    # s.loc['Risk (Ann.) [%]'] = day_returns.std(ddof=1) * np.sqrt(annual_trading_days) * 100

    # Our Sharpe mismatches `empyrical.sharpe_ratio()` because they use arithmetic mean return
    # and simple standard deviation
    s.loc['Sharpe Ratio'] = (s.loc['Return (Ann.) [%]'] - risk_free_rate) / (s.loc['Volatility (Ann.) [%]'] or np.nan)  # noqa: E501
    # Our Sortino mismatches `empyrical.sortino_ratio()` because they use arithmetic mean return
    s.loc['Sortino Ratio'] = (annualized_return - risk_free_rate) / (np.sqrt(np.mean(day_returns.clip(-np.inf, 0)**2)) * np.sqrt(annual_trading_days))  # noqa: E501
    max_dd = -np.nan_to_num(dd.max())
    s.loc['Calmar Ratio'] = annualized_return / (-max_dd or np.nan)
    s.loc['Max. Drawdown [%]'] = max_dd * 100
    s.loc['Avg. Drawdown [%]'] = -dd_peaks.mean() * 100
    s.loc['Max. Drawdown Duration'] = _round_timedelta(dd_dur.max())
    s.loc['Avg. Drawdown Duration'] = _round_timedelta(dd_dur.mean())
    s.loc['# Trades'] = n_trades = len(trades_df)
    win_rate = np.nan if not n_trades else (pl > 0).mean()
    s.loc['Win Rate [%]'] = win_rate * 100
    s.loc['Best Trade [%]'] = returns.max() * 100
    s.loc['Worst Trade [%]'] = returns.min() * 100
    mean_return = geometric_mean(returns)
    s.loc['Avg. Trade [%]'] = mean_return * 100
    s.loc['Max. Trade Duration'] = _round_timedelta(durations.max())
    s.loc['Avg. Trade Duration'] = _round_timedelta(durations.mean())
    s.loc['Profit Factor'] = returns[returns > 0].sum() / (abs(returns[returns < 0].sum()) or np.nan)  # noqa: E501
    s.loc['Expectancy [%]'] = returns.mean() * 100
    s.loc['SQN'] = np.sqrt(n_trades) * pl.mean() / (pl.std() or np.nan)
    s.loc['Kelly Criterion'] = win_rate - (1 - win_rate) / (pl[pl > 0].mean() / -pl[pl < 0].mean())

    s.loc['_strategy'] = strategy_instance
    s.loc['_equity_curve'] = equity_df
    s.loc['_trades'] = trades_df

    s = _Stats(s)
    return s


class _Stats(pd.Series):
    def __repr__(self):
        # Prevent expansion due to _equity and _trades dfs
        with pd.option_context('max_colwidth', 20):
            return super().__repr__()
