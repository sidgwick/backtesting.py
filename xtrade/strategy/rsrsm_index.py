from typing import *

import numpy as np
from numpy_ext import rolling_apply
import pandas as pd

from util.model import ols, zscore


def RSRSM_index(data: pd.DataFrame, n: int, m: int, bias_n: int, md: int, latest_n: int = None) -> pd.DataFrame:
    """RSRS 择时 + 动量轮动策略中用到的指标计算,

    这里前面 max(n, m, bias_n, md) 天的数据不会被计算, 因此回测的时候
    需要提供额外的 max(n, m, bias_n, md) 条历史数据以确保回测日期范围内都是有数据的

    :param data: 要计算的数据, 最少需要包含 'high', 'low', 'close' 三列
    :param n: RSRS 斜率拟合窗口大小(日线就是天数)
    :param m: RSRS 标准分计算使用到的斜率窗口大小
    :param bias_n: 动量乖离因子均值窗口大小
    :param md: momentum day, 动量因子斜率拟合窗口大小
    :param latest_n: 只计算最后 n 行数据, 调用者需保证 n 之前的数据已经计算完毕
    :return:
    """
    l = len(data)
    _data = data.copy()

    # 算一下从哪里开始计算数据比较合适
    ols_start, zscore_start, motion_start = n, m, md
    if latest_n is not None:
        _start = l - latest_n +1
        ols_start, zscore_start, motion_start = _start, _start, _start

    ols_res = [[np.nan, np.nan, np.nan]] * (ols_start - 1)
    ols_res += [ols(data.low[i - n:i], data.high[i - n:i]) for i in range(ols_start, l + 1)]

    df = pd.DataFrame(ols_res, columns=["intercept", "slope", "r2"], index=data.index)
    _data = _data.combine_first(df)

    # 计算标准分
    zscore_res = [np.nan] * (zscore_start - 1)
    zscore_res += [zscore(_data.slope[i - m:i]) * _data.r2[i - 1] for i in range(zscore_start, l + 1)]

    df = pd.DataFrame(zscore_res, columns=["zscore"], index=data.index)
    _data = _data.combine_first(df)

    # 动量因子
    bias = (_data.close / _data.close.rolling(bias_n).mean())  # 乖离因子
    bias_res = [np.nan] * (motion_start - 1)
    bias_res += [ols(np.arange(md), bias[i - md:i] / bias[i - 1])[1] for i in range(motion_start, l + 1)]

    df = pd.DataFrame(bias_res[:l], columns=["motion"], index=data.index)
    _data = _data.combine_first(df)

    return _data
