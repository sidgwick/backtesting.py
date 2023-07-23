from typing import *

import akshare as ak
import numpy as np
import pandas as pd
from statsmodels.formula import api as sml

from .xcache import pickle_cache


# 最小二乘法拟合, 得到: 截距/斜率/R^2
def ols(
    x: Union[np.ndarray, pd.Series], y: Union[np.ndarray, pd.Series]
) -> Tuple[float, float, float]:
    # 全部是 nan 的数据无法计算
    if np.isnan(x).all() or np.isnan(y).all():
        return (np.nan, np.nan, np.nan)

    model = sml.ols(formula="y~x", data={"x": x, "y": y})
    result = model.fit()

    intercept = result.params[0]
    slope = result.params[1]
    r2 = result.rsquared

    return (intercept, slope, r2)


# 标准分计算: (slope - mean) / std
def zscore(data: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
    mean = np.mean(data)
    std = np.std(data)
    return (data[-1] - mean) / std


# 对标准分在做一次最小二乘法拟合, 只要斜率
def zscore_slope(data: Union[np.ndarray, pd.Series]) -> float:
    y = data
    x = np.arange(len(data))
    intercept, slope, r2 = ols(x, y)
    return slope


def xols(data, prefix="", formula="", N=0):
    beta_key = "{}{}".format(prefix, "beta")
    r2_key = "{}{}".format(prefix, "R2")
    data[beta_key] = 0
    data[r2_key] = 0
    for i in range(1, len(data) - 1):
        df_ne = data.loc[i - N + 1 : i + 1, :]
        model = sml.ols(formula="high~low", data=df_ne)
        result = model.fit()

        data.loc[i + 1, beta_key] = result.params[1]
        data.loc[i + 1, r2_key] = result.rsquared

    return data


@pickle_cache
def ak_index_daily(start, end):
    data = ak.stock_zh_index_daily(symbol="sh000300")
    data = data[(start <= data["date"]) & (data["date"] <= end)]
    df = data.reset_index()
    return df


@pickle_cache
def hs300_ols(start, end, N):
    data = ak.stock_zh_index_daily(symbol="sh000300")
    data = data[(start <= data["date"]) & (data["date"] <= end)]
    HS300 = data.reset_index()

    # 斜率
    xols(HS300, prefix="", formula="high~low", N=N)
    HS300["beta"] = 0
    HS300["R2"] = 0
    for i in range(1, len(HS300) - 1):
        df_ne = HS300.loc[i - N + 1 : i + 1, :]
        model = sml.ols(formula="high~low", data=df_ne)
        result = model.fit()

        HS300.loc[i + 1, "beta"] = result.params[1]
        HS300.loc[i + 1, "R2"] = result.rsquared

    return HS300


def norm(s, m):
    res = (s - s.rolling(m).mean().shift(1)) / s.rolling(m).std().shift(1)
    with np.errstate(divide="ignore"):
        for i in range(m):
            a = s[i] - s[: i - 1].mean()
            b = s[: i - 1].std()
            res[i] = a / b

    return res


def getdata(start, end, N, M):
    HS300 = hs300_ols(start, end, N)

    # 日收益率
    HS300["ret"] = HS300.close.pct_change(1)

    # 标准分
    HS300["beta_norm"] = norm(HS300.beta, M)

    # 修正标准分
    HS300["RSRS_R2"] = HS300.beta_norm * HS300.R2

    # 右偏标准分
    HS300["beta_right"] = HS300.RSRS_R2 * HS300.beta

    # 考虑成交量对标准分的影响
    HS300["diff"] = HS300["open"] - HS300["close"]
    xols(HS300, prefix="diff_", formula="diff~volume", N=18)
    HS300["diff_norm"] = norm(HS300.diff_beta, 300)
    HS300["diff_RR2"] = HS300.diff_norm * HS300.diff_R2
    HS300["diff_right"] = HS300.diff_R2 * HS300.diff_beta

    HS300 = HS300.fillna(0)
    return HS300
