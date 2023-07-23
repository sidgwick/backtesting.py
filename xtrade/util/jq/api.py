import os
import akshare as ak
import pandas as pd

from ..xcache import pickle_cache


# 聚宽 API 模拟补充
@pickle_cache
def attribute_history(
    security,
    count,
    unit="1d",
    fields=None,
    skip_paused=True,
    df=True,
    fq="pre",
) -> pd.DataFrame:
    default_fields = ["open", "close", "high", "low", "volume", "money"]
    fields = fields if fields is not None else default_fields

    data_path = f"learn/data/{security}_{unit}.csv"
    if os.path.isfile(data_path):
        df = pd.read_csv(data_path, index_col=0)
    else:
        df = ak.stock_zh_index_daily(symbol=security)
        df = df.set_index(keys=["date"], drop=True)
        # https://www.delftstack.com/howto/python-pandas/pandas-use-rolling-apply/
        df.columns.name = df.index.name
        df.index.name = None

    return df[len(df) - count :]
