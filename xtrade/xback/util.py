from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Optional, cast

from collections import OrderedDict
import warnings
import numpy as np
import pandas as pd

# if TYPE_CHECKING:
#     from main import Main


class _Array(np.ndarray):
    """ndarray extended to supply .name and other
    arbitrary properties in ._opts dict.
    """

    def __new__(cls, array, *, name=None, **kwargs):
        obj = np.asarray(array).view(cls)
        obj.name = name or array.name
        obj._opts = kwargs
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self.name = getattr(obj, "name", "")
        self._opts = getattr(obj, "_opts", {})

    # Make sure properties name and _opts are carried over
    # when (un-)pickling.
    def __reduce__(self):
        value = super().__reduce__()
        return value[:2] + (value[2] + (self.__dict__,),)

    def __setstate__(self, state):
        self.__dict__.update(state[-1])
        super().__setstate__(state[:-1])

    def __bool__(self):
        try:
            return bool(self[-1])
        except IndexError:
            return super().__bool__()

    def __float__(self):
        try:
            return float(self[-1])
        except IndexError:
            return super().__float__()

    @property
    def s(self) -> pd.Series:
        values = np.atleast_2d(self)
        index = self._opts["index"][: values.shape[1]]
        return pd.Series(values[0], index=index, name=self.name)

    @property
    def df(self) -> pd.DataFrame:
        values = np.atleast_2d(np.asarray(self))
        index = self._opts["index"][: values.shape[1]]
        df = pd.DataFrame(values.T, index=index, columns=[self.name] * len(values))
        return df


class Indicator(_Array):
    pass


class Data:
    def __init__(self, data: OrderedDict[str, pd.DataFrame]):
        _data: OrderedDict[str, _Data] = {}
        for k, v in data.items():
            _data[k] = _Data(v)

        self._data = _data
        self._index: int = 0

    def first(self) -> pd.DataFrame:
        return next(iter(self._data.items()))

    def get(self, name: str) -> pd.DataFrame:
        return self._data[name]

    def set_length(self, length: int):
        self._index = length
        for k, v in self._data.items():
            v._set_length(length)


class _Data:
    """
    A data array accessor. Provides access to OHLCV "columns"
    as a standard `pd.DataFrame` would, except it's not a DataFrame
    and the returned "series" are _not_ `pd.Series` but `np.ndarray`
    for performance reasons.
    """

    def __init__(self, df: pd.DataFrame):
        self.__df = df
        self.__i = len(df)
        self.__pip: Optional[float] = None
        self._indicator: Dict[str, pd.DataFrame] = {}
        self.__cache: Dict[str, _Array] = {}
        self.__arrays: Dict[str, _Array] = {}
        self._update()

    # Make pickling in Backtest.optimize() work with our catch-all __getattr__
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def __getitem__(self, item):
        return self.__get_array(item)

    def __getattr__(self, item):
        try:
            return self.__get_array(item)
        except KeyError:
            raise AttributeError(f"Column '{item}' not in data") from None

    def _set_length(self, i):
        self.__i = i
        self.__cache.clear()

    def _update(self):
        index = self.__df.index.copy()
        self.__arrays = {col: _Array(arr, index=index) for col, arr in self.__df.items()}
        # Leave index as Series because pd.Timestamp nicer API to work with
        self.__arrays["__index"] = index

    def __repr__(self):
        i = min(self.__i, len(self.__df)) - 1
        index = self.__arrays["__index"][i]
        items = ", ".join(f"{k}={v}" for k, v in self.__df.iloc[i].items())
        return f"<Data i={i} ({index}) {items}>"

    def __len__(self):
        return self.__i

    def aspect(self, idx: int) -> pd.Series:
        """取到 idx 指定的那个 row 的相关数据切面"""
        idx = idx if idx > 0 else (self.__i + idx + 1)
        assert idx >= 0
        return self.__df.iloc[idx]

    def set_indicator(self, name: str, i: pd.DataFrame):
        self._indicator[name] = i

    def get_indicator(self, name: str) -> pd.DataFrame:
        return self._indicator[name]

    @property
    def df(self) -> pd.DataFrame:
        # 返回截止到 self.__i 的 DataFrame 数据
        return self.__df.iloc[: self.__i] if self.__i < len(self.__df) else self.__df

    @property
    def pip(self) -> float:
        if self.__pip is not None:
            return self.__pip

        tmp = [len(s.partition(".")[-1]) for s in self.__arrays["Close"].astype(str)]
        self.__pip = float(10 ** -np.median(tmp))
        return self.__pip

    def __get_array(self, key) -> _Array:
        arr = self.__cache.get(key)
        if arr is None:
            arr = self.__cache[key] = cast(_Array, self.__arrays[key][: self.__i])
        return arr

    @property
    def Open(self) -> _Array:
        return self.__get_array("Open")

    @property
    def High(self) -> _Array:
        return self.__get_array("High")

    @property
    def Low(self) -> _Array:
        return self.__get_array("Low")

    @property
    def Close(self) -> _Array:
        return self.__get_array("Close")

    @property
    def Volume(self) -> _Array:
        return self.__get_array("Volume")

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.__get_array("__index")
