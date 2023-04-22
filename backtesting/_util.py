import warnings
from numbers import Number
from typing import Dict, List, Optional, Sequence, Union, cast

import numpy as np
import pandas as pd


def try_(lazy_func, default=None, exception=Exception):
    try:
        return lazy_func()
    except exception:
        return default


def _as_str(value) -> str:
    if isinstance(value, (Number, str)):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return 'df'
    name = str(getattr(value, 'name', '') or '')
    if name in ('Open', 'High', 'Low', 'Close', 'Volume'):
        return name[:1]
    if callable(value):
        name = getattr(value, '__name__', value.__class__.__name__).replace('<lambda>', 'λ')
    if len(name) > 10:
        name = name[:9] + '…'
    return name


def _as_list(value) -> List:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return list(value)
    return [value]


def _data_period(index) -> Union[pd.Timedelta, Number]:
    """Return data index period as pd.Timedelta
    TODO: 这个函数的具体用法, 后面看到了在说
    """
    values = pd.Series(index[-100:])
    # values.diff 用来计算序列元素和它的前一个元素之间的差值(结果的第一个值是 NaN)
    # dropna 去掉非数字
    # median 求出来中值(中位数)
    return values.diff().dropna().median()


class _Array(np.ndarray):
    """
    _Array 是一个子类化的 NumPy 数组对象.

    这个类是对 np.ndarray 的一层波安装, 提供了一些方便和 pandas 交互的功能.

    `*` 表示之后的参数都是关键字参数.

    ndarray extended to supply .name and other arbitrary properties
    in ._opts dict.
    """
    def __new__(cls, array, *, name=None, **kwargs):
        obj = np.asarray(array).view(cls)
        obj.name = name or array.name
        obj._opts = kwargs
        return obj

    def __array_finalize__(self, obj):
        '''这个函数是 numpy 使用的: https://numpy.org/doc/stable/user/basics.interoperability.html#returning-foreign-objects
        简单理解这个方法是在类实例化的时候被调用的, 这里给类追加了两个属性: name 和 _opts
        '''
        if obj is not None:
            self.name = getattr(obj, 'name', '')
            self._opts = getattr(obj, '_opts', {})

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

    def to_series(self):
        warnings.warn("`.to_series()` is deprecated. For pd.Series conversion, use accessor `.s`")
        return self.s

    @property
    def s(self) -> pd.Series:
        values = np.atleast_2d(self)
        index = self._opts['index'][:values.shape[1]]
        return pd.Series(values[0], index=index, name=self.name)

    @property
    def df(self) -> pd.DataFrame:
        # TODO: 确认一下当前的数据到底是什么, 为啥这里还需要转置?
        values = np.atleast_2d(np.asarray(self))
        index = self._opts['index'][:values.shape[1]]
        df = pd.DataFrame(values.T, index=index, columns=[self.name] * len(values))
        return df


class _Indicator(_Array):
    pass


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
        self.__cache: Dict[str, _Array] = {}
        self.__arrays: Dict[str, _Array] = {}
        self._update()

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
        '''假如 self.__df 是一个时间序列的话,
        这里 __arrays 就是各个 colums 的值, __arrays[__index] 是日期的值'''
        index = self.__df.index.copy()
        self.__arrays = {col: _Array(arr, index=index)
                         for col, arr in self.__df.items()}
        # Leave index as Series because pd.Timestamp nicer API to work with
        self.__arrays['__index'] = index

    def __repr__(self):
        i = min(self.__i, len(self.__df)) - 1
        index = self.__arrays['__index'][i]
        items = ', '.join(f'{k}={v}' for k, v in self.__df.iloc[i].items())
        return f'<Data i={i} ({index}) {items}>'

    def __len__(self):
        return self.__i

    @property
    def df(self) -> pd.DataFrame:
        # 返回截止到 self.__i 的 DataFrame 数据
        return (self.__df.iloc[:self.__i]
                if self.__i < len(self.__df)
                else self.__df)

    @property
    def pip(self) -> float:
        '''TODO: 啥是 pip
        pip 是外汇交易中的一个术语, 表示价格变动的最小单位. 它通常是货币对的小数点后第四位,
        例如 EUR/USD 的 pip 就是 0.0001, 在代码中, self.pip 是一个属性, 用于计算
        数据中 Close 列的 pip 值. 具体实现是通过计算 Close 列中小数点后位数的中位数来确定 pip 的值
        '''
        if self.__pip is None:
            self.__pip = float(10**-np.median([len(s.partition('.')[-1])
                                               for s in self.__arrays['Close'].astype(str)]))
        return self.__pip

    def __get_array(self, key) -> _Array:
        arr = self.__cache.get(key)
        if arr is None:
            arr = self.__cache[key] = cast(_Array, self.__arrays[key][:self.__i])
        return arr

    @property
    def Open(self) -> _Array:
        return self.__get_array('Open')

    @property
    def High(self) -> _Array:
        return self.__get_array('High')

    @property
    def Low(self) -> _Array:
        return self.__get_array('Low')

    @property
    def Close(self) -> _Array:
        return self.__get_array('Close')

    @property
    def Volume(self) -> _Array:
        return self.__get_array('Volume')

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.__get_array('__index')

    # Make pickling in Backtest.optimize() work with our catch-all __getattr__
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state
