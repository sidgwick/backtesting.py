import os
import pickle
import functools

import pandas as pd


def pd_pickle_cache(f):
    @functools.wraps(f)
    def _(*args, **kwargs):
        a = "".join([str(v) for v in args])
        b = "".join([f"{k}:{v}" for k, v in kwargs.items()])
        pickle_key = f"data/pickle/{f.__name__}_pickle_{a}_{b}"

        try:
            data = pd.read_pickle(pickle_key)
            if len(data):
                return data
        except:
            pass

        data = f(*args, **kwargs)
        data.to_pickle(pickle_key)
        return data

    return _


def pickle_cache(f):
    @functools.wraps(f)
    def _(*args, **kwargs):
        a = "".join([str(v) for v in args])
        b = "".join([f"{k}:{v}" for k, v in kwargs.items()])

        pickle_key = f"data/pickle/{f.__name__}_pickle_{a}_{b}"

        if os.path.isfile(pickle_key):
            fh = open(pickle_key, "rb")
            return pickle.load(fh)

        data = f(*args, **kwargs)

        fh = open(pickle_key, "wb")
        pickle.dump(data, fh)
        return data

    return _
