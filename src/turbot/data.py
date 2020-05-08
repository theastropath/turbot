from io import StringIO
from pathlib import Path

import pandas as pd


class ProxiedCall:
    """Wrapper for DataFrame calls."""

    def __init__(self, store, method_name):
        self.store = store
        self.method_name = method_name

    def __call__(self, *args, **kwargs):
        rvalue = getattr(self.store._trs_df, self.method_name)(*args, **kwargs)

        if isinstance(rvalue, pd.DataFrame):
            return Store(data=self.store._trs_data, name=self.store._trs_name, df=rvalue)
        else:
            return rvalue

    def __getitem__(self, key):
        rvalue = getattr(self.store._trs_df, self.method_name).__getitem__(key)

        if isinstance(rvalue, pd.DataFrame):
            return Store(data=self.store._trs_data, name=self.store._trs_name, df=rvalue)
        else:
            return rvalue


class Store:
    """Wrapper for DataFrames."""

    # Note, if possible this should be refactored to use _metadata support rather
    # than this proxy object. For now, this seems to work fine. Details on _metadata
    # support: https://pandas.pydata.org/pandas-docs/stable/development/extending.html

    def __init__(self, *_, data, name, df):
        self._trs_data = data
        self._trs_name = name
        self._trs_df = df

    def commit(self):
        f_out = self._trs_data.config[self._trs_name]["file"]
        self._trs_df.to_csv(f_out, index=False)
        self._trs_data.in_memory[self._trs_name] = self._trs_df

    def add(self, *args):
        row = pd.DataFrame(columns=self._trs_df.columns, data=[args])
        self._trs_df = self._trs_df.append(row, ignore_index=True)

    def __getattr__(self, attr):
        rvalue = getattr(self._trs_df, attr)

        if callable(rvalue):
            return ProxiedCall(self, attr)
        elif isinstance(rvalue, pd.DataFrame):
            return Store(data=self._trs_data, name=self._trs_name, df=rvalue)
        else:
            return rvalue

    def __getitem__(self, key):
        rvalue = self._trs_df.__getitem__(key)

        if callable(rvalue):
            return ProxiedCall(self, key)
        if isinstance(rvalue, pd.DataFrame):
            return Store(data=self._trs_data, name=self._trs_name, df=rvalue)
        else:
            return rvalue

    def __setitem__(self, key, value):
        self._trs_df[key] = value


class Data:
    """Persistent and in-memory store for user data."""

    def __init__(self, *_, db_dir):
        self.db_dir = db_dir
        self.config = {
            "fossils": {"columns": {"author": "int64", "name": "str"},},
            "art": {"columns": {"author": "int64", "name": "str"}},
            "fish": {"columns": {"author": "int64", "name": "str"}},
            "bugs": {"columns": {"author": "int64", "name": "str"}},
            "prices": {
                "columns": {
                    "author": "int64",
                    "kind": "object",
                    "price": "int64",
                    "timestamp": "datetime64[ns, UTC]",
                },
            },
            "users": {
                "columns": {
                    "author": "int64",
                    "hemisphere": "str",
                    "timezone": "str",
                    "island": "str",
                    "friend": "str",
                    "fruit": "str",
                    "nickname": "str",
                    "creator": "str",
                },
            },
        }
        for data_type in self.config.keys():
            self.config[data_type]["file"] = db_dir / f"{data_type}.csv"
        self.in_memory = {data_type: None for data_type in self.config.keys()}

    def file(self, data_type):
        return self.config[data_type]["file"]

    def __getattr__(self, attr):
        if attr not in self.config.keys():
            raise RuntimeError(f"there is no data store for {attr}")

        cfg = self.config[attr]
        columns = cfg["columns"]

        def build_store(df):
            df = df.fillna("").astype(columns)
            return Store(data=self, name=attr, df=df)

        df = self.in_memory[attr]
        if df is not None:
            return build_store(df)

        f_in = cfg["file"]
        cnames = list(columns.keys())
        dtypes = list(columns.values())
        if Path(f_in).exists():
            parse_dates = any("datetime" in dtype for dtype in dtypes)
            df = pd.read_csv(f_in, names=cnames, parse_dates=parse_dates, skiprows=1)
        else:
            df = pd.read_csv(StringIO(""), names=cnames, dtype=columns)

        return build_store(df)
