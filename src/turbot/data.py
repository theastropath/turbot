import sqlite3

import pandas as pd

# from io import StringIO
# from pathlib import Path


class StoreFrame(pd.DataFrame):
    _metadata = ["data_type"]

    @property
    def _constructor(self):
        s = StoreFrame
        s.data_type = self.data_type
        return s

    @property
    def _constructor_sliced(self):
        s = pd.Series
        s.data_type = self.data_type
        return s


class Data:
    """Persistent and in-memory store for user data."""

    def __init__(self, *, db_file):
        self.db_file = db_file
        self.conn = sqlite3.connect(self.db_file)
        c = self.conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS fossils (
                author INTEGER NOT NULL,
                name TEXT NOT NULL);
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS art (
                author INTEGER NOT NULL,
                name TEXT NOT NULL);
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS fish (
                author INTEGER NOT NULL,
                name TEXT NOT NULL);
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS bugs (
                author INTEGER NOT NULL,
                name TEXT NOT NULL);
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS songs (
                author INTEGER NOT NULL,
                name TEXT NOT NULL);
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS prices (
                author INTEGER NOT NULL,
                kind TEXT NOT NULL,
                price INTEGER NOT NULL,
                timestamp TEXT NOT NULL);
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                author INTEGER NOT NULL,
                hemisphere TEXT,
                timezone TEXT,
                island TEXT,
                friend TEXT,
                fruit TEXT,
                nickname TEXT,
                creator TEXT);
            """
        )

        self.config = {
            "fossils": {"columns": {"author": "int64", "name": "str"},},
            "art": {"columns": {"author": "int64", "name": "str"}},
            "fish": {"columns": {"author": "int64", "name": "str"}},
            "bugs": {"columns": {"author": "int64", "name": "str"}},
            "songs": {"columns": {"author": "int64", "name": "str"}},
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
        # for data_type in self.config.keys():
        #     self.config[data_type]["file"] = db_dir / f"{data_type}.csv"
        self.in_memory = {data_type: None for data_type in self.config.keys()}

    # def file(self, data_type):
    #     return self.config[data_type]["file"]

    def commit(self, sf):
        sf.to_sql(sf.data_type, self.conn, if_exists="replace", index=False)
        self.in_memory[sf.data_type] = sf
        # f_out = self.config[sf.data_type]["file"]
        # sf.to_csv(f_out, index=False)
        # self.in_memory[sf.data_type] = sf

    def __getattr__(self, attr):
        # if attr not in self.config.keys():
        if attr not in ["fossils", "art", "fish", "bugs", "songs", "prices", "users"]:
            raise RuntimeError(f"there is no data store for {attr}")

        cfg = self.config[attr]
        columns = cfg["columns"]

        def add_metadata(df):
            sf = StoreFrame(df.fillna("").astype(columns))
            # sf = StoreFrame(df)
            sf.data_type = attr
            return sf

        df = self.in_memory[attr]
        if df is not None:
            return add_metadata(df)

        parse_dates = ["timestamp"] if attr == "prices" else None
        print("############# parse_dates:", parse_dates)
        df = pd.read_sql_query(
            f"SELECT * FROM {attr};", self.conn, parse_dates=parse_dates
        )

        # f_in = cfg["file"]
        # cnames = list(columns.keys())
        # dtypes = list(columns.values())
        # if Path(f_in).exists():
        #     parse_dates = any("datetime" in dtype for dtype in dtypes)
        #     df = pd.read_csv(f_in, names=cnames, parse_dates=parse_dates, skiprows=1)
        # else:
        #     df = pd.read_csv(StringIO(""), names=cnames, dtype=columns)

        return add_metadata(df)
