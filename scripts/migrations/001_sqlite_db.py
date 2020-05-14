#!/usr/bin/env python3

# NOTE: You do not need to run this script by hand, turbot will automatically apply it.

"""Migrates old .csv db files into sqlite storage."""

import csv
from pathlib import Path

from turbot import Turbot

DB_DIR = Path(".") / "db"
DB_FILE = DB_DIR / "turbot.db"
DATA_TYPES = ["art", "bugs", "fish", "fossils", "songs", "users", "prices"]

# blow away any random db that exists, it should all come from pre-existing csv files
if DB_FILE.exists():
    DB_FILE.unlink()

# start up a client so we can get a connection to the db
client = Turbot()
c = client.data.conn


def migrate(dtype):
    db_file_path = DB_DIR / f"{dtype}.csv"
    if not db_file_path.exists():
        return
    with open(db_file_path, "r", newline="") as db_file:
        reader = csv.reader(db_file)
        next(reader)  # skip header
        for row in reader:
            placeholders = ",".join(["?"] * len(row))
            columns = ",".join(client.data.columns[dtype])
            c.execute(f"INSERT INTO {dtype} ({columns}) VALUES ({placeholders})", row)


for dtype in DATA_TYPES:
    migrate(dtype)
