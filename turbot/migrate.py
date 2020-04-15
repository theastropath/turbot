#!/usr/bin/env python

from pathlib import Path

import pandas as pd

from turbot import FOSSILS, load_fossils, load_prices, save_fossils, save_prices

PRICES_DIR = "prices"
FOSSILS_DIR = "fossils"

Path(PRICES_DIR).mkdir(exist_ok=True)
Path(FOSSILS_DIR).mkdir(exist_ok=True)


def migrate_prices():
    prices = load_prices()
    for user_file in Path(PRICES_DIR).iterdir():
        author = int(user_file.stem)
        columns = ["kind", "price", "timestamp"]
        user_data = pd.read_csv(user_file, header=0, names=columns, sep=" ")
        user_data.timestamp = pd.to_datetime(user_data.timestamp, unit="s", utc=True)
        user_data.insert(0, "author", author)
        prices = prices.append(user_data, ignore_index=True)
    prices = prices.drop_duplicates()
    save_prices(prices)


def migrate_fossils():
    fossils = load_fossils()
    for user_file in Path(FOSSILS_DIR).iterdir():
        author = int(user_file.stem)
        with user_file.open() as fd:
            remaining = set(line.strip() for line in fd.readlines())
        collected = FOSSILS - remaining
        user_rows = [[author, fossil] for fossil in collected]
        user_data = pd.DataFrame(columns=fossils.columns, data=user_rows)
        fossils = fossils.append(user_data, ignore_index=True)
    fossils = fossils.drop_duplicates()
    save_fossils(fossils)


def main():
    """Migrates any existing 1.0 formatted data to 2.0 data."""
    migrate_prices()
    migrate_fossils()


if __name__ == "__main__":
    main()
