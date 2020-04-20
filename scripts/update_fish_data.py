#!/usr/bin/env python3

import csv
import re
from enum import Enum
from pathlib import Path

import requests
from bs4 import BeautifulSoup

page = requests.get("https://animalcrossing.fandom.com/wiki/Fish_(New_Horizons)")
tree = BeautifulSoup(page.content, "lxml")


class Hemisphere(Enum):
    Northern = 1
    Southern = 2


def clean(item):
    return re.sub("^-$", "0", item.strip().replace("✓", "1"))


def ingest(writer, hemisphere):
    table_tag = tree.select("table")[hemisphere.value]
    tab_data = [
        [item.text for item in row_data.select("td")]
        for row_data in table_tag.select("tr")
    ]

    for row in range(1, len(tab_data)):
        data = [clean(i) for i in tab_data[row]]
        if data:
            corrected = [hemisphere.name, data[0], *[d.lower() for d in data[2:]]]
            writer.writerow(corrected)


with open(Path("turbot") / "data" / "fish.csv", "w", newline="") as out:
    writer = csv.writer(out)
    writer.writerow(
        [
            "hemisphere",
            "name",
            "price",
            "location",
            "shadow",
            "time",
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        ]
    )
    ingest(writer, Hemisphere.Northern)
    ingest(writer, Hemisphere.Southern)
