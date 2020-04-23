#!/usr/bin/env python3

import csv
import re
from enum import Enum
from pathlib import Path

import requests
from bs4 import BeautifulSoup

page = requests.get("https://animalcrossing.fandom.com/wiki/Bugs_(New_Horizons)")
tree = BeautifulSoup(page.content, "lxml")


class Hemisphere(Enum):
    NORTHERN = 0
    SOUTHERN = 1


def clean(item):
    return re.sub("^-$", "0", item.strip().replace("✓", "1"))


def ingest(writer, hemisphere):
    table_tag = tree.select(".tabber table")[hemisphere.value]

    # there's some weird nesting on the northern bug table, keep any eye on this,
    # it may eventually be fixed on the wiki page and this code will break
    if hemisphere == Hemisphere.NORTHERN:
        table_tag = table_tag.select("table")[0]

    tab_data = [
        [item.text for item in row_data.select("td")]
        for row_data in table_tag.select("tr")
    ]

    for row in range(1, len(tab_data)):
        data = [clean(i) for i in tab_data[row]]
        if data:
            # lowercase all data and strip out the image column (2nd column)
            corrected = [d.lower() for d in [hemisphere.name, data[0], *data[2:]]]
            writer.writerow(corrected)


with open(Path("turbot") / "data" / "bugs.csv", "w", newline="") as out:
    writer = csv.writer(out)
    writer.writerow(
        [
            "hemisphere",
            "name",
            "price",
            "location",
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
    ingest(writer, Hemisphere.NORTHERN)
    ingest(writer, Hemisphere.SOUTHERN)
