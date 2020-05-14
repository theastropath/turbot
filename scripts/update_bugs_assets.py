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
    def data_from(item):
        img = item.find("img")
        if img:
            return img["data-src"]
        return item.text

    table_tag = tree.select(".tabber table")[hemisphere.value]

    # there's some weird nesting on the northern bug table, keep any eye on this,
    # it may eventually be fixed on the wiki page and this code will break
    if hemisphere == Hemisphere.NORTHERN:
        table_tag = table_tag.select("table")[0]

    tab_data = [
        [data_from(item) for item in row_data.select("td")]
        for row_data in table_tag.select("tr")
    ]

    for row in range(1, len(tab_data)):
        data = [clean(i) for i in tab_data[row]]
        if data:
            corrected = [
                hemisphere.name.lower(),
                data[0].lower(),
                data[1],
                int(data[2].replace(",", "")),
                *[d.lower() for d in data[3:]],
            ]
            writer.writerow(corrected)


with open(Path("src") / "turbot" / "assets" / "bugs.csv", "w", newline="") as out:
    writer = csv.writer(out)
    writer.writerow(
        [
            "hemisphere",
            "name",
            "image",
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
