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
    NORTHERN = 1
    SOUTHERN = 2


def clean(item):
    return re.sub("^-$", "0", item.strip().replace("✓", "1"))


def ingest(writer, hemisphere):
    def data_from(item):
        img = item.find("img")
        if img:
            return img["data-src"]
        return item.text

    table_tag = tree.select("table")[hemisphere.value]
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
                *[d.lower() for d in data[2:]],
            ]
            writer.writerow(corrected)


with open(Path("src") / "turbot" / "data" / "fish.csv", "w", newline="") as out:
    writer = csv.writer(out)
    writer.writerow(
        [
            "hemisphere",
            "name",
            "image",
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
    ingest(writer, Hemisphere.NORTHERN)
    ingest(writer, Hemisphere.SOUTHERN)
