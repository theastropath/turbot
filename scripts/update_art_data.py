#!/usr/bin/env python3

import csv
import os
import re
from enum import Enum
from pathlib import Path

import requests
from bs4 import BeautifulSoup

page = requests.get("https://animalcrossing.fandom.com/wiki/Forgery_(New_Horizons)")
tree = BeautifulSoup(page.content, "lxml")


class ArtType(Enum):
    PAINTING = 1
    SCULPTURE = 2


def clean(item):
    return re.sub("^-$", "0", item.strip().replace("✓", "1"))


def ingest(writer, art_type):
    def data_from(item):
        img = item.find("img")
        if img:
            if "data-src" in img.attrs:
                return img["data-src"]
            return img["src"]
        return item.text

    def unthumbnail(url):
        if url.startswith("http"):
            newurl = url.split("revision/latest")[0] + "revision/latest"
            return newurl
        return url

    table_tag = tree.select("table")[art_type.value]
    tab_data = [
        [data_from(item) for item in row_data.select("td")]
        for row_data in table_tag.select("tr")
    ]

    for row in range(1, len(tab_data)):
        data = [clean(i) for i in tab_data[row]]
        if data:
            dbentry = [
                data[0].lower().replace("\n", " "),
                data[1] != "N/A",
                data[3],
                unthumbnail(data[1]),
                unthumbnail(data[2]),
            ]
            writer.writerow(dbentry)


with open(Path("src") / "turbot" / "data" / "art.csv", "w", newline="") as out:
    writer = csv.writer(out, lineterminator=os.linesep)
    writer.writerow(
        ["name", "has_fake", "fake_description", "fake_image_url", "real_image_url",]
    )
    ingest(writer, ArtType.PAINTING)
    ingest(writer, ArtType.SCULPTURE)
