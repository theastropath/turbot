#!/usr/bin/env python3

from pathlib import Path

import requests
from bs4 import BeautifulSoup

page = requests.get("https://animalcrossing.fandom.com/wiki/Fossils_(New_Horizons)")
tree = BeautifulSoup(page.content, "lxml")


with open(Path("src") / "turbot" / "assets" / "fossils.csv", "w", newline="") as out:

    def data_from(item):
        cells = item.select("td")
        if not cells or len(cells) <= 1:
            return None
        return cells[0].select("a")[0].text.strip()

    # stand-alone fossils
    table_tag = tree.select("table")[3]
    alone = set(filter(None, (data_from(row) for row in table_tag.select("tr"))))

    # multi-part fossils
    table_tag = tree.select("table")[5]
    rows = table_tag.select("tr")
    multi = set(filter(None, [data_from(row) for row in rows[2:]]))

    out.write("name\n")
    for title in sorted(alone | multi):
        out.write(f"{title.lower()}\n")
