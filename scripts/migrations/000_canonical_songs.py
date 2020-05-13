#!/usr/bin/env python3

# NOTE: You do not need to run this script by hand, turbot will automatically apply it.

"""
We once stored names unidecoded and lowercased. We should have stored them in their
canonical form, with capitalization and accents preserved. This script will migrate
rows from user db files from before this was changed.
"""

import csv
from pathlib import Path

from unidecode import unidecode


def munge(item):
    return unidecode(item).lower()


munged_to_canonical = {}

ASSET_FILE = Path("src") / "turbot" / "data" / "songs.csv"
WORKSPACE_FILE = Path("db") / "songs.csv.migration"
DB_FILE = Path("db") / "songs.csv"

if DB_FILE.exists():
    with open(ASSET_FILE, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            name = row[0]
            munged_to_canonical[munge(name)] = name

    with open(WORKSPACE_FILE, "w") as out, open(DB_FILE, "r", newline="") as db:
        reader = csv.reader(db)
        for row in reader:
            author, name = row
            name = munged_to_canonical.get(name, name)
            out.write(f"{author},{name}\n")

    WORKSPACE_FILE.rename(DB_FILE)
