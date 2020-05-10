from os.path import dirname, realpath
from pathlib import Path
from string import Template

import pandas as pd
from yaml import load

try:
    from yaml import CLoader as Loader
except ImportError:  # pragma: no cover
    from yaml import Loader

PACKAGE_ROOT = Path(dirname(realpath(__file__)))
DATA_DIR = PACKAGE_ROOT / "data"
STRINGS_DATA_FILE = DATA_DIR / "strings.yaml"


def load_strings():
    with open(STRINGS_DATA_FILE) as f:
        return load(f, Loader=Loader)


class Asset:
    def __init__(self, info):
        self.info = info

    @property
    def all(self):
        return self.info["items"]

    @property
    def data(self):
        return self.info["data"]


class Assets:
    def __init__(self):
        self.collectables = ["fossils", "bugs", "fish", "art", "songs"]
        all_items = set()
        self.asset = {}
        for kind in self.collectables:
            file = DATA_DIR / f"{kind}.csv"
            data = pd.read_csv(file)
            items = frozenset(data.drop_duplicates(subset="name").name.tolist())
            all_items.update(items)
            self.asset[kind] = {"file": file, "data": data, "items": items}
        self.all_collectable_items = frozenset(all_items)

    def validate(self, items, *_, kinds=None):
        valid = {}
        for kind in self.collectables:
            if kinds and isinstance(kinds, list) and kind not in kinds:
                continue
            valid[kind] = items.intersection(self.asset[kind]["items"])
        invalid = items.difference(self.all_collectable_items)
        return valid, invalid

    def files(self):  # pragma: no cover
        return [STRINGS_DATA_FILE] + [
            self.asset[kind]["file"] for kind in self.collectables
        ]

    def __getitem__(self, kind):
        return Asset(self.asset[kind])


def s(key, **kwargs):
    """Returns a string from data/strings.yaml with subsitutions."""
    if not s.strings:
        s.strings = load_strings()

    data = s.strings.get(key, "")
    assert data, f"error: missing strings key: {key}"
    return Template(data).substitute(kwargs)


s.strings = None
