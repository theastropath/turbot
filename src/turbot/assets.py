import re
from os.path import dirname, realpath
from pathlib import Path
from string import Template

import pandas as pd
from unidecode import unidecode
from yaml import load

try:
    from yaml import CLoader as Loader
except ImportError:  # pragma: no cover
    from yaml import Loader

PACKAGE_ROOT = Path(dirname(realpath(__file__)))
ASSETS_DIR = PACKAGE_ROOT / "assets"
STRINGS_DATA_FILE = ASSETS_DIR / "strings.yaml"


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
            file = ASSETS_DIR / f"{kind}.csv"
            data = pd.read_csv(file)
            items = frozenset(data.drop_duplicates(subset="name").name.tolist())
            all_items.update(items)
            self.asset[kind] = {"file": file, "data": data, "items": items}
        self.all_collectable_items = frozenset(all_items)

    def validate(self, items, *, kinds=None):
        def clean(item):
            return re.sub(" ?k.k. ?|'", "", unidecode(item).lower())

        items = [clean(item) for item in items]
        valid = {}
        for kind in self.collectables:
            if kinds and isinstance(kinds, list) and kind not in kinds:
                continue
            valid_items = set()
            for item in self.asset[kind]["items"]:
                cleaned_item = clean(item)
                if cleaned_item in items:
                    valid_items.add(item)
                    items.remove(cleaned_item)
            valid[kind] = valid_items
        return valid, items

    def files(self):  # pragma: no cover
        return [STRINGS_DATA_FILE] + [
            self.asset[kind]["file"] for kind in self.collectables
        ]

    def __getitem__(self, kind):
        return Asset(self.asset[kind])


def s(key, **kwargs):
    """Returns a string from strings.yaml asset with subsitutions."""
    if not s.strings:
        s.strings = load_strings()

    data = s.strings.get(key, "")
    assert data, f"error: missing strings key: {key}"
    return Template(data).substitute(kwargs)


s.strings = None
