from __future__ import annotations

import hashlib
import json
import logging
import pickle
from typing import Any, Dict, Generator, Optional

import pydantic
from pydantic import validator
import yaml

from turnips.model import ModelEnum
from turnips.meta import MetaModel
from turnips.multi import MultiModel, BumpModels
from turnips.ttime import TimePeriod

from turnips.plots import plot_models_range_interactive, plot_models_range_data, global_plot


# pylint: disable=too-few-public-methods, no-self-use, no-self-argument

class Base(pydantic.BaseModel):
    class Config:
        extra = 'forbid'


class IslandModel(Base):
    previous_week: ModelEnum = ModelEnum.unknown
    initial_week: bool = False
    timeline: Dict[TimePeriod, Optional[int]]

    @validator('previous_week', pre=True)
    def coerce_model(cls, value: Any) -> Any:
        if isinstance(value, str):
            return ModelEnum[value.lower()]
        return value

    @validator('timeline', pre=True)
    def normalize(cls, value: Any) -> Any:
        if isinstance(value, Dict):
            return {
                TimePeriod.normalize(key): price
                for key, price in value.items()
            }
        return value

    @validator('initial_week')
    def cajole(cls, initial_week: bool, values: Dict[str, Any]) -> Any:
        if values['previous_week'] != ModelEnum.unknown and initial_week:
            raise ValueError("Cannot set initial_week = True when previous_week is set")
        return initial_week

    @property
    def base_price(self) -> Optional[int]:
        return self.timeline.get(TimePeriod.Sunday_AM, None)


class ArchipelagoModel(Base):
    islands: Dict[str, IslandModel]


class Island:
    def __init__(self, name: str, data: IslandModel):
        self._name = name
        self._data = data
        self._models: Optional[MultiModel] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def previous_week(self) -> ModelEnum:
        return self._data.previous_week

    @property
    def initial_week(self) -> bool:
        return self._data.initial_week

    @property
    def model_group(self) -> MultiModel:
        if self._models is None:
            self.process()
            assert self._models is not None
        return self._models

    def process(self) -> None:
        logging.info(f" == {self.name} island == ")

        base = self._data.base_price
        if self.initial_week:
            self._models = BumpModels()
        else:
            self._models = MetaModel.blank(base)

        logging.info(f"  (%d models)  ", len(self._models))

        for time, price in self._data.timeline.items():
            if price is None:
                continue
            if time.value < TimePeriod.Monday_AM.value:
                continue
            logging.info(f"[{time.name}]: fixing price @ {price}")
            self._models.fix_price(time, price)

    def plot(self):
        return plot_models_range_data(
            self.name,
            list(self.model_group.models),
            self.previous_week
        )

    def checksum(self):
        data = pickle.dumps((self._data.dict(), self.name))
        return hashlib.md5(data).hexdigest()


class Archipelago:
    def __init__(self, data: ArchipelagoModel):
        self._data = data
        self._islands = {name: Island(name, idata) for name, idata
                         in self._data.islands.items()}

    @classmethod
    def parse_obj(cls, obj: Dict[Any, Any]) -> Archipelago:
        return cls(ArchipelagoModel.parse_obj(obj))

    @classmethod
    def load_json(cls, data: str) -> Archipelago:
        jdata = json.loads(data)
        return cls.parse_obj(jdata)

    @classmethod
    def load_yaml(cls, data: str) -> Archipelago:
        ydata = yaml.safe_load(data)
        return cls.parse_obj(ydata)

    @classmethod
    def load_file(cls, filename: str) -> Archipelago:
        if '.yml' in filename or '.yaml' in filename:
            with open(filename, "r") as infile:
                ydoc = yaml.safe_load(infile)
                return cls.parse_obj(ydoc)

        return cls(ArchipelagoModel.parse_file(filename))

    @property
    def groups(self) -> Generator[MultiModel, None, None]:
        for island in self._islands.values():
            yield island.model_group

    @property
    def islands(self) -> Generator[Island, None, None]:
        for island in self._islands.values():
            yield island

    def summary(self) -> None:
        for island in self.islands:
            print(f"{island.name}")
            print('-' * len(island.model_group))
            print('')
            island.model_group.report()
            print('')

        # The initial doesn't matter here, it's ignored.
        print('Archipelago Summary')
        print('-' * 80)
        archipelago = MetaModel(-1, self.groups)
        archipelago.summary()
        print('-' * 80)

    def plot(self) -> None:
        for island in self.islands:
            plot_models_range_interactive(island.name,
                                          list(island.model_group.models),
                                          island.previous_week)
        global_plot(self.groups)
