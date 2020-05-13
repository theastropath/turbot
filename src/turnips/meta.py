#!/usr/bin/env python3

from __future__ import annotations

from typing import Generator, Iterable, Optional

from turnips.model import Model
from turnips.multi import (
    MultiModel,
    TripleModels,
    SpikeModels,
    DecayModels,
    BumpModels,
)
from turnips.ttime import AnyTime


class MetaModel(MultiModel):
    # This is either a horrid abuse of OO, or brilliant.
    # We override only methods that refer to self._models directly;
    # otherwise, self.models takes care of it.

    def __init__(self,
                 initial: Optional[int],
                 groups: Iterable[MultiModel]):
        super().__init__(initial)
        self._children = list(groups)

    @classmethod
    def blank(cls, initial: Optional[int] = None) -> MetaModel:
        groups = [
            TripleModels(initial),
            SpikeModels(initial),
            DecayModels(initial),
            BumpModels(initial),
        ]
        return cls(initial, groups)

    @property
    def models(self) -> Generator[Model, None, None]:
        for child in self._children:
            yield from child.models

    def fix_price(self, time: AnyTime, price: int) -> None:
        for child in self._children:
            child.fix_price(time, price)

    def __bool__(self) -> bool:
        return len(self) != 0

    def __len__(self) -> int:
        return sum(len(child) for child in self._children)

    def detailed_report(self) -> None:
        self._children = list(filter(None, self._children))

        if not self._children:
            print("--- No Viable Model Groups ---")
            return

        if len(self._children) == 1:
            self._children[0].report(show_summary=False)
            return

        print("Meta-Analysis: ")
        print("total possible models: {}".format(len(self)))
        print('----------')
        for child in self._children:
            if child:
                child.report()
                print('----------')
