#!/usr/bin/env python3

from __future__ import annotations

from collections import Counter
import enum
import math
import random
from typing import Dict, Generator, List, Optional, Type
from typing import Counter as CounterType

from turnips.price import Price
from turnips.modifier import (
    Modifier,
    SmallProfit,
    MediumProfit,
    LargeProfit,
    MediumLoss,
    WideLoss,
    RapidDecay,
    InitialDecay,
    SlowDecay,
    CappedPassthrough,
    Passthrough,
)
from turnips.ttime import AnyTime, TimePeriod

class ModelEnum(enum.Enum):
    unknown = -1
    triple = 0
    spike = 1
    decay = 2
    bump = 3

UNKNOWN = ModelEnum.unknown
TRIPLE = ModelEnum.triple
SPIKE = ModelEnum.spike
DECAY = ModelEnum.decay
BUMP = ModelEnum.bump


class Model:
    _model_class: ModelEnum

    def __init__(self, initial: Price):
        self.initial = initial
        self.timeline: Dict[TimePeriod, Modifier] = {}

    @property
    def model_type(self) -> ModelEnum:
        return self._model_class

    @property
    def model_name(self) -> str:
        return self.model_type.name

    @property
    def params(self) -> str:
        raise NotImplementedError("Abstract Model has no Parameters")

    @property
    def name(self) -> str:
        basis = [f"{self.model_name}@{str(self.initial)}"]
        if self.params:
            basis.append(self.params)
        return "; ".join(basis)

    def __str__(self) -> str:
        lines = [
            f"# Model: {self.name}\n",
        ]
        for time, modifier in self.timeline.items():
            col = f"{time.name}:"
            price = f"{str(modifier.price)}"
            modrange = f"{str(modifier)}"
            modtype = f"{modifier.name}"
            lines.append(f"{col:13} {price:10} {modrange:20} ({modtype})")
        return '\n'.join(lines)

    def histogram(self) -> Dict[str, CounterType[int]]:
        histogram: Dict[str, CounterType[int]] = {}
        for time, modifier in self.timeline.items():
            price_low = math.ceil(modifier.price.lower)
            price_high = math.ceil(modifier.price.upper)
            counter = histogram.setdefault(time.name, Counter())
            for price in range(price_low, price_high + 1):
                counter[price] += 1
        return histogram

    @classmethod
    def inner_permutations(cls, initial: int) -> Generator[Model, None, None]:
        yield cls(Price(initial))

    @classmethod
    def permutations(cls, initial: Optional[int] = None) -> Generator[Model, None, None]:
        if initial is not None:
            assert 90 <= initial <= 110, "Initial should be within [90, 110]"
            lower = initial
            upper = initial
        else:
            lower = 90
            upper = 110

        for tprice in range(lower, upper + 1):
            yield from cls.inner_permutations(tprice)

    def fix_price(self, time: AnyTime, price: int) -> None:
        timeslice = TimePeriod.normalize(time)
        self.timeline[timeslice].fix_price(price)

    def random_price(self, time: AnyTime):
        '''Provide a random price for the given time, unless that time has data.'''
        timeslice = TimePeriod.normalize(time)
        if self.timeline[timeslice].price.is_atomic:
            return self.timeline[timeslice].price.value
        return random.randint(self.timeline[timeslice].price.lower,
                              self.timeline[timeslice].price.upper)


class TripleModel(Model):
    _model_class = ModelEnum.triple

    def __init__(self,
                 initial: Price,
                 length_phase1: int,
                 length_decay1: int,
                 length_phase2: int):

        super().__init__(initial)

        if not 0 <= length_phase1 <= 6:
            raise ValueError("Phase1 length must be between [0, 6]")
        self._length_phase1 = length_phase1

        if not 2 <= length_decay1 <= 3:
            raise ValueError("Decay1 length must be 2 or 3")
        self._length_decay1 = length_decay1
        self._length_decay2 = 5 - length_decay1

        remainder = 7 - length_phase1
        if not 1 <= length_phase2 <= remainder:
            raise ValueError(f"Phase2 must be between [1, {remainder}]")
        self._length_phase2 = length_phase2
        self._length_phase3 = remainder - length_phase2

        assert (self._length_phase1
                + self._length_phase2
                + self._length_phase3
                + self._length_decay1
                + self._length_decay2) == 12

        chain: List[Modifier] = []
        decay_class: Type[Modifier]

        def _push_node(mod_cls: Type[Modifier]) -> None:
            mod = mod_cls(self.initial, chain[-1] if chain else None)
            chain.append(mod)

        # Phase 1 [0, 6]
        for _ in range(0, self._length_phase1):
            _push_node(SmallProfit)

        # Decay 1 [2, 3]
        decay_class = MediumLoss
        for _ in range(0, self._length_decay1):
            _push_node(decay_class)
            decay_class = RapidDecay

        # Phase 2 [1, 6]
        for _ in range(0, self._length_phase2):
            _push_node(SmallProfit)

        # Decay 2 [2, 3]
        decay_class = MediumLoss
        for _ in range(0, self._length_decay2):
            _push_node(decay_class)
            decay_class = RapidDecay

        # Phase 3 [0, 6]
        for _ in range(0, self._length_phase3):
            _push_node(SmallProfit)

        # Build timeline
        assert len(chain) == 12
        for i, mod in enumerate(chain):
            time = TimePeriod(i + 2)
            self.timeline[time] = mod

    @property
    def phases(self) -> List[int]:
        return [
            self._length_phase1,
            self._length_decay1,
            self._length_phase2,
            self._length_decay2,
            self._length_phase3,
        ]

    @property
    def params(self) -> str:
        return str(self.phases)

    @classmethod
    def inner_permutations(cls, initial: int) -> Generator[TripleModel, None, None]:
        for phase1 in range(0, 6 + 1):  # [0, 6] inclusive
            for decay1 in (2, 3):
                for phase2 in range(1, 7 - phase1 + 1):  # [1, 7 - phase1] inclusive
                    yield cls(Price(initial), phase1, decay1, phase2)


class DecayModel(Model):
    _model_class = ModelEnum.decay

    def __init__(self, initial: Price):
        super().__init__(initial)
        mod: Modifier
        parent: Modifier

        # Slice 2: InitialDecay
        mod = InitialDecay(self.initial, None)
        self.timeline[TimePeriod(2)] = mod
        parent = mod

        # Slices 3-13: Decay
        for i in range(3, 14):
            time = TimePeriod(i)
            mod = SlowDecay(self.initial, parent)
            self.timeline[time] = mod
            parent = mod

    @property
    def params(self) -> str:
        # None params with left beef
        return ""


class PeakModel(Model):
    _pattern_latest = 9
    _pattern_earliest: int
    _peak_time: int

    def __init__(self,
                 initial: 'Price',
                 pattern_start: int):

        low = self._pattern_earliest
        high = self._pattern_latest
        if not low <= pattern_start <= high:
            raise ValueError(f"pattern_start must be between {low} and {high}, inclusive")

        super().__init__(initial)
        self._pattern_start = pattern_start
        self._pattern_peak = TimePeriod(pattern_start + self._peak_time)
        self._tail = 9 - pattern_start

    @property
    def peak(self) -> TimePeriod:
        return self._pattern_peak

    @property
    def params(self) -> str:
        return f"peak@{self.peak.name}"

    @classmethod
    def inner_permutations(cls, initial: int) -> Generator[PeakModel, None, None]:
        for patt in range(cls._pattern_earliest, cls._pattern_latest + 1):
            yield cls(Price(initial), patt)


class BumpModel(PeakModel):
    _model_class = ModelEnum.bump
    _pattern_earliest = 2  # Monday AM
    _peak_time = 3         # Fourth price of pattern

    def __init__(self,
                 initial: 'Price',
                 pattern_start: int):

        super().__init__(initial, pattern_start)

        chain: List[Modifier] = []
        decay_class: Type[Modifier]

        def _push_node(mod_cls: Type[Modifier],
                       parent_override: Optional[Modifier] = None) -> Modifier:
            if parent_override is None:
                my_parent = chain[-1] if chain else None
            else:
                my_parent = parent_override
            mod = mod_cls(self.initial, my_parent)
            chain.append(mod)
            return mod

        decay_class = WideLoss
        for _ in range(2, pattern_start):
            _push_node(decay_class)
            decay_class = SlowDecay

        # Pattern starts:
        _push_node(SmallProfit)
        _push_node(SmallProfit)

        # And then gets weird!
        # Create an unlisted parent that represents the peak shared across the next three prices.
        cap = MediumProfit(self.initial, chain[-1])
        _push_node(CappedPassthrough, cap).sub1 = True
        _push_node(Passthrough, cap)
        _push_node(CappedPassthrough, cap).sub1 = True

        # Alright, phew.
        decay_class = WideLoss
        for _ in range(0, self._tail):
            _push_node(decay_class)
            decay_class = SlowDecay

        # Build timeline
        assert len(chain) == 12
        for i, mod in enumerate(chain):
            time = TimePeriod(i + 2)
            self.timeline[time] = mod


class SpikeModel(PeakModel):
    _model_class = ModelEnum.spike
    _pattern_earliest = 3  # Monday PM
    _peak_time = 2         # Third price of pattern

    def __init__(self,
                 initial: 'Price',
                 pattern_start: int):

        super().__init__(initial, pattern_start)

        def _get_pattern(value: int) -> Type[Modifier]:
            # The default:
            cls: Type[Modifier] = SlowDecay

            # Pattern takes priority:
            if value == pattern_start:
                cls = SmallProfit
            elif value == pattern_start + 1:
                cls = MediumProfit
            elif value == pattern_start + 2:
                cls = LargeProfit
            elif value == pattern_start + 3:
                cls = MediumProfit
            elif value == pattern_start + 4:
                cls = SmallProfit
            elif value >= pattern_start + 5:
                # Week finishes out with independent low prices
                cls = WideLoss
            elif value == 2:
                # Normal start-of-week pattern
                cls = InitialDecay

            return cls

        parent = None
        # Slices 2-13: Magic!
        for i in range(2, 14):
            time = TimePeriod(i)
            mod = _get_pattern(i)(self.initial, parent)
            self.timeline[time] = mod
            parent = mod
