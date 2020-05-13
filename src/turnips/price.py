#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple


class Price:
    def __init__(self,
                 price: float,
                 upper: Optional[float] = None,
                 ufilter: Optional[Callable[[float], float]] = None):
        self._lower = price
        self._upper = price if upper is None else upper
        self._user_filter = ufilter

    def _filter(self, value: float) -> float:
        if self._user_filter is not None:
            return self._user_filter(math.ceil(value))
        return math.ceil(value)

    @property
    def is_atomic(self) -> bool:
        return self._lower == self._upper

    @property
    def lower(self) -> float:
        return self._filter(self._lower)

    @property
    def upper(self) -> float:
        return self._filter(self._upper)

    @property
    def value(self) -> float:
        if not self.is_atomic:
            raise ValueError("No single value available; this is a range")
        return self.lower

    @property
    def raw(self) -> Tuple[float, float]:
        return (self._lower, self._upper)

    def __repr__(self) -> str:
        if self.is_atomic:
            return '{}({})'.format(self.__class__.__name__, self._lower)
        return '{}({}, {})'.format(self.__class__.__name__, self._lower, self._upper)

    def __str__(self) -> str:
        if self.is_atomic:
            return str(self.value)
        return f"[{self.lower}, {self.upper}]"
