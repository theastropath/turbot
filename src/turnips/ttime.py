#!/usr/bin/env python3

from __future__ import annotations

import enum
from typing import Union


AnyTime = Union[str, int, 'TimePeriod']

class TimePeriod(enum.Enum):
    Sunday_AM = 0
    Sunday_PM = 1
    Monday_AM = 2
    Monday_PM = 3
    Tuesday_AM = 4
    Tuesday_PM = 5
    Wednesday_AM = 6
    Wednesday_PM = 7
    Thursday_AM = 8
    Thursday_PM = 9
    Friday_AM = 10
    Friday_PM = 11
    Saturday_AM = 12
    Saturday_PM = 13

    @classmethod
    def normalize(cls, value: AnyTime) -> TimePeriod:
        if isinstance(value, TimePeriod):
            return value
        if isinstance(value, int):
            return cls(value)
        return cls[value]
