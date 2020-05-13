#!/usr/bin/env python3

from __future__ import annotations

from typing import Dict, Union

from turnips.model import ModelEnum, TRIPLE, SPIKE, DECAY, BUMP, UNKNOWN

MARKOV: Dict[Union[None, ModelEnum], Dict[ModelEnum, float]] = {
    TRIPLE: {
        TRIPLE:  0.2,
        SPIKE:   0.3,
        DECAY:   0.15,
        BUMP:    0.35,
    },
    SPIKE: {
        TRIPLE:  0.5,
        SPIKE:   0.05,
        DECAY:   0.2,
        BUMP:    0.25,
    },
    DECAY: {
        TRIPLE:  0.25,
        SPIKE:   0.45,
        DECAY:   0.05,
        BUMP:    0.25,
    },
    BUMP: {
        TRIPLE:  0.45,
        SPIKE:   0.25,
        DECAY:   0.15,
        BUMP:    0.15,
    },
    UNKNOWN: {
        TRIPLE:  0.25,
        SPIKE:   0.25,
        DECAY:   0.25,
        BUMP:    0.25,
    },
    None: {
        BUMP:    1.0,
    },
}
