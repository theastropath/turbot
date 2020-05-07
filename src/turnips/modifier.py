#!/usr/bin/env python3

from __future__ import annotations

from decimal import Decimal, ROUND_DOWN, ROUND_UP
import logging
import math
from typing import Optional

from turnips.price import Price

def find_lower_bound(value: int, base: float, precision: str = '0.0001') -> float:
    """
    This awful function computes the smallest float x such that:
    ceil(x * base) == value
    to within some arbitrary quantized precision, e.g. 0.0001.
    """
    lower_int = value - 1
    lower_bound = Decimal(lower_int / base)
    lower_approx = float(lower_bound.quantize((Decimal(precision)),
                                              rounding=ROUND_DOWN))
    while math.ceil(lower_approx * base) < value:
        lower_approx += 0.0001

    return lower_approx


def find_upper_bound(value: int, base: float, precision: str = '0.0001') -> float:
    """
    This awful function computes the largest float x such that:
    ceil(x * base) == value
    to within some arbitrary quantized precision, e.g. 0.0001.
    """
    upper_bound = Decimal(value / base)
    upper_approx = float(upper_bound.quantize((Decimal(precision)),
                                              rounding=ROUND_UP))
    while math.ceil(upper_approx * base) > value:
        upper_approx -= 0.0001

    return upper_approx


class Modifier:
    def __init__(self,
                 base: Price,
                 parent: Optional[Modifier] = None):
        self._base = base
        self._parent = parent
        self._exact_price: Optional[float] = None
        self._static_low: Optional[float] = None
        self._static_high: Optional[float] = None
        self.sub1 = False

    def __str__(self) -> str:
        low = self.lower_bound
        high = self.upper_bound
        return f"x[{low:0.4f}, {high:0.4f}]"

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def parent(self) -> Modifier:
        if self._parent is None:
            raise ValueError("Modifier node has no parent")
        return self._parent

    def _default_lower_bound(self) -> float:
        raise NotImplementedError

    @property
    def lower_bound(self) -> float:
        if self._static_low is not None:
            return self._static_low
        return self._default_lower_bound()

    def _default_upper_bound(self) -> float:
        raise NotImplementedError

    @property
    def upper_bound(self) -> float:
        if self._static_high is not None:
            return self._static_high
        return self._default_upper_bound()

    @property
    def price(self) -> Price:
        if self._exact_price is not None:
            return Price(self._exact_price)

        lower = self.lower_bound * self._base.lower
        upper = self.upper_bound * self._base.upper

        if self.sub1:
            return Price(lower, upper, ufilter=lambda f: f - 1)

        return Price(lower, upper)

    def tighten_bounds(self,
                       lower: Optional[float] = None,
                       upper: Optional[float] = None) -> None:
        if lower is not None and upper is not None:
            if lower > upper:
                raise ValueError("Invalid Argument (Range is reversed): [{lower}, {upper}]")

        if lower is not None:
            if self.lower_bound <= lower <= self.upper_bound:
                self._static_low = lower

        if upper is not None:
            if self.lower_bound <= upper <= self.upper_bound:
                self._static_high = upper

    def fix_price(self, price: int) -> None:
        # Protection against typos and fat fingers:
        if self._exact_price is not None:
            if self._exact_price != price:
                raise ValueError("Invalid Argument: Cannot re-fix price range with new price")

        price_window = self.price

        # Check that the new price isn't immediately outside of what we presently consider possible
        if price < price_window.lower:
            msg = f"Cannot fix price at {price:d}, below model minimum {price_window.lower:d}"
            raise ArithmeticError(msg)
        if price > price_window.upper:
            msg = f"Cannot fix price at {price:d}, above model maximum {price_window.upper:d}"
            raise ArithmeticError(msg)

        # Calculate our presently understood modifier bounds
        current_mod_low = self.lower_bound
        current_mod_high = self.upper_bound
        logging.debug('current modifier range: [{:0.4f}, {:0.4f}]'.format(
            current_mod_low, current_mod_high))

        # Sometimes a price given is (modifier * price) - 1;
        # to compute the correct coefficients, we need to add one!
        if self.sub1:
            unfiltered_price = price + 1
        else:
            unfiltered_price = price

        # For the price given, determine the modifiers that could have produced that value.
        #
        # NB: Accommodate an unknown base price by computing against
        # the lower and upper bounds of the base price.
        #
        # NB2: The boundaries here can change over time, depending on a few things:
        #
        # (1) If the base price range is refined in the future, this
        #     calculation could change.  (At present, MultiModel
        #     always builds models with fixed base prices, not
        #     ranges.)
        #
        # (2) If we are a dynamic modifier node and any of our parents
        #     refine THEIR bounds, this could change.  This is
        #     generally only a problem when there are gaps in the
        #     data; Subsequent constraints will be necessarily less
        #     accurate.
        #
        # FIXME: Once this computation is performed, it's completely
        #        static. Oops!  This code needs to be a little more
        #        dynamically adaptable.

        modifiers = []
        for bound in (self._base.lower, self._base.upper):
            modifiers.append(find_lower_bound(unfiltered_price, bound))
            modifiers.append(find_upper_bound(unfiltered_price, bound))
        new_mod_low = min(modifiers)
        new_mod_high = max(modifiers)
        logging.debug('fixed modifier range: [{:0.4f}, {:0.4f}]'.format(new_mod_low, new_mod_high))

        # If our modifiers are out of scope, we can reject the price for this model.
        if new_mod_low > current_mod_high:
            msg = 'New low modifier ({:0.4f}) out of range [{:0.4f}, {:0.4f}]'.format(
                new_mod_low, current_mod_low, current_mod_high)
            raise ArithmeticError(msg)
        if new_mod_high < current_mod_low:
            msg = 'New high modifier ({:0.4f}) out of range [{:0.4f}, {:0.4f}]'.format(
                new_mod_high, current_mod_low, current_mod_high)
            raise ArithmeticError(msg)
        assert new_mod_low <= new_mod_high, 'God has left us'

        # Our calculated modifier range might overlap with the base model range.
        # Base: [    LxxxxxxH      ]
        # Calc: [ lxxxxxh          ]
        # Calc: [        lxxxxh    ]
        #
        # i.e. our low end may be lower then the existing low end,
        # or the high end might be higher then the existing high end.
        #
        # The assertions in the code block above only assert that:
        # (1) l <= H
        # (2) h >= L
        #
        # Which means that these formulations are valid:
        # Base: [    LxxxxxxH      ]
        # Calc: [           lxxxxh ]
        # Calc: [ lxxh             ]
        #
        # This is fine, it just means that we can rule out some of the modifiers
        # in the range of possibilities.
        # Clamp the new computed ranges to respect the original boundaries.
        new_mod_low = max(new_mod_low, current_mod_low)
        new_mod_high = min(new_mod_high, current_mod_high)
        logging.debug('clamped modifier range: [{:0.4f}, {:0.4f}]'.format(
            new_mod_low, new_mod_high))

        self.tighten_bounds(new_mod_low, new_mod_high)
        self._exact_price = price


class RootModifier(Modifier):
    def _default_lower_bound(self) -> float:
        return 0.00

    def _default_upper_bound(self) -> float:
        return 6.00


class StaticModifier(Modifier):
    default_lower = 0.00
    default_upper = 1.00

    def _default_lower_bound(self) -> float:
        return self.default_lower

    def _default_upper_bound(self) -> float:
        return self.default_upper


class InitialDecay(StaticModifier):
    default_lower = 0.85
    default_upper = 0.90


class WideLoss(StaticModifier):
    default_lower = 0.40
    default_upper = 0.90


class MediumLoss(StaticModifier):
    default_lower = 0.6
    default_upper = 0.8


class SmallProfit(StaticModifier):
    default_lower = 0.90
    default_upper = 1.40


class MediumProfit(StaticModifier):
    default_lower = 1.4
    default_upper = 2.0


class LargeProfit(StaticModifier):
    default_lower = 2.0
    default_upper = 6.0


class DynamicModifier(Modifier):
    """
    DynamicModifier inherits the modifier coefficients of its parent.
    It can modify these coefficients with two deltas.
    """
    delta_lower = 0.00
    delta_upper = 0.00

    def _default_lower_bound(self) -> float:
        return self.parent.lower_bound + self.delta_lower

    def _default_upper_bound(self) -> float:
        return self.parent.upper_bound + self.delta_upper


class Passthrough(DynamicModifier):
    @property
    def name(self) -> str:
        return f"{self.parent.name}*"

    def fix_price(self, price: int) -> None:
        self.parent.fix_price(price)
        super().fix_price(price)


class CappedPassthrough(DynamicModifier):
    # Inherit the *static* lower bound of our parent
    def _default_lower_bound(self) -> float:
        assert isinstance(self.parent, StaticModifier)
        return self.parent.default_lower

    # Upper bound is our parent's dynamic upper bound, as normal for Dynamic nodes

    def fix_price(self, price: int) -> None:
        super().fix_price(price)
        self.parent.tighten_bounds(lower=self.lower_bound)

    @property
    def name(self) -> str:
        return f"{self.parent.name} (Capped)"


class SlowDecay(DynamicModifier):
    delta_lower = -0.05
    delta_upper = -0.03


class RapidDecay(DynamicModifier):
    delta_lower = -0.10
    delta_upper = -0.04
