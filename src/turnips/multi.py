#!/usr/bin/env python3

from __future__ import annotations

from collections import Counter
import logging
from typing import Callable, Dict, Generator, List, Optional, Set, Type
from typing import Counter as CounterType

from turnips.model import (
    Model,
    TripleModel,
    SpikeModel,
    DecayModel,
    BumpModel,
)
from turnips.ttime import AnyTime, TimePeriod


class RangeSet:
    def __init__(self) -> None:
        self._set: Set[int] = set()

    def add(self, value: int) -> None:
        self._set.add(value)

    def __str__(self) -> str:
        sortlist = sorted(list(self._set))
        ranges = []

        seed = sortlist.pop(0)
        current = [seed, seed]

        for item in sortlist:
            if item == current[1] + 1:
                current[1] = item
            else:
                ranges.append((current[0], current[1]))
                current = [item, item]
        ranges.append((current[0], current[1]))

        chunks = []
        for prange in ranges:
            if prange[0] == prange[1]:
                chunk = f"{prange[0]}"
            else:
                chunk = f"[{prange[0]}, {prange[1]}]"
            chunks.append(chunk)

        if len(chunks) == 1:
            return str(chunks[0])
        return "{" + ", ".join(chunks) + "}"


class MultiModel:
    """
    A collection of several models that we can aggregate data about.
    """
    def __init__(self, initial: Optional[int] = None):
        self._initial = initial
        self._models: Dict[int, Model] = {}

    @property
    def models(self) -> Generator[Model, None, None]:
        for model in self._models.values():
            yield model

    def fix_price(self, time: AnyTime, price: int) -> None:
        if not self:
            assert RuntimeError("No viable models to fix prices on!")

        timeslice = TimePeriod.normalize(time)
        remove_queue = []
        for index, model in self._models.items():
            try:
                model.fix_price(timeslice, price)
            except ArithmeticError as exc:
                logging.info(f"Ruled out model: {model.name}")
                logging.debug(f"  Reason: {timeslice.name} price={price} not possible:")
                logging.debug(f"    {str(exc)}")
                remove_queue.append(index)

        for i in remove_queue:
            del self._models[i]

    def __bool__(self) -> bool:
        return bool(self._models)

    def __len__(self) -> int:
        return len(self._models)

    def __str__(self) -> str:
        if not self:
            return "--- No Viable Models ---"
        return "\n\n".join([str(model) for model in self.models])

    def histogram(self) -> Dict[str, CounterType[int]]:
        histogram: Dict[str, CounterType[int]] = {}
        for model in self.models:
            mhist = model.histogram()
            for timename in mhist:
                counter = histogram.setdefault(timename, Counter())
                counter.update(mhist[timename])
        return histogram

    def summary(self) -> None:
        print('')
        print("  Summary: ")
        print("    {:13} {:23} {:23} {:6}".format('Time', 'Price', 'Likely', 'Odds'))
        hist = self.histogram()
        for time, pricecounts in hist.items():
            # Gather possible prices
            pset = RangeSet()
            for price in pricecounts.keys():
                pset.add(price)

            # Determine likeliest price(s)
            n_possibilities = sum(pricecounts.values())
            likeliest = max(pricecounts.items(), key=lambda x: x[1])
            likelies = list(filter(lambda x: x[1] >= likeliest[1], pricecounts.items()))

            sample_size = len(likelies) * likeliest[1]
            pct = 100 * (sample_size / n_possibilities)

            rset = RangeSet()
            for likely in likelies:
                rset.add(likely[0])

            time_col = f"{time}:"
            price_col = f"{str(pset)};"
            likely_col = f"{str(rset)};"
            chance_col = f"({pct:0.2f}%)"
            print(f"    {time_col:13} {price_col:23} {likely_col:23} {chance_col:6}")

    def detailed_report(self) -> None:
        raise NotImplementedError

    def report(self, show_summary: bool = True) -> None:
        if len(self) == 1:
            for model in self.models:
                print(model)
        else:
            self.detailed_report()
            if show_summary:
                self.summary()

    def chatty_fix_price(self, time: AnyTime, price: int) -> None:
        timeslice = TimePeriod.normalize(time)
        self.fix_price(time, price)
        print(f"Added {timeslice.name} @ {price};")
        self.report()


class PeakModels(MultiModel):
    """
    PeakModels gathers aggregate data about Bump and Spike models.
    """
    _interior_class: Type[Model]

    def __init__(self, initial: Optional[int] = None):
        super().__init__(initial)
        for i, model in enumerate(self._interior_class.permutations(initial)):
            self._models[i] = model

    def detailed_report(self) -> None:
        buckets: Dict[TimePeriod, List[Model]] = {}
        prices: Dict[TimePeriod, RangeSet] = {}
        global_prices = RangeSet()
        for model in self._models.values():
            assert isinstance(model, (BumpModel, SpikeModel))  # Shut up mypy, I know it's bad OO
            peak = model.peak
            buckets.setdefault(peak, []).append(model)
            prices.setdefault(peak, RangeSet()).add(int(model.initial.value))
            global_prices.add(int(model.initial.value))

        print(f"{self._interior_class.__name__} Model Analyses:")
        print(f"  {len(self._models)} model(s) remaining")
        print(f"  base price: {str(global_prices)}")

        if len(buckets) != 1:
            print('')
            print(f"  {len(buckets)} possible peak times:")

        for key, _models in buckets.items():
            groupprices = prices[key]
            bullet = '' if len(buckets) == 1 else '- '
            print(f"  {bullet}peak time: {key}")
            if len(buckets) == 1:
                # Don't re-print the prices for just one group.
                continue
            print(f"      base price: {str(groupprices)}")


class BumpModels(PeakModels):
    _interior_class = BumpModel


class SpikeModels(PeakModels):
    _interior_class = SpikeModel


class DecayModels(MultiModel):
    def __init__(self, initial: Optional[int] = None):
        super().__init__(initial)
        for i, model in enumerate(DecayModel.permutations(initial)):
            self._models[i] = model

    def detailed_report(self) -> None:
        global_prices = RangeSet()
        for model in self._models.values():
            global_prices.add(int(model.initial.value))

        print(f"Decay Model Analyses:")
        print(f"  {len(self._models)} model(s) remaining")
        print(f"  base price: {str(global_prices)}")


class TripleModels(MultiModel):
    def __init__(self, initial: Optional[int] = None):
        super().__init__(initial)
        for i, model in enumerate(TripleModel.permutations(initial)):
            self._models[i] = model

    def detailed_report(self) -> None:
        # Alright, long story short:
        # This method (and _analyze_models) loop over the known parameters (params)
        # and their values, and look for parameters which are now "fixed" in the remaining data.
        #
        # If any are found to have only one possible value, we print that value and
        # delete it from the dict.
        #
        # If there are parameters with multiple values, loop over each value
        # and recursively call _analyze_models on just those models.

        params = {
            'phase1': lambda m: m.phases[0],
            'decay1': lambda m: m.phases[1],
            'phase2': lambda m: m.phases[2],
            'price': lambda m: int(m.initial.value),
        }

        print(f"Triple Model Analyses:")
        print(f"  {len(self._models)} model(s) remaining")

        mlist: List[TripleModel] = []
        for model in self._models.values():
            assert isinstance(model, TripleModel)
            mlist.append(model)

        self._analyze_models(mlist, params)

    def _analyze_models(self,
                        models: List[TripleModel],
                        params: Dict[str, Callable[[TripleModel], int]],
                        indent: int = 2) -> None:
        indent_str = ' ' * indent

        buckets: Dict[str, Dict[int, List[TripleModel]]] = {}
        for model in models:
            for param, param_fn in params.items():
                pdict = buckets.setdefault(param, {})
                pvalue = param_fn(model)
                pdict.setdefault(pvalue, []).append(model)

        remove_queue = []
        for param, pvalues in buckets.items():
            if len(pvalues) == 1:
                pvalue = list(pvalues.keys())[0]
                print(f"{indent_str}{param}: {pvalue}")
                remove_queue.append(param)

        for remove in remove_queue:
            buckets.pop(remove)
            params.pop(remove)

        if len(buckets) == 1:
            # Only one parameter left, so just print it now, as a list.
            pset = RangeSet()
            for param, pvalues in buckets.items():
                for pvalue in pvalues.keys():
                    pset.add(pvalue)
            print(f"{indent_str}{param}: {str(pset)}")
            return

        for param, pvalues in buckets.items():
            params.pop(param)
            for pvalue in pvalues.keys():
                print(f"{indent_str}- {param}: {pvalue}")
                pcopy = params.copy()
                self._analyze_models(pvalues[pvalue], pcopy, indent + 2)
            return
