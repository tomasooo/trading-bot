"""Ensemble / voting strategy.

Combines multiple strategies. A BUY requires at least `min_agreement` children
to signal BUY (and none to signal SELL). A SELL fires if any child says SELL
(conservative exit).
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from .base import Signal, Strategy, StrategyContext


class EnsembleStrategy:
    name = "ensemble"

    def __init__(self, children: Iterable[Strategy], min_agreement: int = 2):
        self.children = list(children)
        if min_agreement < 1 or min_agreement > len(self.children):
            raise ValueError(
                f"min_agreement must be in [1, {len(self.children)}], got {min_agreement}"
            )
        self.min_agreement = min_agreement

    def required_bars(self) -> int:
        return max(c.required_bars() for c in self.children)

    def signal(self, bars: pd.DataFrame, ctx: StrategyContext) -> Signal:
        votes = [c.signal(bars, ctx) for c in self.children]
        buys = sum(1 for v in votes if v == Signal.BUY)
        sells = sum(1 for v in votes if v == Signal.SELL)

        if ctx.in_position:
            # Any child calling SELL closes the trade — conservative.
            return Signal.SELL if sells >= 1 else Signal.HOLD

        if buys >= self.min_agreement and sells == 0:
            return Signal.BUY
        return Signal.HOLD
