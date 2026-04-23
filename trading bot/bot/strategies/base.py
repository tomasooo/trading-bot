"""Strategy interface.

A strategy takes a DataFrame of OHLCV bars and returns a Signal for the
latest bar: BUY / SELL (exit) / HOLD. Strategies are stateless — the engine
handles position tracking.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import pandas as pd


class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"       # exit a long
    HOLD = "HOLD"


@dataclass
class StrategyContext:
    """Everything a strategy needs besides bars."""
    symbol: str
    in_position: bool


class Strategy(Protocol):
    name: str

    def signal(self, bars: pd.DataFrame, ctx: StrategyContext) -> Signal: ...

    def required_bars(self) -> int: ...
