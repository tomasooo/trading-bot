"""EMA crossover with long-term trend filter.

BUY  when fast EMA crosses above slow EMA AND price is above trend SMA.
SELL when fast EMA crosses below slow EMA.
"""
from __future__ import annotations

import pandas as pd

from ..indicators import ema, sma
from .base import Signal, Strategy, StrategyContext


class MACrossoverStrategy:
    name = "ma_crossover"

    def __init__(self, fast_period: int = 12, slow_period: int = 26,
                 trend_filter_period: int = 200):
        if fast_period >= slow_period:
            raise ValueError("fast_period must be < slow_period")
        self.fast = fast_period
        self.slow = slow_period
        self.trend = trend_filter_period

    def required_bars(self) -> int:
        return max(self.slow, self.trend) + 5

    def signal(self, bars: pd.DataFrame, ctx: StrategyContext) -> Signal:
        if len(bars) < self.required_bars():
            return Signal.HOLD
        close = bars["close"]
        fast = ema(close, self.fast)
        slow = ema(close, self.slow)
        trend = sma(close, self.trend)

        # Need two points to detect a crossover
        if fast.iloc[-1] != fast.iloc[-1] or slow.iloc[-2] != slow.iloc[-2]:  # NaN
            return Signal.HOLD

        crossed_up = fast.iloc[-2] <= slow.iloc[-2] and fast.iloc[-1] > slow.iloc[-1]
        crossed_down = fast.iloc[-2] >= slow.iloc[-2] and fast.iloc[-1] < slow.iloc[-1]
        above_trend = close.iloc[-1] > trend.iloc[-1]

        if ctx.in_position:
            # Exit on bearish cross
            return Signal.SELL if crossed_down else Signal.HOLD

        # Entry only in uptrend
        if crossed_up and above_trend:
            return Signal.BUY
        return Signal.HOLD
