"""Donchian-style breakout with ATR buffer.

BUY when close exceeds the prior N-bar high by more than `atr_buffer * ATR`.
SELL when close falls below the prior N-bar low.
"""
from __future__ import annotations

import pandas as pd

from ..filters import volume_confirmation
from ..indicators import atr, rolling_high, rolling_low
from .base import Signal, Strategy, StrategyContext


class BreakoutStrategy:
    name = "breakout"

    def __init__(self, lookback: int = 20, atr_period: int = 14,
                 atr_buffer: float = 0.5, volume_mult: float = 1.3):
        self.lookback = lookback
        self.atr_period = atr_period
        self.atr_buffer = atr_buffer
        self.volume_mult = volume_mult

    def required_bars(self) -> int:
        return max(self.lookback, self.atr_period) + 25  # +25 for volume SMA

    def signal(self, bars: pd.DataFrame, ctx: StrategyContext) -> Signal:
        if len(bars) < self.required_bars():
            return Signal.HOLD

        close = bars["close"]
        # "Prior" window — shift by 1 so we compare against bars that ended BEFORE current.
        hi = rolling_high(bars["high"], self.lookback).shift(1)
        lo = rolling_low(bars["low"], self.lookback).shift(1)
        a = atr(bars, self.atr_period)

        if pd.isna(hi.iloc[-1]) or pd.isna(a.iloc[-1]):
            return Signal.HOLD

        c = close.iloc[-1]
        up_break = c > hi.iloc[-1] + self.atr_buffer * a.iloc[-1]
        down_break = c < lo.iloc[-1] - self.atr_buffer * a.iloc[-1]

        if ctx.in_position:
            return Signal.SELL if down_break else Signal.HOLD

        if up_break:
            # Volume must confirm the breakout — cheap alpha filter
            vol_ok = volume_confirmation(bars["volume"], sma_period=20,
                                         mult=self.volume_mult)
            if not vol_ok.ok:
                return Signal.HOLD
            return Signal.BUY
        return Signal.HOLD
