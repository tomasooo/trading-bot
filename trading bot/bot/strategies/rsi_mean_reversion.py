"""RSI mean-reversion with trend filter.

Long-only: buy when RSI crosses up from oversold while price is still in
an overall uptrend (above long SMA). Exit when RSI reaches overbought.
"""
from __future__ import annotations

import pandas as pd

from ..indicators import rsi, sma
from .base import Signal, Strategy, StrategyContext


class RSIMeanReversionStrategy:
    name = "rsi_mean_reversion"

    def __init__(self, rsi_period: int = 14, oversold: float = 30,
                 overbought: float = 70, trend_filter_period: int = 200):
        self.period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.trend = trend_filter_period

    def required_bars(self) -> int:
        return max(self.period, self.trend) + 5

    def signal(self, bars: pd.DataFrame, ctx: StrategyContext) -> Signal:
        if len(bars) < self.required_bars():
            return Signal.HOLD

        close = bars["close"]
        r = rsi(close, self.period)
        trend = sma(close, self.trend)

        if pd.isna(r.iloc[-1]) or pd.isna(trend.iloc[-1]):
            return Signal.HOLD

        above_trend = close.iloc[-1] > trend.iloc[-1]
        rsi_now = r.iloc[-1]
        rsi_prev = r.iloc[-2]

        if ctx.in_position:
            if rsi_now >= self.overbought:
                return Signal.SELL
            return Signal.HOLD

        # Cross up out of oversold zone, while in broader uptrend
        if rsi_prev < self.oversold <= rsi_now and above_trend:
            return Signal.BUY
        return Signal.HOLD
