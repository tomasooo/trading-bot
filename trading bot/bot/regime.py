"""Market regime filter — SPY above its 200-day SMA is the master switch.

Core idea (Faber 2007 / Clenow / countless others): block new longs when
the broad market is in a down-regime. Dramatically reduces drawdowns
during 2008/2020/2022-style declines.

Two variants:
- `market_regime_ok(spy_daily_bars)`: strict on/off at SMA-200.
- `market_regime_score(spy_daily_bars)`: soft scalar in [0.5, 1.0] that
  can be multiplied into position sizing for a less binary approach.
"""
from __future__ import annotations

import logging

import pandas as pd

from .indicators import sma

log = logging.getLogger("bot.regime")


def market_regime_ok(spy_daily: pd.DataFrame, sma_period: int = 200) -> bool:
    """True when SPY close > SMA(200). Requires DAILY bars."""
    if spy_daily is None or spy_daily.empty or len(spy_daily) < sma_period:
        return True  # not enough data — permissive default
    close = spy_daily["close"]
    ma = sma(close, sma_period)
    if pd.isna(ma.iloc[-1]):
        return True
    return float(close.iloc[-1]) > float(ma.iloc[-1])


def market_regime_score(spy_daily: pd.DataFrame, sma_period: int = 200) -> float:
    """Soft regime: 1.0 when clearly above SMA, 0.5 when below.
    Useful as a sizing multiplier rather than a hard cut-off."""
    if spy_daily is None or spy_daily.empty or len(spy_daily) < sma_period:
        return 1.0
    close = spy_daily["close"]
    ma = sma(close, sma_period)
    if pd.isna(ma.iloc[-1]):
        return 1.0
    ratio = float(close.iloc[-1]) / float(ma.iloc[-1])
    if ratio >= 1.0:
        return 1.0
    # Scale from 0.5 at ratio=0.9 to 1.0 at ratio=1.0, clamp
    return max(0.5, min(1.0, 5.0 * (ratio - 0.9) + 0.5))
