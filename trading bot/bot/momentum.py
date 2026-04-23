"""Clenow-style + time-series momentum ranking of a universe.

Score = blend of:
- Clenow (2015): annualized slope of log-prices × R²   (over last N bars)
- TSMom (Moskowitz, Ooi, Pedersen 2012): 252-day return skipping most recent
  21 bars (avoids short-term reversal noise in the ranking)

Both scores are expressed as percentage returns and averaged 50/50 when
enough data is available. The 12-1 skip filter is empirically worth
~+0.15 Sharpe across equity trend-following studies.

The higher the combined score, the stronger AND smoother the uptrend. Only
the top-K ranked symbols are allowed to take new entries in a given period.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger("bot.momentum")


def _clenow_slope_score(closes: pd.Series, lookback: int,
                        bars_per_year: int) -> float:
    s = closes.dropna().tail(lookback)
    if len(s) < max(20, lookback // 2):
        return 0.0
    log_p = np.log(s.values)
    x = np.arange(len(log_p), dtype=float)

    x_mean = x.mean()
    y_mean = log_p.mean()
    x_var = ((x - x_mean) ** 2).sum()
    if x_var <= 0:
        return 0.0
    slope = ((x - x_mean) * (log_p - y_mean)).sum() / x_var
    y_pred = y_mean + slope * (x - x_mean)
    ss_res = ((log_p - y_pred) ** 2).sum()
    ss_tot = ((log_p - y_mean) ** 2).sum()
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    annualized = (math.exp(slope) ** bars_per_year - 1) * 100.0
    return float(annualized * r_squared)


def _tsmom_skip1_score(closes: pd.Series, lookback: int = 252,
                       skip: int = 21) -> float:
    """12-1 momentum: total return over `lookback` bars EXCLUDING the most
    recent `skip` bars. Returns a percentage."""
    s = closes.dropna()
    if len(s) < lookback + 1:
        return 0.0
    # Close at t-skip vs t-lookback
    end = s.iloc[-skip - 1] if skip > 0 else s.iloc[-1]
    start = s.iloc[-lookback - 1] if len(s) > lookback else s.iloc[0]
    if start <= 0:
        return 0.0
    return float((end / start - 1.0) * 100.0)


def momentum_score(closes: pd.Series, lookback: int = 90,
                   bars_per_year: int = 252) -> float:
    """Blended momentum score. Falls back to pure Clenow if series is too
    short for the 12-1 skip component."""
    clenow = _clenow_slope_score(closes, lookback, bars_per_year)
    # Need lookback + skip + some slack for the 12-1 component.
    if len(closes.dropna()) >= 252 + 21 + 1:
        tsmom = _tsmom_skip1_score(closes, lookback=252, skip=21)
        return 0.5 * clenow + 0.5 * tsmom
    return clenow


@dataclass
class MomentumRanking:
    """Frozen ranking for the current decision window."""
    scores: dict[str, float]

    def _ranked_positive(self) -> list[tuple[str, float]]:
        ranked = sorted(self.scores.items(), key=lambda kv: kv[1], reverse=True)
        return [(s, sc) for s, sc in ranked if sc > 0]

    def top(self, k: int) -> list[str]:
        return [s for s, _ in self._ranked_positive()[:k]]

    def contains(self, symbol: str, top_k: int) -> bool:
        return symbol in self.top(top_k)

    def rank_of(self, symbol: str) -> int | None:
        """1-indexed rank among positive-score symbols, or None if not ranked."""
        for i, (s, _) in enumerate(self._ranked_positive(), start=1):
            if s == symbol:
                return i
        return None


def rank_universe(
    symbol_to_closes: dict[str, pd.Series],
    lookback: int = 90,
    bars_per_year: int = 252,
) -> MomentumRanking:
    scores = {s: momentum_score(c, lookback, bars_per_year)
              for s, c in symbol_to_closes.items()}
    return MomentumRanking(scores=scores)
