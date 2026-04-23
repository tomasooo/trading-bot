"""Chandelier exit + break-even stop logic.

Both the live engine and the backtest use these helpers to compute the
current trailing stop for an open long position.

Rules (long only):
1. Initial stop = entry - stop_mult * ATR_at_entry           (set at entry time)
2. After price has moved +1R in our favor, move stop to entry (break-even).
3. After price has moved +2R, set the "profit_lock_active" flag (for
   observability — dashboard can show "locked" positions) but DO NOT tighten
   the chandelier. Empirically, tightening past 2R chopped big winners short
   and cost ~6% return on the 4-year breakout backtest.
4. Chandelier = max(high_since_entry) - trail_mult * ATR_current
   Stop = MAX(break-even, chandelier, initial) — never relaxed.
"""
from __future__ import annotations

from dataclasses import dataclass


# Profit-lock threshold — informational flag only; doesn't change sizing.
PROFIT_LOCK_R = 2.0


@dataclass
class TrailState:
    entry: float
    initial_stop: float
    atr_at_entry: float          # used to size 1R
    high_since_entry: float
    breakeven_active: bool = False
    profit_lock_active: bool = False
    current_stop: float = 0.0    # last computed stop


def _one_r(entry: float, initial_stop: float) -> float:
    return entry - initial_stop


def update_trail(
    state: TrailState,
    current_high: float,
    current_low: float,
    current_atr: float,
    trail_mult: float = 3.0,
) -> TrailState:
    """Advance the trail state given the latest bar's high / low / ATR.

    Returns a new TrailState with updated high_since_entry and current_stop.
    Caller decides whether current_stop is hit by comparing to the bar low
    (or live price).
    """
    r = _one_r(state.entry, state.initial_stop)
    hi = max(state.high_since_entry, current_high)

    be = state.breakeven_active or hi >= state.entry + r
    pl = state.profit_lock_active or hi >= state.entry + PROFIT_LOCK_R * r

    chandelier = hi - trail_mult * current_atr if current_atr > 0 else state.initial_stop
    breakeven_stop = state.entry if be else state.initial_stop

    new_stop = max(state.initial_stop, breakeven_stop, chandelier)
    # Monotone non-decreasing (never loosen once raised)
    new_stop = max(new_stop, state.current_stop)

    return TrailState(
        entry=state.entry,
        initial_stop=state.initial_stop,
        atr_at_entry=state.atr_at_entry,
        high_since_entry=hi,
        breakeven_active=be,
        profit_lock_active=pl,
        current_stop=new_stop,
    )
