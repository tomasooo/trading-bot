"""Entry filters & regime detection.

Gating functions the engine / backtest consult BEFORE submitting a new entry.
Each filter returns (allowed: bool, reason: str) so the logs explain rejections.

Currently implemented:
- time_filter: no entries in the first / last N minutes of the US session
- volatility_regime: reject when ATR is in the top percentile of its history
                     (vol blow-offs are breakout traps and mean-reversion knives)
- volume_confirmation: breakout entries require above-average volume
- sector_cap: cap exposure per sector at max_sector_pct of equity
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time, timezone, timedelta
from datetime import datetime
from typing import Iterable

import pandas as pd


# Minimal hard-coded GICS sector map for common tickers. Extend as needed.
SECTORS: dict[str, str] = {
    "AAPL": "TECH", "MSFT": "TECH", "NVDA": "TECH", "GOOGL": "TECH",
    "GOOG": "TECH", "META": "TECH", "AMD": "TECH", "AVGO": "TECH",
    "INTC": "TECH", "CRM": "TECH", "ADBE": "TECH", "ORCL": "TECH",
    "TSLA": "CONSUMER_DISC", "AMZN": "CONSUMER_DISC", "HD": "CONSUMER_DISC",
    "NKE": "CONSUMER_DISC",
    "JPM": "FINANCIALS", "BAC": "FINANCIALS", "GS": "FINANCIALS",
    "XOM": "ENERGY", "CVX": "ENERGY",
    "JNJ": "HEALTH", "PFE": "HEALTH", "UNH": "HEALTH",
    "SPY": "INDEX", "QQQ": "INDEX", "IWM": "INDEX", "DIA": "INDEX",
}


def sector_of(symbol: str) -> str:
    return SECTORS.get(symbol.upper(), "OTHER")


@dataclass
class FilterResult:
    ok: bool
    reason: str = ""


# ---------- time ----------
# US regular session: 09:30–16:00 ET. Skip the first 15 and last 15 min.
_ET = timezone(timedelta(hours=-5))  # EST; DST handled only approximately
_SESSION_OPEN = time(9, 30)
_SESSION_CLOSE = time(16, 0)
_SKIP_OPEN_MIN = 15
_SKIP_CLOSE_MIN = 15


def time_filter(now_utc: datetime | None = None) -> FilterResult:
    """Reject entries in the first/last 15 min of the US regular session.

    We read the broker's clock for market_open in the engine, so this is only
    a secondary guard that also works in backtests. ET here is approximated —
    for strict DST-correct logic, use `zoneinfo` in production.
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    # Approximate ET with UTC-4 (EDT) — accepted for filter purposes.
    et = (now_utc.astimezone(timezone.utc) - timedelta(hours=4)).time()
    if et < _SESSION_OPEN or et > _SESSION_CLOSE:
        return FilterResult(True, "outside-session")  # engine already guards on market_open
    open_cutoff = time(9, 30 + _SKIP_OPEN_MIN)
    close_cutoff = time(15, 60 - _SKIP_CLOSE_MIN if _SKIP_CLOSE_MIN < 60 else 45)
    if et < open_cutoff:
        return FilterResult(False, f"first {_SKIP_OPEN_MIN} min of session")
    if et >= close_cutoff:
        return FilterResult(False, f"last {_SKIP_CLOSE_MIN} min of session")
    return FilterResult(True)


# ---------- volatility regime ----------
def atr_percentile(atr_series: pd.Series, lookback: int = 100) -> float:
    """Return the percentile rank (0-1) of the current ATR within `lookback` bars."""
    s = atr_series.dropna().tail(lookback)
    if len(s) < 10:
        return 0.5  # not enough data — be permissive
    current = s.iloc[-1]
    return float((s <= current).mean())


def volatility_regime_ok(atr_series: pd.Series, high_pct: float = 0.95,
                         lookback: int = 100) -> FilterResult:
    pct = atr_percentile(atr_series, lookback)
    if pct >= high_pct:
        return FilterResult(False, f"vol regime blow-off (ATR pct={pct:.2f} >= {high_pct})")
    return FilterResult(True)


# ---------- volume confirmation (breakout entries) ----------
def volume_confirmation(volume: pd.Series, sma_period: int = 20,
                        mult: float = 1.3) -> FilterResult:
    """Current bar volume must exceed `mult` * SMA(volume, sma_period)."""
    if len(volume) < sma_period + 1:
        return FilterResult(True, "not enough vol history")
    avg = volume.tail(sma_period).mean()
    cur = volume.iloc[-1]
    if avg <= 0 or cur <= 0:
        return FilterResult(True, "zero-volume bar")
    if cur < mult * avg:
        return FilterResult(False, f"volume {cur:.0f} < {mult}x avg {avg:.0f}")
    return FilterResult(True)


# ---------- Clenow per-stock filters ----------
# "Stocks on the Move" (Clenow 2015, ch. 7) filters applied BEFORE the entry
# signal — they reject symbols with idiosyncratic gaps or broken individual
# trends, independent of where they rank in the universe.

def has_recent_gap(close: pd.Series, threshold: float = 0.15,
                   lookback: int = 90) -> bool:
    """Return True if any single-bar absolute return exceeded `threshold`
    in the last `lookback` bars. Proxy for earnings / news gaps — per Clenow
    you don't want to ride a stock that just had a 15 %+ overnight move."""
    s = close.tail(lookback + 1)
    if len(s) < 3:
        return False
    rets = s.pct_change().abs().dropna()
    return bool((rets > threshold).any())


def above_own_sma(close: pd.Series, period: int = 100) -> bool:
    """Individual-stock trend filter: last close above its own `period`-SMA.
    Even if SPY is bullish, a specific name in a private downtrend is a skip."""
    if len(close) < period:
        return True  # not enough data — be permissive
    sma_val = close.tail(period).mean()
    return bool(close.iloc[-1] > sma_val)


def clenow_filters_ok(close: pd.Series,
                      gap_threshold: float = 0.15,
                      gap_lookback: int = 90,
                      sma_period: int = 100) -> FilterResult:
    """Combined Clenow stock-selection filters. Returns (ok, reason)."""
    if has_recent_gap(close, gap_threshold, gap_lookback):
        return FilterResult(False,
                            f"recent >{gap_threshold*100:.0f}% gap in last {gap_lookback}d")
    if not above_own_sma(close, sma_period):
        return FilterResult(False, f"close below own {sma_period}-SMA")
    return FilterResult(True)


# ---------- sector concentration ----------
def sector_cap(
    symbol: str,
    positions: Iterable,   # list of Position with .symbol and .market_value
    equity: float,
    new_notional: float,
    max_pct: float = 0.25,
) -> FilterResult:
    """Reject if adding this trade would push the symbol's sector above max_pct."""
    if equity <= 0:
        return FilterResult(True)
    target_sector = sector_of(symbol)
    if target_sector == "OTHER":
        return FilterResult(True)
    existing = sum(
        abs(float(p.market_value)) for p in positions
        if sector_of(p.symbol) == target_sector
    )
    total = existing + new_notional
    pct = total / equity
    if pct > max_pct:
        return FilterResult(False,
                            f"{target_sector} exposure would be {pct*100:.1f}% > {max_pct*100:.0f}%")
    return FilterResult(True)
