"""Historical + recent bar data from Alpaca."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

log = logging.getLogger("bot.data")


_TF_MAP = {
    "1Min": TimeFrame(1, TimeFrameUnit.Minute),
    "5Min": TimeFrame(5, TimeFrameUnit.Minute),
    "15Min": TimeFrame(15, TimeFrameUnit.Minute),
    "30Min": TimeFrame(30, TimeFrameUnit.Minute),
    "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
    "1Day": TimeFrame(1, TimeFrameUnit.Day),
}


def parse_timeframe(tf: str) -> TimeFrame:
    if tf not in _TF_MAP:
        raise ValueError(f"Unsupported timeframe: {tf}. Use one of {list(_TF_MAP)}")
    return _TF_MAP[tf]


class DataFeed:
    def __init__(self, api_key: str, secret_key: str):
        self.client = StockHistoricalDataClient(api_key, secret_key)

    def bars(
        self,
        symbol: str,
        timeframe: str,
        lookback_bars: int = 300,
    ) -> pd.DataFrame:
        """Return last `lookback_bars` bars for `symbol` as a DataFrame.

        Columns: open, high, low, close, volume. Indexed by UTC timestamp.
        """
        tf = parse_timeframe(timeframe)
        # Pad lookback heuristically based on timeframe to get enough calendar coverage.
        # Alpaca free tier has a 15-min delay; we fetch up to 2 minutes ago to be safe.
        now = datetime.now(timezone.utc) - timedelta(minutes=2)

        # Estimate calendar span needed.
        minutes_per_bar = {
            "1Min": 1, "5Min": 5, "15Min": 15, "30Min": 30,
            "1Hour": 60, "1Day": 60 * 24,
        }[timeframe]
        # US market = ~6.5h of trading per day. For intraday, scale accordingly.
        if timeframe == "1Day":
            span_days = lookback_bars + 20
        else:
            bars_per_day = (6.5 * 60) // minutes_per_bar
            span_days = int(lookback_bars / max(bars_per_day, 1)) + 5
        start = now - timedelta(days=max(span_days, 5))

        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=tf,
            start=start,
            end=now,
            feed="iex",  # free-tier friendly
        )
        bars = self.client.get_stock_bars(req)
        df = bars.df
        if df is None or df.empty:
            log.warning("No bars for %s", symbol)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        # Multi-index (symbol, timestamp) -> drop symbol level
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df = df.tail(lookback_bars)
        return df
