"""Sanity tests for the hand-rolled indicator math.

Run:  python -m unittest bot.tests.test_indicators
"""
import unittest

import numpy as np
import pandas as pd

from bot.indicators import atr, ema, rolling_high, rolling_low, rsi, sma


def _ohlc(closes: list[float]) -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame({
        "high": [c + 1 for c in closes],
        "low": [c - 1 for c in closes],
        "close": closes,
    }, index=pd.date_range("2024-01-01", periods=n, freq="D"))


class IndicatorsTest(unittest.TestCase):
    def test_sma_matches_mean(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(s, 3)
        self.assertTrue(np.isnan(result.iloc[0]))
        self.assertAlmostEqual(result.iloc[2], 2.0)
        self.assertAlmostEqual(result.iloc[4], 4.0)

    def test_ema_converges(self):
        s = pd.Series([10.0] * 50)
        result = ema(s, 10)
        self.assertAlmostEqual(result.iloc[-1], 10.0, places=6)

    def test_rsi_range(self):
        # Strictly up -> RSI should approach 100
        s = pd.Series(np.arange(1.0, 50.0))
        r = rsi(s, 14).dropna()
        self.assertTrue(r.iloc[-1] > 90)
        # Strictly down -> RSI should approach 0
        r2 = rsi(s[::-1].reset_index(drop=True), 14).dropna()
        self.assertTrue(r2.iloc[-1] < 10)

    def test_atr_positive_when_range(self):
        df = _ohlc([100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                    111, 110, 112, 114, 113])
        a = atr(df, 14).dropna()
        self.assertTrue((a > 0).all())

    def test_rolling_high_low(self):
        s = pd.Series([1, 5, 3, 8, 2, 7, 4], dtype=float)
        hi = rolling_high(s, 3)
        lo = rolling_low(s, 3)
        # window [1,5,3] -> max 5, min 1
        self.assertEqual(hi.iloc[2], 5)
        self.assertEqual(lo.iloc[2], 1)
        # window [3,8,2] -> max 8, min 2
        self.assertEqual(hi.iloc[4], 8)
        self.assertEqual(lo.iloc[4], 2)


if __name__ == "__main__":
    unittest.main()
