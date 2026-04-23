"""Tests for Clenow-style momentum ranking.

Run:  python -m unittest bot.tests.test_momentum
"""
import unittest

import numpy as np
import pandas as pd

from bot.momentum import momentum_score, rank_universe


def _trend(start: float, n: int, daily_ret: float) -> pd.Series:
    x = np.arange(n)
    return pd.Series(start * np.exp(daily_ret * x))


class MomentumTest(unittest.TestCase):
    def test_uptrend_positive_score(self):
        up = _trend(100, 100, 0.002)
        s = momentum_score(up, lookback=90)
        self.assertGreater(s, 0)

    def test_downtrend_negative_score(self):
        down = _trend(100, 100, -0.002)
        s = momentum_score(down, lookback=90)
        self.assertLess(s, 0)

    def test_stronger_trend_ranks_higher(self):
        strong = _trend(100, 100, 0.003)
        weak = _trend(100, 100, 0.0005)
        sideways = pd.Series(100 + np.sin(np.arange(100) / 5.0))
        ranking = rank_universe(
            {"STRONG": strong, "WEAK": weak, "SIDE": sideways},
            lookback=90,
        )
        self.assertEqual(ranking.top(1), ["STRONG"])
        self.assertIn(ranking.top(2)[1], ("WEAK", "SIDE"))

    def test_contains_respects_topk(self):
        ranking = rank_universe(
            {
                "A": _trend(100, 100, 0.003),
                "B": _trend(100, 100, 0.002),
                "C": _trend(100, 100, 0.001),
                "D": _trend(100, 100, -0.001),
            },
            lookback=90,
        )
        self.assertTrue(ranking.contains("A", 2))
        self.assertFalse(ranking.contains("D", 2))


if __name__ == "__main__":
    unittest.main()
