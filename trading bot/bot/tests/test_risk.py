"""Tests for position sizing: vol-targeted qty, stop/TP placement, caps.

Run:  python -m unittest bot.tests.test_risk
"""
import unittest

import numpy as np
import pandas as pd

from bot.config import RiskConfig
from bot.risk import plan_trade, realized_vol, vol_targeted_scalar


class RiskTest(unittest.TestCase):
    def test_risk_per_trade_respected(self):
        cfg = RiskConfig(risk_per_trade_pct=0.01, stop_loss_atr_mult=2.0,
                         take_profit_atr_mult=4.0, max_position_pct=0.5)
        plan = plan_trade(price=100, atr_value=2, equity=100_000,
                          risk_cfg=cfg, close_series=None)
        self.assertIsNotNone(plan)
        # risk_usd should be close to equity * risk_per_trade_pct (after int floor)
        self.assertAlmostEqual(plan.risk_usd, 1000, delta=10)
        self.assertAlmostEqual(plan.stop_loss, 96.0)
        self.assertAlmostEqual(plan.take_profit, 108.0)

    def test_max_position_pct_caps_qty(self):
        cfg = RiskConfig(risk_per_trade_pct=0.05,  # large risk budget
                         stop_loss_atr_mult=0.1,   # tiny stop -> otherwise huge qty
                         take_profit_atr_mult=4.0,
                         max_position_pct=0.1)     # cap at 10%
        plan = plan_trade(price=100, atr_value=2, equity=100_000,
                          risk_cfg=cfg, close_series=None)
        self.assertIsNotNone(plan)
        # Notional should not exceed 10% of equity
        self.assertLessEqual(plan.notional, 100_000 * 0.1 + 1e-6)

    def test_rejects_nonpositive_inputs(self):
        cfg = RiskConfig()
        self.assertIsNone(plan_trade(0, 2, 100_000, cfg))
        self.assertIsNone(plan_trade(100, 0, 100_000, cfg))
        self.assertIsNone(plan_trade(100, 2, 0, cfg))

    def test_vol_scalar_bounded(self):
        # Extremely low vol -> scalar clamped at 2.0
        flat = pd.Series([100.0] * 30)
        low_s = vol_targeted_scalar(flat, target_vol=0.2)
        self.assertLessEqual(low_s, 2.0 + 1e-6)
        # Extremely high vol -> scalar clamped at 0.25
        rng = np.random.default_rng(42)
        noisy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.2, 100))))
        high_s = vol_targeted_scalar(noisy, target_vol=0.01, annualize=252)
        self.assertGreaterEqual(high_s, 0.25 - 1e-6)

    def test_realized_vol_roughly_matches(self):
        # Synthetic series with known daily vol of 1% -> annualized ~ 0.01 * sqrt(252)
        rng = np.random.default_rng(0)
        daily_rets = rng.normal(0, 0.01, 5000)
        prices = pd.Series(100 * np.exp(np.cumsum(daily_rets)))
        rv = realized_vol(prices, lookback=1000, annualize=252)
        self.assertAlmostEqual(rv, 0.01 * (252 ** 0.5), delta=0.05)


if __name__ == "__main__":
    unittest.main()
