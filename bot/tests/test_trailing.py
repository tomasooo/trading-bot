"""Tests for the chandelier-exit / break-even trail state machine.

Run:  python -m unittest bot.tests.test_trailing
"""
import unittest

from bot.trailing import TrailState, update_trail, PROFIT_LOCK_R


def new_state(entry=100.0, stop=95.0, atr_entry=2.5):
    # Initial trail state as the engine/backtest would create at entry time
    return TrailState(
        entry=entry, initial_stop=stop, atr_at_entry=atr_entry,
        high_since_entry=entry, breakeven_active=False,
        profit_lock_active=False, current_stop=stop,
    )


class TrailTest(unittest.TestCase):
    def test_stop_never_loosens(self):
        s = new_state()
        s = update_trail(s, 110, 108, 3.0, trail_mult=3.0)
        first_stop = s.current_stop
        # Next bar: huge drop in high, but stop must not decrease
        s = update_trail(s, 100, 99, 3.0, trail_mult=3.0)
        self.assertGreaterEqual(s.current_stop, first_stop)

    def test_breakeven_after_1r(self):
        s = new_state(entry=100, stop=95)  # 1R = 5
        # price doesn't reach +1R
        s = update_trail(s, 103, 101, 2.5, trail_mult=3.0)
        self.assertFalse(s.breakeven_active)
        # price reaches entry + 1R
        s = update_trail(s, 106, 104, 2.5, trail_mult=3.0)
        self.assertTrue(s.breakeven_active)
        # Stop is at least entry (break-even)
        self.assertGreaterEqual(s.current_stop, 100.0)

    def test_profit_lock_flag_raised_at_2r(self):
        s = new_state(entry=100, stop=95, atr_entry=2.5)  # 1R = 5, 2R = 10
        # Below 2R -> flag off
        s = update_trail(s, 108, 106, 2.0, trail_mult=4.0)
        self.assertFalse(s.profit_lock_active)
        # At/above 2R -> flag on (informational only, doesn't tighten trail)
        s = update_trail(s, 110, 108, 2.0, trail_mult=4.0)
        self.assertTrue(s.profit_lock_active)
        # Chandelier = 110 - 4.0 * 2.0 = 102 (unchanged)
        self.assertAlmostEqual(s.current_stop, 110 - 4.0 * 2.0, places=4)

    def test_profit_lock_flag_sticky(self):
        s = new_state(entry=100, stop=95)
        s = update_trail(s, 112, 110, 2.0, trail_mult=4.0)  # >2R
        self.assertTrue(s.profit_lock_active)
        # Even if price comes back, flag stays on
        s = update_trail(s, 108, 107, 2.0, trail_mult=4.0)
        self.assertTrue(s.profit_lock_active)


if __name__ == "__main__":
    unittest.main()
