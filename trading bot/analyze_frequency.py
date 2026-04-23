"""Quick analysis: how often does the default config open/close trades?

Runs the default 1500-day backtest, then dumps per-trade holding-period and
frequency statistics. Not imported by the bot — just a local script.
"""
from __future__ import annotations

import statistics
from collections import Counter

import pandas as pd

from bot.backtest import run_backtest
from bot.config import Config
from bot.data import DataFeed
from bot.logging_setup import setup_logging


def main() -> None:
    cfg = Config.load()
    setup_logging("WARNING", cfg.log_file)

    days = 1500
    feed = DataFeed(cfg.api_key, cfg.secret_key)
    tf = cfg.timeframe
    minutes_per_bar = {"1Min": 1, "5Min": 5, "15Min": 15, "30Min": 30,
                       "1Hour": 60, "1Day": 60 * 24}[tf]
    bars_per_day = max((6.5 * 60) // minutes_per_bar, 1)
    total_bars = int(bars_per_day * days) + 300

    spy_bars = feed.bars("SPY", tf, lookback_bars=total_bars) if cfg.market_regime.enabled else None
    universe_closes: dict[str, pd.Series] = {}
    if cfg.momentum.enabled:
        for sym in cfg.symbols:
            try:
                u = feed.bars(sym, tf, lookback_bars=total_bars)
                if not u.empty:
                    universe_closes[sym] = u["close"]
            except Exception:
                pass

    all_trades = []
    for sym in cfg.symbols:
        res = run_backtest(cfg, sym, days=days, starting_equity=100_000.0,
                           spy_bars=spy_bars, universe_closes=universe_closes)
        if res and res.get("trades"):
            all_trades.extend(res["trades"])

    if not all_trades:
        print("No trades in backtest.")
        return

    durations_days = []
    reasons = Counter()
    by_year = Counter()
    for t in all_trades:
        delta = t.exit_ts - t.entry_ts
        days_held = delta.total_seconds() / 86400.0
        durations_days.append(days_held)
        reasons[t.reason] += 1
        by_year[t.entry_ts.year] += 1

    n = len(all_trades)
    backtest_years = days / 365.25
    print(f"=== Trade-frequency analysis (default config, {days}d = "
          f"{backtest_years:.1f} years) ===\n")
    print(f"Total trades:            {n}")
    print(f"Trades per year (avg):   {n / backtest_years:.1f}")
    print(f"Trades per week (avg):   {n / (backtest_years * 52):.2f}")
    print(f"Trades per day (avg):    {n / (backtest_years * 252):.2f}  "
          f"(251 US trading days per year)")
    print()
    print("Holding-period distribution (days held per trade):")
    print(f"  min:     {min(durations_days):6.1f} days")
    print(f"  p25:     {statistics.quantiles(durations_days, n=4)[0]:6.1f} days")
    print(f"  median:  {statistics.median(durations_days):6.1f} days")
    print(f"  mean:    {statistics.mean(durations_days):6.1f} days")
    print(f"  p75:     {statistics.quantiles(durations_days, n=4)[2]:6.1f} days")
    print(f"  p95:     {statistics.quantiles(durations_days, n=20)[18]:6.1f} days")
    print(f"  max:     {max(durations_days):6.1f} days")
    print()
    print("Exit-reason mix:")
    for r, c in reasons.most_common():
        print(f"  {r:>14}  {c:4d}  ({c / n * 100:5.1f}%)")
    print()
    print("Entries by year:")
    for y in sorted(by_year):
        print(f"  {y}: {by_year[y]:3d} trades")


if __name__ == "__main__":
    main()
