"""Walk-forward validation harness.

Runs the backtest across all symbols, aggregates their equity curves into a
single portfolio equity series, and splits that series into consecutive
windows. For each window we report the Sharpe ratio and total return.

If Sharpe / return are consistent in sign across windows, the edge is
robust. Wild sign flips across windows = curve-fit to the last regime.

This variant does NOT auto-optimize parameters per window — it validates
that the config as-is generalizes across different market regimes.

Run:
    python -m bot.walkforward --days 1500 --test-days 180
    python -m bot.walkforward --days 1500 --test-days 365
"""
from __future__ import annotations

import argparse
import logging
import math

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .config import Config
from .data import DataFeed
from .logging_setup import setup_logging

log = logging.getLogger("bot.walkforward")


def _sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    if sigma < 1e-12:
        return 0.0
    return float(mu / sigma * math.sqrt(periods_per_year))


def _equity_to_returns(eqs: np.ndarray) -> np.ndarray:
    if len(eqs) < 2:
        return np.array([])
    rets = np.diff(eqs) / eqs[:-1]
    return rets[np.isfinite(rets)]


def _aggregate_equity(curves: list[list[tuple[pd.Timestamp, float]]],
                      starting_equity: float) -> pd.Series:
    """Combine per-symbol equity curves into a single portfolio equity series.

    Each symbol's curve is its PnL path. We sum per-symbol PnL (symbol_equity -
    starting_equity) across symbols at each timestamp to get total portfolio
    PnL, then add back starting_equity. Missing timestamps are forward-filled.
    """
    if not curves:
        return pd.Series(dtype=float)
    series_list = []
    for c in curves:
        if not c:
            continue
        ts, eq = zip(*c)
        s = pd.Series(eq, index=pd.DatetimeIndex(ts)).sort_index()
        # Deduplicate any repeated timestamps (defensive)
        s = s[~s.index.duplicated(keep="last")]
        series_list.append(s - starting_equity)  # reduce to PnL
    if not series_list:
        return pd.Series(dtype=float)
    df = pd.concat(series_list, axis=1).sort_index().ffill().fillna(0.0)
    pnl_total = df.sum(axis=1)
    return pnl_total + starting_equity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=1500,
                        help="Total lookback window (days)")
    parser.add_argument("--test-days", type=int, default=180,
                        help="OOS window length per segment")
    parser.add_argument("--equity", type=float, default=100_000.0)
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--timeframe", default=None)
    args = parser.parse_args()

    cfg = Config.load()
    if args.strategy:
        cfg.active_strategy = args.strategy
    if args.timeframe:
        cfg.timeframe = args.timeframe
    setup_logging(cfg.log_level, cfg.log_file)

    # Pre-fetch regime + universe once
    feed = DataFeed(cfg.api_key, cfg.secret_key)
    tf = cfg.timeframe
    minutes_per_bar = {"1Min": 1, "5Min": 5, "15Min": 15, "30Min": 30,
                       "1Hour": 60, "1Day": 60 * 24}[tf]
    bars_per_day = max((6.5 * 60) // minutes_per_bar, 1)
    total_bars = int(bars_per_day * args.days) + 300

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

    # Run backtest for each symbol, collect equity curves
    print(f"\nPortfolio walk-forward: total={args.days}d, test={args.test_days}d")
    all_curves = []
    for sym in cfg.symbols:
        res = run_backtest(cfg, sym, days=args.days,
                           starting_equity=args.equity,
                           spy_bars=spy_bars,
                           universe_closes=universe_closes)
        if res and res.get("equity_curve"):
            all_curves.append(res["equity_curve"])

    portfolio = _aggregate_equity(all_curves, args.equity)
    if portfolio.empty:
        print("No equity data — nothing to validate.")
        return

    # Periods per year for Sharpe
    annualize = {"1Min": 252 * 390, "5Min": 252 * 78, "15Min": 252 * 26,
                 "30Min": 252 * 13, "1Hour": 252 * 6, "1Day": 252}.get(tf, 252)

    # Split into n non-overlapping windows of ~test_days each (by calendar time)
    start = portfolio.index.min()
    end = portfolio.index.max()
    window = pd.Timedelta(days=args.test_days)
    edges = []
    t = start
    while t < end:
        edges.append(t)
        t += window
    edges.append(end)

    print(f"Windows: {len(edges) - 1}\n")
    window_rows = []
    for i in range(len(edges) - 1):
        seg = portfolio.loc[edges[i]:edges[i + 1]]
        if len(seg) < 2:
            continue
        rets = _equity_to_returns(seg.values)
        sh = _sharpe(rets, periods_per_year=annualize)
        ret_pct = (seg.iloc[-1] / seg.iloc[0] - 1) * 100
        window_rows.append((i + 1, edges[i].date(), edges[i + 1].date(), sh, ret_pct))
        print(f"  window {i + 1} [{edges[i].date()} -> {edges[i + 1].date()}]: "
              f"Sharpe={sh:+.2f}  return={ret_pct:+.2f}%")

    if window_rows:
        sharpes = [r[3] for r in window_rows]
        rets = [r[4] for r in window_rows]
        pos = sum(1 for r in rets if r > 0)
        n = len(rets)
        full_rets = _equity_to_returns(portfolio.values)
        full_sharpe = _sharpe(full_rets, periods_per_year=annualize)
        full_ret = (portfolio.iloc[-1] / portfolio.iloc[0] - 1) * 100

        print("\n=== Portfolio walk-forward summary ===")
        print(f"Full-period Sharpe:     {full_sharpe:+.2f}")
        print(f"Full-period return:     {full_ret:+.2f}%")
        print(f"Avg window Sharpe:      {np.mean(sharpes):+.2f} "
              f"(std {np.std(sharpes, ddof=1) if len(sharpes) > 1 else 0:.2f})")
        print(f"Avg window return:      {np.mean(rets):+.2f}%")
        print(f"Positive windows:       {pos}/{n}  ({pos / n * 100:.0f}%)")
        print("\nRule of thumb: Sharpe > 0.5 is acceptable, > 1.0 is good;")
        print("we want >=60% positive windows and no huge losing window.")


if __name__ == "__main__":
    main()
