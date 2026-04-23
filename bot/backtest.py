"""Backtest harness with chandelier trailing stop, vol regime filter,
vol-targeted sizing, and simple slippage / commission modeling.

Run:
    python -m bot.backtest --symbol AAPL --days 180
    python -m bot.backtest --days 180        # all symbols in config
"""
from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass

import pandas as pd

from .config import Config
from .data import DataFeed
from .filters import clenow_filters_ok, volatility_regime_ok
from .indicators import atr, sma
from .logging_setup import setup_logging
from .momentum import rank_universe
from .risk import momentum_rank_scalar, plan_trade
from .strategies import Signal, StrategyContext, build_strategy
from .trailing import TrailState, update_trail


log = logging.getLogger("bot.backtest")

# Fixed per-side slippage in bps (2.5 bps = 0.025 %) — conservative for liquid US
# equities on IEX feed. Commission is $0 on Alpaca. We still charge a per-share
# "borrow" cost of 0 for simplicity (longs only).
SLIPPAGE_BPS = 2.5


def _apply_slippage(price: float, side: str) -> float:
    sign = 1.0 if side == "BUY" else -1.0
    return price * (1.0 + sign * SLIPPAGE_BPS / 10_000.0)


@dataclass
class BTTrade:
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    symbol: str
    qty: int
    entry: float
    exit: float
    pnl: float
    reason: str


_BARS_PER_YEAR = {
    "1Min": 252 * 390, "5Min": 252 * 78, "15Min": 252 * 26,
    "30Min": 252 * 13, "1Hour": 252 * 6, "1Day": 252,
}


def run_backtest(
    cfg: Config,
    symbol: str,
    days: int = 180,
    starting_equity: float = 100_000.0,
    spy_bars: pd.DataFrame | None = None,
    universe_closes: dict[str, pd.Series] | None = None,
) -> dict:
    feed = DataFeed(cfg.api_key, cfg.secret_key)

    tf = cfg.timeframe
    minutes_per_bar = {
        "1Min": 1, "5Min": 5, "15Min": 15, "30Min": 30,
        "1Hour": 60, "1Day": 60 * 24,
    }[tf]
    bars_per_day = max((6.5 * 60) // minutes_per_bar, 1)
    total_bars = int(bars_per_day * days) + 300

    bars = feed.bars(symbol, tf, lookback_bars=total_bars)
    if bars.empty:
        log.error("No bars returned for %s", symbol)
        return {}

    strategy = build_strategy(cfg.active_strategy,
                              cfg.strategies.get(cfg.active_strategy, {}),
                              all_params=cfg.strategies)
    equity = starting_equity
    peak_equity = starting_equity
    max_dd = 0.0
    open_trade: dict | None = None
    trades: list[BTTrade] = []
    equity_curve: list[tuple[pd.Timestamp, float]] = []

    required = max(strategy.required_bars(),
                   cfg.filters.vol_regime_lookback + 5, 25,
                   cfg.momentum.clenow_own_sma_period + 5)
    atr_series_full = atr(bars, 14)
    annualize = _BARS_PER_YEAR.get(tf, 252)

    # Pre-compute regime series (SPY close > SPY SMA-200) aligned to our bars
    regime_ok_series: pd.Series | None = None
    if cfg.market_regime.enabled and spy_bars is not None and not spy_bars.empty:
        spy_close = spy_bars["close"]
        spy_sma = sma(spy_close, cfg.market_regime.sma_period)
        spy_regime = (spy_close > spy_sma).rename("regime")
        # Forward-fill onto our bars' index (daily timeframe align trivial)
        regime_ok_series = spy_regime.reindex(bars.index, method="ffill").fillna(True)

    for i in range(required, len(bars)):
        window = bars.iloc[: i + 1]
        bar = bars.iloc[i]
        ts = bars.index[i]
        atr_now = float(atr_series_full.iloc[i]) if not pd.isna(atr_series_full.iloc[i]) else 0.0

        # --- manage open trade: trail stop + SL/TP hit intra-bar ---
        if open_trade is not None:
            state: TrailState = open_trade["state"]
            state = update_trail(
                state, float(bar["high"]), float(bar["low"]),
                atr_now, trail_mult=cfg.trail.atr_mult,
            )
            open_trade["state"] = state

            exit_price = None
            reason = ""
            # Hard take-profit ceiling (bracket TP)
            if bar["high"] >= open_trade["take_profit"]:
                exit_price = open_trade["take_profit"]
                reason = "TP"
            # Trail / initial stop (whichever is higher)
            elif bar["low"] <= state.current_stop:
                exit_price = state.current_stop
                reason = "TRAIL" if state.current_stop > state.initial_stop else "SL"
            # Clenow regime exit: close < own 100-SMA
            elif (cfg.momentum.enabled and cfg.momentum.exit_below_own_sma
                  and i + 1 >= cfg.momentum.clenow_own_sma_period):
                own_sma = window["close"].tail(
                    cfg.momentum.clenow_own_sma_period).mean()
                if float(bar["close"]) < own_sma:
                    exit_price = float(bar["close"])
                    reason = "SMA_EXIT"

            if exit_price is not None:
                exit_fill = _apply_slippage(exit_price, "SELL")
                pnl = (exit_fill - open_trade["entry_fill"]) * open_trade["qty"]
                equity += pnl
                trades.append(BTTrade(
                    entry_ts=open_trade["entry_ts"], exit_ts=ts,
                    symbol=symbol, qty=open_trade["qty"],
                    entry=open_trade["entry_fill"], exit=exit_fill,
                    pnl=pnl, reason=reason,
                ))
                open_trade = None

        # --- signal ---
        ctx = StrategyContext(symbol=symbol, in_position=open_trade is not None)
        sig = strategy.signal(window, ctx)

        if sig == Signal.SELL and open_trade is not None:
            exit_fill = _apply_slippage(float(bar["close"]), "SELL")
            pnl = (exit_fill - open_trade["entry_fill"]) * open_trade["qty"]
            equity += pnl
            trades.append(BTTrade(
                entry_ts=open_trade["entry_ts"], exit_ts=ts,
                symbol=symbol, qty=open_trade["qty"],
                entry=open_trade["entry_fill"], exit=exit_fill,
                pnl=pnl, reason="signal exit",
            ))
            open_trade = None

        elif open_trade is None and atr_now > 0:
            # Market regime (SPY 200-SMA) — hard gate
            if regime_ok_series is not None and not bool(regime_ok_series.loc[ts]):
                equity_curve.append((ts, equity))
                continue

            strategy_buy = sig == Signal.BUY

            # --- momentum ranking + tier-based entry gate ---
            tier1_ok = False
            current_rank: int | None = None
            if cfg.momentum.enabled and universe_closes:
                ranking_closes: dict[str, pd.Series] = {}
                # 12-1 blend wants 273+ bars; 252-day lookback with slack
                min_need = max(cfg.momentum.lookback_bars, 252) + 21
                for s, series in universe_closes.items():
                    sub = series.loc[:ts].tail(min_need + 5)
                    if len(sub) >= cfg.momentum.lookback_bars // 2:
                        ranking_closes[s] = sub
                if ranking_closes:
                    r = rank_universe(ranking_closes,
                                      lookback=cfg.momentum.lookback_bars,
                                      bars_per_year=annualize)
                    rank = r.rank_of(symbol)
                    if rank is None or rank > cfg.momentum.top_k:
                        equity_curve.append((ts, equity))
                        continue
                    current_rank = rank
                    # Tier 1 bypass: top-N ranked → enter on close > trend_sma.
                    # Tier 2: require the configured strategy signal.
                    if rank <= cfg.momentum.top_n_bypass:
                        trend_sma_val = sma(window["close"],
                                            cfg.momentum.trend_sma_period).iloc[-1]
                        if pd.isna(trend_sma_val) or float(bar["close"]) <= trend_sma_val:
                            equity_curve.append((ts, equity))
                            continue
                        tier1_ok = True
                    else:
                        if not strategy_buy:
                            equity_curve.append((ts, equity))
                            continue
            else:
                if not strategy_buy:
                    equity_curve.append((ts, equity))
                    continue

            # --- Clenow per-stock filters (gap + own-SMA) ---
            if cfg.momentum.enabled:
                cf = clenow_filters_ok(
                    window["close"],
                    gap_threshold=cfg.momentum.clenow_gap_threshold,
                    gap_lookback=cfg.momentum.clenow_gap_lookback,
                    sma_period=cfg.momentum.clenow_own_sma_period,
                )
                if not cf.ok:
                    equity_curve.append((ts, equity))
                    continue

            # ATR regime filter: skip high-vol blow-offs
            vr = volatility_regime_ok(
                atr_series_full.iloc[: i + 1],
                high_pct=cfg.filters.vol_regime_high_pct,
                lookback=cfg.filters.vol_regime_lookback,
            )
            if not vr.ok:
                equity_curve.append((ts, equity))
                continue

            mom_scalar = momentum_rank_scalar(current_rank, cfg.momentum.top_k)
            plan = plan_trade(
                float(bar["close"]), atr_now, equity, cfg.risk,
                close_series=window["close"] if cfg.vol_target.enabled else None,
                target_vol=cfg.vol_target.target_vol,
                momentum_scalar=mom_scalar,
            )
            if plan is not None:
                entry_fill = _apply_slippage(plan.entry, "BUY")
                open_trade = {
                    "entry_ts": ts,
                    "entry_fill": entry_fill,
                    "qty": plan.qty,
                    "take_profit": plan.take_profit,
                    "state": TrailState(
                        entry=entry_fill,
                        initial_stop=plan.stop_loss,
                        atr_at_entry=plan.atr_at_entry,
                        high_since_entry=entry_fill,
                        breakeven_active=False,
                        current_stop=plan.stop_loss,
                    ),
                }

        # Equity + DD tracking (mark-to-market would be fancier)
        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
        max_dd = max(max_dd, dd)
        equity_curve.append((ts, equity))

    # Flush trailing open position
    if open_trade is not None:
        last_ts = bars.index[-1]
        exit_fill = _apply_slippage(float(bars["close"].iloc[-1]), "SELL")
        pnl = (exit_fill - open_trade["entry_fill"]) * open_trade["qty"]
        equity += pnl
        trades.append(BTTrade(
            entry_ts=open_trade["entry_ts"], exit_ts=last_ts,
            symbol=symbol, qty=open_trade["qty"],
            entry=open_trade["entry_fill"], exit=exit_fill,
            pnl=pnl, reason="EOF",
        ))

    # --- stats ---
    if trades:
        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total_pnl = sum(pnls)
        win_rate = len(wins) / len(pnls)
        avg_win = (sum(wins) / len(wins)) if wins else 0.0
        avg_loss = (sum(losses) / len(losses)) if losses else 0.0
        profit_factor = (sum(wins) / -sum(losses)) if losses and sum(losses) != 0 else float("inf")
        expectancy = total_pnl / len(pnls)
    else:
        total_pnl = 0.0
        win_rate = avg_win = avg_loss = expectancy = 0.0
        profit_factor = float("nan")

    summary = {
        "symbol": symbol,
        "strategy": strategy.name,
        "bars": len(bars),
        "starting_equity": starting_equity,
        "ending_equity": equity,
        "total_pnl": total_pnl,
        "return_pct": (equity / starting_equity - 1) * 100,
        "trades": len(trades),
        "win_rate_pct": win_rate * 100,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "max_drawdown_pct": max_dd * 100,
    }
    return {"summary": summary, "trades": trades, "equity_curve": equity_curve}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=None,
                        help="Single symbol (default: all config symbols)")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--equity", type=float, default=100_000.0)
    parser.add_argument("--strategy", default=None,
                        help="Override active_strategy from config")
    parser.add_argument("--timeframe", default=None,
                        help="Override timeframe from config (1Min..1Day)")
    args = parser.parse_args()

    cfg = Config.load()
    if args.strategy:
        cfg.active_strategy = args.strategy
    if args.timeframe:
        cfg.timeframe = args.timeframe
    setup_logging(cfg.log_level, cfg.log_file)

    symbols = [args.symbol] if args.symbol else cfg.symbols

    # Pre-fetch SPY for regime and universe closes for momentum ranking.
    # One fetch per symbol instead of per-symbol-per-test.
    feed = DataFeed(cfg.api_key, cfg.secret_key)
    tf = cfg.timeframe
    minutes_per_bar = {
        "1Min": 1, "5Min": 5, "15Min": 15, "30Min": 30,
        "1Hour": 60, "1Day": 60 * 24,
    }[tf]
    bars_per_day = max((6.5 * 60) // minutes_per_bar, 1)
    total_bars = int(bars_per_day * args.days) + 300

    spy_bars = None
    if cfg.market_regime.enabled:
        try:
            spy_bars = feed.bars("SPY", tf, lookback_bars=total_bars)
            log.info("Fetched %d SPY bars for regime filter", len(spy_bars))
        except Exception as e:
            log.warning("SPY regime fetch failed: %s", e)

    universe_closes: dict[str, pd.Series] = {}
    if cfg.momentum.enabled:
        for sym in cfg.symbols:
            try:
                u = feed.bars(sym, tf, lookback_bars=total_bars)
                if not u.empty:
                    universe_closes[sym] = u["close"]
            except Exception as e:
                log.debug("Universe fetch %s failed: %s", sym, e)
        log.info("Universe closes for %d symbols", len(universe_closes))

    rows = []
    for sym in symbols:
        log.info("=== Backtesting %s (%s, %d days) ===",
                 sym, cfg.active_strategy, args.days)
        res = run_backtest(cfg, sym, days=args.days, starting_equity=args.equity,
                           spy_bars=spy_bars, universe_closes=universe_closes)
        if not res:
            continue
        s = res["summary"]
        rows.append(s)
        pf = f"{s['profit_factor']:.2f}" if math.isfinite(s['profit_factor']) else "inf"
        print(
            f"{s['symbol']:6} | {s['strategy']:<18} | "
            f"trades={s['trades']:3d}  win%={s['win_rate_pct']:5.1f}  "
            f"pnl=${s['total_pnl']:>10,.2f}  ret={s['return_pct']:+6.2f}%  "
            f"PF={pf:<6}  DD={s['max_drawdown_pct']:4.1f}%  "
            f"exp=${s['expectancy']:>7.2f}"
        )

    if len(rows) > 1:
        tot = sum(r["total_pnl"] for r in rows)
        tot_trades = sum(r["trades"] for r in rows)
        win_trades = sum(r["trades"] * r["win_rate_pct"] / 100 for r in rows)
        agg_win = (win_trades / tot_trades * 100) if tot_trades else 0
        print("-" * 110)
        print(
            f"TOTAL  | {'':<18} | trades={tot_trades:3d}  win%={agg_win:5.1f}  "
            f"pnl=${tot:>10,.2f}  ret={tot/args.equity*100:+6.2f}%"
        )

        # Buy-and-hold benchmark: equal-weight across universe over the same period.
        # Tells us whether our edge beats simply holding.
        bh_rets = []
        for sym, ser in universe_closes.items():
            if len(ser) < 2:
                continue
            bh_rets.append(ser.iloc[-1] / ser.iloc[0] - 1)
        if not bh_rets and spy_bars is not None and not spy_bars.empty:
            bh_rets = [spy_bars["close"].iloc[-1] / spy_bars["close"].iloc[0] - 1]
        if bh_rets:
            bh_ret = sum(bh_rets) / len(bh_rets) * 100
            edge = tot / args.equity * 100 - bh_ret
            print(f"Buy&hold (equal-weight universe): {bh_ret:+6.2f}%  "
                  f"=> strategy edge: {edge:+6.2f}%")


if __name__ == "__main__":
    main()
