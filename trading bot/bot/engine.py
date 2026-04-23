"""Main trading loop.

Each iteration:
1. Check market is open (or bail + sleep).
2. Snapshot account + positions.
3. Manage existing positions (trailing stop / break-even exit).
4. For each symbol with no open position:
     - fetch bars
     - apply regime filters (time, vol regime, sector cap)
     - ask strategy for a signal
     - on BUY: size + submit bracket order (vol-targeted)
5. Record equity snapshot.
6. Sleep poll_interval.

Safety layering:
- Bracket SL on the broker = disaster stop even if bot dies.
- Client-side chandelier trail tightens the exit on winners.
- Daily loss breaker halts new entries after `max_daily_loss_pct`.
"""
from __future__ import annotations

import logging
import time as time_mod
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from .broker import Broker
from .config import Config
from .data import DataFeed
from .filters import (
    clenow_filters_ok, sector_cap, time_filter, volatility_regime_ok,
)
from .indicators import atr, sma
from .momentum import rank_universe
from .protections import ProtectionConfig, ProtectionManager
from .regime import market_regime_ok, market_regime_score
from .risk import momentum_rank_scalar, plan_trade
from .storage import Storage
from .strategies import Signal, StrategyContext, build_strategy
from .trailing import TrailState, update_trail

log = logging.getLogger("bot.engine")


_BARS_PER_YEAR = {
    "1Min": 252 * 390, "5Min": 252 * 78, "15Min": 252 * 26,
    "30Min": 252 * 13, "1Hour": 252 * 6, "1Day": 252,
}


class Engine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.broker = Broker(cfg.api_key, cfg.secret_key, paper=cfg.paper)
        self.data = DataFeed(cfg.api_key, cfg.secret_key)
        self.storage = Storage(cfg.trades_db)
        self.strategy = build_strategy(
            cfg.active_strategy,
            cfg.strategies.get(cfg.active_strategy, {}),
            all_params=cfg.strategies,
        )
        self.protections = ProtectionManager(
            self.storage,
            ProtectionConfig(
                enabled=cfg.protections.enabled,
                cooldown_hours=cfg.protections.cooldown_hours,
                stoploss_guard_hours=cfg.protections.stoploss_guard_hours,
                stoploss_guard_max_losses=cfg.protections.stoploss_guard_max_losses,
                stoploss_guard_halt_hours=cfg.protections.stoploss_guard_halt_hours,
                max_drawdown_pct=cfg.protections.max_drawdown_pct,
                drawdown_resume_pct=cfg.protections.drawdown_resume_pct,
            ),
        )
        self._starting_equity_today: Optional[float] = None
        self._current_day: Optional[str] = None
        self._halted_today = False
        self._cached_universe_bars: dict[str, pd.DataFrame] = {}
        self._cached_universe_ts: Optional[datetime] = None
        self._cached_ranking = None
        self._cached_spy: Optional[pd.DataFrame] = None
        self._cached_spy_ts: Optional[datetime] = None

    # ----- daily circuit breaker -----
    def _check_daily_loss(self, equity: float) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._current_day:
            self._current_day = today
            self._starting_equity_today = equity
            self._halted_today = False
            log.info("New trading day %s — starting equity $%.2f", today, equity)
            return
        if self._starting_equity_today is None:
            self._starting_equity_today = equity
            return
        loss_pct = (self._starting_equity_today - equity) / self._starting_equity_today
        if loss_pct >= self.cfg.risk.max_daily_loss_pct and not self._halted_today:
            self._halted_today = True
            log.warning(
                "DAILY LOSS CIRCUIT BREAKER tripped — loss %.2f%% >= %.2f%%. "
                "New entries halted for the day.",
                loss_pct * 100, self.cfg.risk.max_daily_loss_pct * 100,
            )

    # ----- market regime (SPY) cache -----
    def _get_spy_bars(self) -> pd.DataFrame | None:
        """Fetch SPY daily bars once per day (cached)."""
        if not self.cfg.market_regime.enabled:
            return None
        now = datetime.now(timezone.utc)
        if (self._cached_spy is not None and self._cached_spy_ts is not None
                and (now - self._cached_spy_ts).total_seconds() < 3600):
            return self._cached_spy
        try:
            spy = self.data.bars("SPY", "1Day",
                                 lookback_bars=self.cfg.market_regime.sma_period + 20)
            self._cached_spy = spy
            self._cached_spy_ts = now
            return spy
        except Exception as e:
            log.warning("Failed to fetch SPY regime bars: %s", e)
            return None

    # ----- universe ranking cache -----
    def _get_ranking(self):
        """Re-rank universe at most once per hour."""
        if not self.cfg.momentum.enabled:
            return None
        now = datetime.now(timezone.utc)
        if (self._cached_ranking is not None and self._cached_universe_ts is not None
                and (now - self._cached_universe_ts).total_seconds() < 3600):
            return self._cached_ranking
        closes: dict[str, pd.Series] = {}
        # 12-1 skip-month blend needs 252 + 21 bars; bump lookback accordingly.
        need = max(self.cfg.momentum.lookback_bars, 252) + 30
        for sym in self.cfg.symbols:
            try:
                b = self.data.bars(sym, self.cfg.timeframe, lookback_bars=need)
                if not b.empty:
                    closes[sym] = b["close"]
            except Exception as e:
                log.debug("Ranking fetch failed for %s: %s", sym, e)
        if not closes:
            return None
        bars_per_year = {
            "1Min": 252 * 390, "5Min": 252 * 78, "15Min": 252 * 26,
            "30Min": 252 * 13, "1Hour": 252 * 6, "1Day": 252,
        }.get(self.cfg.timeframe, 252)
        ranking = rank_universe(closes, lookback=self.cfg.momentum.lookback_bars,
                                bars_per_year=bars_per_year)
        top = ranking.top(self.cfg.momentum.top_k)
        log.info("Momentum top-%d: %s", self.cfg.momentum.top_k, top)
        self._cached_ranking = ranking
        self._cached_universe_ts = now
        return ranking

    # ----- trailing stop management -----
    def _manage_trailing(self, symbol: str, bars: pd.DataFrame) -> bool:
        """Returns True if the position was closed by the trail."""
        if not self.cfg.trail.enabled:
            return False
        state = self.storage.load_trail(symbol)
        if state is None:
            return False

        a_series = atr(bars, 14)
        current_atr = float(a_series.iloc[-1]) if not pd.isna(a_series.iloc[-1]) else state.atr_at_entry
        last_bar = bars.iloc[-1]
        new_state = update_trail(
            state, float(last_bar["high"]), float(last_bar["low"]),
            current_atr, trail_mult=self.cfg.trail.atr_mult,
        )
        current_price = float(last_bar["close"])

        if current_price <= new_state.current_stop:
            log.info(
                "%s TRAIL EXIT — price %.2f <= stop %.2f (entry %.2f, high %.2f, BE=%s)",
                symbol, current_price, new_state.current_stop,
                new_state.entry, new_state.high_since_entry, new_state.breakeven_active,
            )
            self.broker.close_position(symbol)
            self.storage.record_trade(
                symbol=symbol, side="SELL", qty=0, price=current_price,
                strategy=self.strategy.name, note="trail/be exit",
            )
            self.storage.delete_trail(symbol)
            return True

        # Clenow regime exit: force-close if close drops below own N-SMA
        if (self.cfg.momentum.enabled and self.cfg.momentum.exit_below_own_sma
                and len(bars) >= self.cfg.momentum.clenow_own_sma_period):
            own_sma_val = sma(bars["close"],
                              self.cfg.momentum.clenow_own_sma_period).iloc[-1]
            if not pd.isna(own_sma_val) and current_price < own_sma_val:
                log.info("%s SMA EXIT — price %.2f below own %d-SMA %.2f",
                         symbol, current_price,
                         self.cfg.momentum.clenow_own_sma_period, own_sma_val)
                self.broker.close_position(symbol)
                self.storage.record_trade(
                    symbol=symbol, side="SELL", qty=0, price=current_price,
                    strategy=self.strategy.name, note="sma regime exit",
                )
                self.storage.delete_trail(symbol)
                return True

        self.storage.save_trail(symbol, new_state)
        return False

    # ----- per-symbol -----
    def _process_symbol(self, symbol: str, equity: float,
                         positions: list, open_count: int) -> None:
        try:
            bars = self.data.bars(
                symbol, self.cfg.timeframe,
                lookback_bars=max(self.strategy.required_bars() + 120, 300),
            )
        except Exception as e:
            log.warning("data.bars(%s) failed: %s", symbol, e)
            return

        if bars.empty or len(bars) < self.strategy.required_bars():
            return

        # 1) Manage existing trail first — may close the position
        position = self.broker.position_for(symbol)
        if position is not None and position.qty != 0:
            if self._manage_trailing(symbol, bars):
                return
            # Also allow strategy to signal SELL (e.g. bearish cross)
            ctx = StrategyContext(symbol=symbol, in_position=True)
            sig = self.strategy.signal(bars, ctx)
            if sig == Signal.SELL:
                self.broker.close_position(symbol)
                self.storage.record_trade(
                    symbol=symbol, side="SELL", qty=position.qty,
                    price=float(bars["close"].iloc[-1]),
                    strategy=self.strategy.name, note="strategy exit",
                )
                self.storage.delete_trail(symbol)
            return

        # 2) Entry evaluation — compute strategy signal up front but gate it
        #    based on rank tier. Clenow's "Stocks on the Move" enters top-decile
        #    names on trend alone (no breakout). We apply that to the top-N,
        #    and fall back to the configured strategy signal for the rest of
        #    the top-K universe.
        ctx = StrategyContext(symbol=symbol, in_position=False)
        strategy_buy = self.strategy.signal(bars, ctx) == Signal.BUY

        if self._halted_today:
            return
        if open_count >= self.cfg.risk.max_open_positions:
            return

        # --- market regime (SPY 200-SMA master switch) ---
        regime_scalar = 1.0
        if self.cfg.market_regime.enabled:
            spy = self._get_spy_bars()
            if self.cfg.market_regime.use_soft_score:
                regime_scalar = market_regime_score(spy, self.cfg.market_regime.sma_period)
                if regime_scalar < 1.0:
                    log.info("%s market regime weak — size scaled to %.2f",
                             symbol, regime_scalar)
            else:
                if not market_regime_ok(spy, self.cfg.market_regime.sma_period):
                    log.info("%s BUY skipped — SPY below %d-SMA",
                             symbol, self.cfg.market_regime.sma_period)
                    return

        # --- momentum ranking + tier-based entry gate ---
        current_rank: int | None = None
        if self.cfg.momentum.enabled:
            ranking = self._get_ranking()
            if ranking is None:
                if not strategy_buy:
                    return
            else:
                rank = ranking.rank_of(symbol)
                if rank is None or rank > self.cfg.momentum.top_k:
                    log.info("%s BUY skipped — not in momentum top-%d",
                             symbol, self.cfg.momentum.top_k)
                    return
                current_rank = rank
                # Tier 1 (top-N): enter on close above own trend SMA, no
                # breakout required. Tier 2: require the configured strategy.
                if rank <= self.cfg.momentum.top_n_bypass:
                    trend_sma = sma(bars["close"],
                                    self.cfg.momentum.trend_sma_period).iloc[-1]
                    if pd.isna(trend_sma) or bars["close"].iloc[-1] <= trend_sma:
                        log.info("%s (rank %d) BUY skipped — below %d-SMA",
                                 symbol, rank, self.cfg.momentum.trend_sma_period)
                        return
                else:
                    if not strategy_buy:
                        return
        else:
            # Momentum disabled — fall back to pure strategy gating
            if not strategy_buy:
                return

        # --- Clenow per-stock filters (gap + own-SMA) ---
        if self.cfg.momentum.enabled:
            cf = clenow_filters_ok(
                bars["close"],
                gap_threshold=self.cfg.momentum.clenow_gap_threshold,
                gap_lookback=self.cfg.momentum.clenow_gap_lookback,
                sma_period=self.cfg.momentum.clenow_own_sma_period,
            )
            if not cf.ok:
                log.info("%s BUY skipped — %s", symbol, cf.reason)
                return

        # --- protections ---
        gate = self.protections.can_enter(symbol, equity)
        if not gate.ok:
            log.info("%s BUY skipped — %s", symbol, gate.reason)
            return

        # --- filters ---
        if self.cfg.filters.skip_session_edges:
            tf = time_filter()
            if not tf.ok:
                log.info("%s BUY skipped — %s", symbol, tf.reason)
                return

        a_series = atr(bars, 14)
        atr_value = float(a_series.iloc[-1]) if not pd.isna(a_series.iloc[-1]) else 0.0
        if atr_value <= 0:
            return

        vr = volatility_regime_ok(
            a_series, high_pct=self.cfg.filters.vol_regime_high_pct,
            lookback=self.cfg.filters.vol_regime_lookback,
        )
        if not vr.ok:
            log.info("%s BUY skipped — %s", symbol, vr.reason)
            return

        # --- sizing ---
        price = float(bars["close"].iloc[-1])
        close_series = bars["close"] if self.cfg.vol_target.enabled else None
        mom_scalar = momentum_rank_scalar(current_rank, self.cfg.momentum.top_k)
        plan = plan_trade(
            price, atr_value, equity, self.cfg.risk,
            close_series=close_series,
            target_vol=self.cfg.vol_target.target_vol,
            momentum_scalar=mom_scalar,
        )
        if plan is None:
            return
        # Apply regime soft-scaling to quantity
        if regime_scalar < 1.0 and plan.qty > 1:
            new_qty = max(1, int(plan.qty * regime_scalar))
            plan = plan.__class__(
                qty=new_qty, entry=plan.entry, stop_loss=plan.stop_loss,
                take_profit=plan.take_profit,
                risk_usd=plan.risk_usd * (new_qty / plan.qty),
                notional=new_qty * plan.entry,
                atr_at_entry=plan.atr_at_entry, vol_scalar=plan.vol_scalar,
            )

        # sector cap after we know notional
        sc = sector_cap(symbol, positions, equity, plan.notional,
                        max_pct=self.cfg.filters.sector_max_pct)
        if not sc.ok:
            log.info("%s BUY skipped — %s", symbol, sc.reason)
            return

        # --- submit ---
        try:
            self.broker.submit_bracket_buy(
                symbol, plan.qty, plan.stop_loss, plan.take_profit,
            )
            self.storage.record_trade(
                symbol=symbol, side="BUY", qty=plan.qty, price=price,
                stop_loss=plan.stop_loss, take_profit=plan.take_profit,
                strategy=self.strategy.name,
                note=f"risk=${plan.risk_usd:.2f} notional=${plan.notional:.2f} "
                     f"vol_scalar={plan.vol_scalar:.2f}",
            )
            # Initialize trail state
            self.storage.save_trail(symbol, TrailState(
                entry=plan.entry, initial_stop=plan.stop_loss,
                atr_at_entry=plan.atr_at_entry, high_since_entry=plan.entry,
                breakeven_active=False, current_stop=plan.stop_loss,
            ))
        except Exception as e:
            log.warning("submit_bracket_buy(%s) failed: %s", symbol, e)

    # ----- main loop -----
    def run(self) -> None:
        log.info(
            "Engine starting — strategy=%s symbols=%s timeframe=%s paper=%s",
            self.strategy.name, self.cfg.symbols, self.cfg.timeframe, self.cfg.paper,
        )
        while True:
            try:
                self._tick()
            except KeyboardInterrupt:
                log.info("Interrupted — shutting down cleanly.")
                break
            except Exception as e:
                log.exception("Tick failed: %s", e)
            time_mod.sleep(self.cfg.poll_interval_seconds)

    def _tick(self) -> None:
        if self.cfg.market_hours_only and not self.broker.is_market_open():
            acct = self.broker.account()
            self.storage.record_equity(acct.equity, acct.cash,
                                       len(self.broker.positions()))
            return

        acct = self.broker.account()
        self._check_daily_loss(acct.equity)
        positions = self.broker.positions()
        self.storage.record_equity(acct.equity, acct.cash, len(positions))

        for symbol in self.cfg.symbols:
            try:
                current_positions = self.broker.positions()
                self._process_symbol(
                    symbol, acct.equity, current_positions, len(current_positions),
                )
            except Exception as e:
                # Isolate per-symbol failures so one bad ticker doesn't kill
                # the tick for the rest of the universe.
                log.exception("Symbol %s raised during tick: %s", symbol, e)
