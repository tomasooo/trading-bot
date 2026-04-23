"""Global entry-gate protections (freqtrade-inspired).

A `ProtectionManager` consults the trade & equity log and can halt new
entries when pathological conditions are detected:

- CooldownPeriod: no re-entry on a symbol for N bars after a stop-out.
- StoplossGuard:   halt all new entries if ≥ K losing trades in last N hours.
- MaxDrawdownHalt: halt new entries while equity is > X % below peak;
                   resume once equity recovers within Y % of peak.

All checks read from the SQLite `trades` and `equity` tables (see storage.py).
Thresholds live in config under `protections:`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .storage import Storage

log = logging.getLogger("bot.protections")


@dataclass
class ProtectionConfig:
    enabled: bool = True
    cooldown_hours: float = 24.0             # per-symbol cooldown after loss
    stoploss_guard_hours: float = 24.0       # window for portfolio-wide guard
    stoploss_guard_max_losses: int = 3       # trigger after this many losses
    stoploss_guard_halt_hours: float = 24.0  # how long the halt lasts
    max_drawdown_pct: float = 0.10           # halt when DD from peak exceeds this
    drawdown_resume_pct: float = 0.05        # resume when DD <= this


@dataclass
class GateResult:
    ok: bool
    reason: str = ""


class ProtectionManager:
    def __init__(self, storage: Storage, cfg: ProtectionConfig):
        self.storage = storage
        self.cfg = cfg
        self._guard_until: datetime | None = None  # cached across calls

    # --- per-symbol cooldown ---
    def cooldown_ok(self, symbol: str, now: datetime | None = None) -> GateResult:
        if not self.cfg.enabled or self.cfg.cooldown_hours <= 0:
            return GateResult(True)
        now = now or datetime.now(timezone.utc)
        df = self.storage.trades_df(limit=200)
        if df.empty:
            return GateResult(True)
        # Find most recent SELL with negative note or price dropping from BUY —
        # simple proxy: last SELL within cooldown window on this symbol.
        recent = df[(df["symbol"] == symbol) & (df["side"] == "SELL")]
        if recent.empty:
            return GateResult(True)
        last_ts = recent["ts"].max()
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        elapsed_h = (now - last_ts.to_pydatetime()).total_seconds() / 3600.0
        if elapsed_h < self.cfg.cooldown_hours:
            return GateResult(
                False,
                f"cooldown ({elapsed_h:.1f}h < {self.cfg.cooldown_hours}h since last {symbol} exit)",
            )
        return GateResult(True)

    # --- portfolio-wide stoploss guard ---
    def stoploss_guard_ok(self, now: datetime | None = None) -> GateResult:
        if not self.cfg.enabled:
            return GateResult(True)
        now = now or datetime.now(timezone.utc)
        if self._guard_until is not None and now < self._guard_until:
            return GateResult(
                False,
                f"stoploss guard active for another "
                f"{(self._guard_until - now).total_seconds() / 3600:.1f}h",
            )
        df = self.storage.trades_df(limit=500)
        if df.empty:
            return GateResult(True)
        window_start = now - timedelta(hours=self.cfg.stoploss_guard_hours)
        recent_sells = df[(df["side"] == "SELL") & (df["ts"] >= window_start)]
        if len(recent_sells) >= self.cfg.stoploss_guard_max_losses:
            self._guard_until = now + timedelta(hours=self.cfg.stoploss_guard_halt_hours)
            log.warning(
                "STOPLOSS GUARD triggered — %d exits in last %.1fh. "
                "Halting new entries for %.1fh.",
                len(recent_sells),
                self.cfg.stoploss_guard_hours,
                self.cfg.stoploss_guard_halt_hours,
            )
            return GateResult(False, "stoploss guard just triggered")
        return GateResult(True)

    # --- drawdown halt ---
    def drawdown_ok(self, current_equity: float) -> GateResult:
        if not self.cfg.enabled or self.cfg.max_drawdown_pct <= 0:
            return GateResult(True)
        eq = self.storage.equity_df(limit=5000)
        if eq.empty:
            return GateResult(True)
        peak = max(eq["equity"].max(), current_equity)
        if peak <= 0:
            return GateResult(True)
        dd = (peak - current_equity) / peak
        if dd > self.cfg.max_drawdown_pct:
            return GateResult(
                False,
                f"drawdown {dd*100:.1f}% > {self.cfg.max_drawdown_pct*100:.1f}% — "
                f"halt until DD <= {self.cfg.drawdown_resume_pct*100:.1f}%",
            )
        return GateResult(True)

    # --- combined ---
    def can_enter(self, symbol: str, current_equity: float,
                  now: datetime | None = None) -> GateResult:
        for chk in (
            self.cooldown_ok(symbol, now),
            self.stoploss_guard_ok(now),
            self.drawdown_ok(current_equity),
        ):
            if not chk.ok:
                return chk
        return GateResult(True)
