"""Config loader — reads config.yaml + .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 0.01
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 4.0
    max_open_positions: int = 5
    max_position_pct: float = 0.20
    max_daily_loss_pct: float = 0.03


@dataclass
class TrailingConfig:
    enabled: bool = True
    atr_mult: float = 3.0
    breakeven_r_mult: float = 1.0


@dataclass
class VolTargetConfig:
    enabled: bool = True
    target_vol: float = 0.20
    lookback_bars: int = 20


@dataclass
class FiltersConfig:
    skip_session_edges: bool = True
    vol_regime_high_pct: float = 0.95
    vol_regime_lookback: int = 100
    sector_max_pct: float = 0.30


@dataclass
class MarketRegimeConfig:
    enabled: bool = True
    sma_period: int = 200
    use_soft_score: bool = False


@dataclass
class MomentumConfig:
    enabled: bool = True
    lookback_bars: int = 90
    top_k: int = 10
    # Tier-1 bypass: for the top-N ranked symbols, skip the strategy (Donchian
    # breakout) gate and enter on "close > trend_sma_period SMA" instead. This
    # is Clenow's original entry — ranking + trend = entry, no breakout event
    # required. Set top_n_bypass=0 to disable and always require strategy.
    top_n_bypass: int = 5
    trend_sma_period: int = 50
    # Clenow per-stock filters
    clenow_gap_threshold: float = 0.15
    clenow_gap_lookback: int = 90
    clenow_own_sma_period: int = 100
    # Force-exit rule: close position if its close drops below own 100-SMA
    exit_below_own_sma: bool = True


@dataclass
class ProtectionsConfig:
    enabled: bool = True
    cooldown_hours: float = 48.0
    stoploss_guard_hours: float = 24.0
    stoploss_guard_max_losses: int = 3
    stoploss_guard_halt_hours: float = 24.0
    max_drawdown_pct: float = 0.08
    drawdown_resume_pct: float = 0.04


@dataclass
class Config:
    api_key: str
    secret_key: str
    paper: bool
    symbols: list[str]
    timeframe: str
    poll_interval_seconds: int
    active_strategy: str
    strategies: dict[str, dict[str, Any]]
    risk: RiskConfig
    trail: TrailingConfig
    vol_target: VolTargetConfig
    filters: FiltersConfig
    market_regime: MarketRegimeConfig
    momentum: MomentumConfig
    protections: ProtectionsConfig
    market_hours_only: bool
    log_level: str
    log_file: str
    trades_db: str

    @classmethod
    def load(cls, config_path: Path | None = None) -> "Config":
        load_dotenv(ROOT / ".env")
        config_path = config_path or (ROOT / "config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        api_key = os.environ.get("ALPACA_API_KEY", "").strip()
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "").strip()
        paper = os.environ.get("ALPACA_PAPER", "true").lower() != "false"

        if not api_key or not secret_key:
            raise RuntimeError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env. "
                "Copy .env.example to .env and fill in paper-trading keys from alpaca.markets."
            )

        return cls(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
            symbols=list(raw["symbols"]),
            timeframe=raw["timeframe"],
            poll_interval_seconds=int(raw["poll_interval_seconds"]),
            active_strategy=raw["active_strategy"],
            strategies=raw["strategies"],
            risk=RiskConfig(**raw["risk"]),
            trail=TrailingConfig(**raw.get("trail", {})),
            vol_target=VolTargetConfig(**raw.get("vol_target", {})),
            filters=FiltersConfig(**raw.get("filters", {})),
            market_regime=MarketRegimeConfig(**raw.get("market_regime", {})),
            momentum=MomentumConfig(**raw.get("momentum", {})),
            protections=ProtectionsConfig(**raw.get("protections", {})),
            market_hours_only=bool(raw.get("market_hours_only", True)),
            log_level=raw.get("log_level", "INFO"),
            log_file=raw.get("log_file", "logs/bot.log"),
            trades_db=raw.get("trades_db", "data/trades.db"),
        )
