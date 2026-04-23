"""Risk & position sizing.

Given account equity, entry price, ATR, and (optionally) recent returns,
compute:
- stop loss price (entry - stop_mult * ATR)
- take profit price (entry + tp_mult * ATR) — used only as a hard ceiling;
  the chandelier trail is the real exit mechanism
- position size:
    base_risk  = equity * risk_per_trade_pct
    vol_scalar = target_vol / realized_vol,  clamped to [0.25, 2.0]
    qty        = (base_risk * vol_scalar) / stop_distance
  then capped by max_position_pct of equity.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import RiskConfig

log = logging.getLogger("bot.risk")


@dataclass
class TradePlan:
    qty: int
    entry: float
    stop_loss: float
    take_profit: float
    risk_usd: float
    notional: float
    atr_at_entry: float
    vol_scalar: float


def realized_vol(close: pd.Series, lookback: int = 20,
                 annualize: int = 252 * 26) -> float:
    """Annualized realized vol from log returns. `annualize` defaults to
    ~trading_days * bars_per_day(15-min). Caller may override."""
    rets = np.log(close / close.shift(1)).dropna().tail(lookback)
    if len(rets) < 5:
        return 0.0
    return float(rets.std(ddof=1) * math.sqrt(annualize))


def vol_targeted_scalar(close: pd.Series, target_vol: float = 0.20,
                        lookback: int = 20,
                        annualize: int = 252 * 26) -> float:
    rv = realized_vol(close, lookback, annualize)
    if rv <= 1e-9:
        return 1.0
    s = target_vol / rv
    return max(0.25, min(2.0, s))


def momentum_rank_scalar(rank: int | None, top_k: int,
                         max_boost: float = 0.5) -> float:
    """Map a momentum rank (1 = strongest) to a size-boost multiplier.

    rank=1 with top_k=10, max_boost=0.5 -> 1.45
    rank=top_k                          -> 1.00
    rank outside top-k / None           -> 1.00
    """
    if rank is None or top_k <= 0 or rank > top_k or rank < 1:
        return 1.0
    return 1.0 + max_boost * (top_k - rank) / top_k


def plan_trade(
    price: float,
    atr_value: float,
    equity: float,
    risk_cfg: RiskConfig,
    close_series: pd.Series | None = None,
    target_vol: float = 0.20,
    momentum_scalar: float = 1.0,
) -> TradePlan | None:
    """Build a TradePlan or return None if the trade cannot be sized.

    `momentum_scalar` multiplies the risk budget for higher-ranked names
    (e.g. 1.45 for rank-1, 1.0 for rank-10). Clamped to [0.5, 2.0].
    """
    if price <= 0 or atr_value <= 0 or equity <= 0:
        return None

    stop_distance = risk_cfg.stop_loss_atr_mult * atr_value
    if stop_distance <= 0:
        return None

    stop_loss = price - stop_distance
    take_profit = price + risk_cfg.take_profit_atr_mult * atr_value

    # Volatility-targeted risk budget
    scalar = 1.0
    if close_series is not None and len(close_series) >= 20:
        scalar = vol_targeted_scalar(close_series, target_vol=target_vol)

    mom_scalar = max(0.5, min(2.0, momentum_scalar))
    risk_budget = equity * risk_cfg.risk_per_trade_pct * scalar * mom_scalar
    qty_by_risk = risk_budget / stop_distance

    max_notional = equity * risk_cfg.max_position_pct
    qty_by_notional = max_notional / price

    qty = int(math.floor(min(qty_by_risk, qty_by_notional)))
    if qty < 1:
        log.info(
            "Trade skipped — qty<1 (price=%.2f stop_dist=%.2f equity=%.2f scalar=%.2f)",
            price, stop_distance, equity, scalar,
        )
        return None

    notional = qty * price
    return TradePlan(
        qty=qty,
        entry=price,
        stop_loss=round(stop_loss, 2),
        take_profit=round(take_profit, 2),
        risk_usd=qty * stop_distance,
        notional=notional,
        atr_at_entry=atr_value,
        vol_scalar=scalar,
    )
