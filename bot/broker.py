"""Alpaca broker wrapper.

Thin layer over alpaca-py that exposes exactly what the engine needs:
- account equity / buying power
- open positions
- submit market orders (with optional bracket SL/TP)
- close position
- is market open
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
from alpaca.trading.requests import (
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

log = logging.getLogger("bot.broker")


@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float
    unrealized_pl: float
    side: str  # "long" or "short"


@dataclass
class AccountSnapshot:
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float


class Broker:
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.paper = paper
        self.client = TradingClient(api_key, secret_key, paper=paper)
        log.info("Broker initialized (paper=%s)", paper)

    # ---------- account ----------
    def account(self) -> AccountSnapshot:
        a = self.client.get_account()
        return AccountSnapshot(
            equity=float(a.equity),
            cash=float(a.cash),
            buying_power=float(a.buying_power),
            portfolio_value=float(a.portfolio_value),
        )

    def is_market_open(self) -> bool:
        clock = self.client.get_clock()
        return bool(clock.is_open)

    def next_market_open(self) -> datetime:
        return self.client.get_clock().next_open

    # ---------- positions ----------
    def positions(self) -> list[Position]:
        raw = self.client.get_all_positions()
        return [
            Position(
                symbol=p.symbol,
                qty=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                market_value=float(p.market_value),
                unrealized_pl=float(p.unrealized_pl),
                side=p.side.value if hasattr(p.side, "value") else str(p.side),
            )
            for p in raw
        ]

    def position_for(self, symbol: str) -> Position | None:
        for p in self.positions():
            if p.symbol == symbol:
                return p
        return None

    # ---------- orders ----------
    def submit_bracket_buy(
        self,
        symbol: str,
        qty: float,
        stop_loss: float,
        take_profit: float,
    ) -> Any:
        """Submit a market BUY with attached stop-loss and take-profit (bracket order)."""
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            stop_loss=StopLossRequest(stop_price=round(stop_loss, 2)),
            take_profit=TakeProfitRequest(limit_price=round(take_profit, 2)),
        )
        order = self.client.submit_order(req)
        log.info(
            "BUY bracket %s qty=%s SL=%.2f TP=%.2f id=%s",
            symbol, qty, stop_loss, take_profit, order.id,
        )
        return order

    def close_position(self, symbol: str) -> Any:
        try:
            order = self.client.close_position(symbol)
            log.info("CLOSE %s id=%s", symbol, getattr(order, "id", "?"))
            return order
        except Exception as e:
            log.warning("close_position(%s) failed: %s", symbol, e)
            return None

    def cancel_all_orders(self) -> None:
        try:
            self.client.cancel_orders()
        except Exception as e:
            log.warning("cancel_orders failed: %s", e)
