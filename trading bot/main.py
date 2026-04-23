"""Trading bot entry point.

Usage:
    python main.py              # run the live (paper) trading engine
    python main.py --check      # smoke test: connect, print account, exit

Start the dashboard separately with:
    streamlit run dashboard.py
"""
from __future__ import annotations

import argparse
import logging
import sys

from bot.config import Config
from bot.logging_setup import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Connect to Alpaca, print account info, and exit.",
    )
    parser.add_argument(
        "--close-all",
        action="store_true",
        help="EMERGENCY: cancel all open orders and flatten all positions, then exit.",
    )
    args = parser.parse_args()

    try:
        cfg = Config.load()
    except Exception as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2

    setup_logging(cfg.log_level, cfg.log_file)
    log = logging.getLogger("bot.main")

    if args.check:
        from bot.broker import Broker
        b = Broker(cfg.api_key, cfg.secret_key, paper=cfg.paper)
        acct = b.account()
        log.info(
            "Connected OK. Paper=%s  equity=$%.2f  cash=$%.2f  buying_power=$%.2f",
            cfg.paper, acct.equity, acct.cash, acct.buying_power,
        )
        log.info("Market open: %s", b.is_market_open())
        log.info("Open positions: %d", len(b.positions()))
        return 0

    if args.close_all:
        from bot.broker import Broker
        from bot.storage import Storage
        b = Broker(cfg.api_key, cfg.secret_key, paper=cfg.paper)
        store = Storage(cfg.trades_db)
        positions = b.positions()
        log.warning("EMERGENCY CLOSE-ALL: flattening %d position(s)", len(positions))
        b.cancel_all_orders()
        for p in positions:
            try:
                b.close_position(p.symbol)
                store.record_trade(
                    symbol=p.symbol, side="SELL", qty=p.qty,
                    price=p.market_value / p.qty if p.qty else 0.0,
                    strategy="emergency", note="manual close-all",
                )
                store.delete_trail(p.symbol)
            except Exception as e:
                log.warning("close_position(%s) failed: %s", p.symbol, e)
        log.warning("Close-all done. Verify on Alpaca dashboard.")
        return 0

    from bot.engine import Engine
    Engine(cfg).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
