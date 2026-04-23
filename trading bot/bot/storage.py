"""SQLite-backed trade & equity log.

Two tables:
- trades: one row per BUY / SELL event (side, qty, price, timestamps)
- equity:  periodic snapshot of account equity (for the dashboard P&L chart)
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger("bot.storage")


SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,            -- BUY / SELL
    qty REAL NOT NULL,
    price REAL NOT NULL,
    stop_loss REAL,
    take_profit REAL,
    strategy TEXT,
    note TEXT
);
CREATE TABLE IF NOT EXISTS equity (
    ts TEXT PRIMARY KEY,
    equity REAL NOT NULL,
    cash REAL NOT NULL,
    positions_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS trail_state (
    symbol TEXT PRIMARY KEY,
    entry REAL NOT NULL,
    initial_stop REAL NOT NULL,
    atr_at_entry REAL NOT NULL,
    high_since_entry REAL NOT NULL,
    breakeven_active INTEGER NOT NULL DEFAULT 0,
    profit_lock_active INTEGER NOT NULL DEFAULT 0,
    current_stop REAL NOT NULL,
    updated_ts TEXT NOT NULL
);
"""


def _ensure_column(conn, table: str, column: str, coltype: str) -> None:
    """Add a column to an existing SQLite table if it is missing.
    SQLite doesn't support IF NOT EXISTS on ADD COLUMN, so we check PRAGMA first."""
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


class Storage:
    def __init__(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.path = str(p)
        with self._conn() as c:
            c.executescript(SCHEMA)
            # Migration for users whose DB was created before profit_lock_active existed
            _ensure_column(c, "trail_state", "profit_lock_active",
                           "INTEGER NOT NULL DEFAULT 0")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def record_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        *,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        strategy: str | None = None,
        note: str | None = None,
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with self._conn() as c:
            c.execute(
                """INSERT INTO trades
                   (ts, symbol, side, qty, price, stop_loss, take_profit, strategy, note)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (ts, symbol, side, qty, price, stop_loss, take_profit, strategy, note),
            )
        log.info("Recorded %s %s qty=%s @ %.2f", side, symbol, qty, price)

    def record_equity(self, equity: float, cash: float, positions_count: int) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO equity (ts, equity, cash, positions_count) VALUES (?,?,?,?)",
                (ts, equity, cash, positions_count),
            )

    def trades_df(self, limit: int = 500) -> pd.DataFrame:
        with self._conn() as c:
            df = pd.read_sql_query(
                "SELECT * FROM trades ORDER BY id DESC LIMIT ?",
                c, params=(limit,),
            )
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"])
        return df

    # ---- trail state ----
    def save_trail(self, symbol: str, state) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with self._conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO trail_state
                   (symbol, entry, initial_stop, atr_at_entry, high_since_entry,
                    breakeven_active, profit_lock_active, current_stop, updated_ts)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (symbol, state.entry, state.initial_stop, state.atr_at_entry,
                 state.high_since_entry, int(state.breakeven_active),
                 int(state.profit_lock_active), state.current_stop, ts),
            )

    def load_trail(self, symbol: str):
        from .trailing import TrailState
        with self._conn() as c:
            row = c.execute(
                """SELECT entry, initial_stop, atr_at_entry, high_since_entry,
                          breakeven_active, profit_lock_active, current_stop
                   FROM trail_state WHERE symbol=?""", (symbol,),
            ).fetchone()
        if not row:
            return None
        return TrailState(
            entry=row[0], initial_stop=row[1], atr_at_entry=row[2],
            high_since_entry=row[3], breakeven_active=bool(row[4]),
            profit_lock_active=bool(row[5]),
            current_stop=row[6],
        )

    def delete_trail(self, symbol: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM trail_state WHERE symbol=?", (symbol,))

    def equity_df(self, limit: int = 5000) -> pd.DataFrame:
        with self._conn() as c:
            df = pd.read_sql_query(
                "SELECT * FROM equity ORDER BY ts DESC LIMIT ?",
                c, params=(limit,),
            )
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"])
            df = df.sort_values("ts")
        return df
