"""Streamlit dashboard.

Live read-only view of the bot: account, positions, trades, equity curve.
Run:  streamlit run dashboard.py
"""
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from bot.broker import Broker
from bot.config import Config
from bot.storage import Storage


st.set_page_config(page_title="Trading Bot", layout="wide", page_icon="📈")
st.title("📈 Trading Bot — Paper")


@st.cache_resource
def _get_clients() -> tuple[Config, Broker, Storage]:
    cfg = Config.load()
    return cfg, Broker(cfg.api_key, cfg.secret_key, paper=cfg.paper), Storage(cfg.trades_db)


cfg, broker, storage = _get_clients()

refresh = st.sidebar.number_input("Auto-refresh (s)", 0, 300, 15, step=5)
if refresh > 0:
    st.sidebar.caption(f"Refreshing every {refresh}s — press R or reload to force refresh")
    # Optional: install `streamlit-autorefresh` for automatic refresh
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=int(refresh) * 1000, key="autorefresh")
    except ImportError:
        st.sidebar.info("Install `streamlit-autorefresh` for auto-refresh:\n`pip install streamlit-autorefresh`")

st.sidebar.subheader("Config")
st.sidebar.write(f"**Strategy:** `{cfg.active_strategy}`")
st.sidebar.write(f"**Timeframe:** `{cfg.timeframe}`")
st.sidebar.write(f"**Symbols:** {', '.join(cfg.symbols)}")
st.sidebar.write(f"**Paper:** `{cfg.paper}`")

# -------- Account row --------
try:
    acct = broker.account()
    market_open = broker.is_market_open()
except Exception as e:
    st.error(f"Alpaca connection failed: {e}")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Equity", f"${acct.equity:,.2f}")
c2.metric("Cash", f"${acct.cash:,.2f}")
c3.metric("Buying Power", f"${acct.buying_power:,.2f}")
c4.metric("Market", "OPEN" if market_open else "CLOSED")

st.divider()

# -------- Equity curve --------
eq = storage.equity_df(limit=5000)
st.subheader("Equity Curve")
if eq.empty:
    st.info("No equity snapshots yet — run the bot (`python main.py`) to start collecting.")
else:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq["ts"], y=eq["equity"], mode="lines", name="Equity"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10),
                      yaxis_title="USD", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

# -------- Open positions --------
st.subheader("Open Positions")
try:
    positions = broker.positions()
except Exception as e:
    st.error(f"Could not fetch positions: {e}")
    positions = []

if not positions:
    st.caption("No open positions.")
else:
    import pandas as pd
    pdf = pd.DataFrame([p.__dict__ for p in positions])
    pdf = pdf.rename(columns={
        "symbol": "Symbol", "qty": "Qty", "avg_entry_price": "Avg Entry",
        "market_value": "Market Value", "unrealized_pl": "Unrealized P&L",
        "side": "Side",
    })
    st.dataframe(pdf, use_container_width=True, hide_index=True)

# -------- Recent trades --------
st.subheader("Recent Trades")
tdf = storage.trades_df(limit=200)
if tdf.empty:
    st.caption("No trades recorded yet.")
else:
    tdf = tdf.drop(columns=["id"])
    st.dataframe(tdf, use_container_width=True, hide_index=True)

st.caption("Read-only view. All trading happens on the Alpaca paper account — no real money at risk.")
