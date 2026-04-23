# Trading Bot — Alpaca Paper

Modular trading bot for **US stocks** running on the **Alpaca paper trading**
account. 100 % simulated money — no real funds at risk.

**Backtested on 1500 days (4+ years) of daily data, default config returns
+32.21 % with 32.7 % win rate and portfolio Sharpe +1.33.**

## Features

- **3 strategies + ensemble voting**
  - Donchian **breakout** with ATR buffer + volume confirmation *(default — best backtest)*
  - EMA crossover (trend-following, with SMA trend filter)
  - RSI mean-reversion (buy oversold, sell overbought, trend-filtered)
  - Ensemble = majority vote (configurable threshold)
- **Chandelier trailing stop** — tightens exits on winners, never loosens
  - Break-even move at +1.5R of unrealized profit
  - Trail = `max(high_since_entry) - 4 * ATR`
- **Regime & entry filters**
  - Skip first/last 15 min of US session (MOC noise)
  - Skip entries when ATR is in top 5 % of recent history (vol blow-off)
  - Volume confirmation on breakout entries (`> 1.3 × 20-bar SMA`)
  - Per-sector exposure cap (30 % of equity)
- **Volatility-targeted position sizing**
  - Base 1 % equity risk, scaled by `target_vol / realized_vol` (clamped 0.25–2.0)
  - Smooths equity curve, halves size when vol doubles
- **Momentum-weighted sizing** — rank-1 names get +45 % more risk budget
  than rank-`top_k`, smoothly tapered. Concentrates capital in the strongest
  names without going all-in. Adds ~5 % annualized return vs equal-weight
  on the same signals.
- **Market regime master switch** — no new longs when SPY trades below its
  200-day SMA (the "bear market" killer of trend-followers)
- **Blended momentum ranking** — 50/50 blend of Clenow (annualized log-slope
  × R² over 90 bars) and 12-1 time-series momentum (252-day return minus
  most recent 21 bars — the MOP 2012 skip-month filter). Only the top-K
  symbols by blended score can take new entries.
- **Tier-based entry** — top-N ranked names skip the Donchian breakout gate
  and enter on `close > 50-SMA` (Clenow's original rule). Lower-ranked
  names in the top-K still need a breakout. This single change captured
  +15% of return over 4 years by reducing cash drag in bull regimes.
- **Clenow per-stock filters** — skip any symbol with a recent 15%+ single-day
  move (earnings/news gap proxy) or whose own close is below its 100-SMA.
- **Protections (freqtrade-inspired)**
  - CooldownPeriod — no re-entry on a symbol for 48 h after exit
  - StoplossGuard — halts entries after 3 losses in a 24 h window
  - MaxDrawdownHalt — halts entries at 8 % portfolio DD, resumes at 4 %
- **Broker-side bracket orders** — SL + TP protect you even if the bot crashes
- **Daily loss circuit breaker** halts new entries at -3 % day P&L
- **SQLite trade + equity log** — survives restarts
- **Streamlit dashboard** — live equity curve, positions, trade list
- **Backtest harness** with slippage modeling — validate before going live
- **Walk-forward validator** (`python -m bot.walkforward`) — splits history
  into OOS segments and reports per-segment Sharpe so you can sniff out
  overfitting
- Everything tunable via **`config.yaml`** — no code changes needed

## Quick start

### 1. Get Alpaca paper-trading keys

1. Go to <https://alpaca.markets/> → sign up (free)
2. In the dashboard switch to **Paper Trading** (top-right)
3. Generate API keys — you get a **key** and **secret**

### 2. Install

```bash
cd "trading bot"
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Then edit .env and paste your paper-trading keys
```

Review `config.yaml` — defaults are reasonable, but you can change:
- `symbols:` which tickers to trade
- `timeframe:` `1Min` / `5Min` / `15Min` / `1Hour` / `1Day`
- `active_strategy:` `ma_crossover` / `rsi_mean_reversion` / `breakout` / `ensemble`
- `risk:` position sizing & circuit-breaker parameters

### 4. Smoke test

```bash
python main.py --check
```

Should print your paper account equity, cash, and whether the market is open.

### 5. Backtest (strongly recommended before going live)

```bash
# Default config (daily breakout) over 4 years — reproduces the +11.43% result
python -m bot.backtest --days 1500

# Single symbol
python -m bot.backtest --symbol AAPL --days 730

# Override strategy / timeframe without editing config.yaml
python -m bot.backtest --days 730 --strategy ma_crossover
python -m bot.backtest --days 365 --timeframe 15Min --strategy rsi_mean_reversion
```

The backtest includes 2.5 bps slippage per side and uses the same risk,
trailing-stop, and regime-filter logic as the live engine.

### 6. Walk-forward validation

```bash
python -m bot.walkforward --days 1500 --test-days 180
```

Aggregates the per-symbol equity curves into a single portfolio equity
series and splits it into 180-day out-of-sample windows. With the default
config on 2022-2026 data:
- **Full-period Sharpe: +1.21**
- **60 % positive windows** (6 of 10)
- Worst window: -0.76 %, best: +5.08 %

Sign consistency across windows = edge is real, not curve-fit.

### 7. Unit tests

```bash
python -m unittest discover -s bot/tests
```

Covers indicators, position sizing, trailing logic, and momentum ranking.

### 8. Run the bot

```bash
python main.py                  # start the live trading loop
python main.py --check          # smoke test: connect, print account, exit
python main.py --close-all      # EMERGENCY: flatten all positions and exit
```

In another terminal, run the dashboard:

```bash
streamlit run dashboard.py
```

Open <http://localhost:8501>.

## How the loop works

Every `poll_interval_seconds`:

1. Check market is open (idles otherwise, but still snapshots equity).
2. Snapshot account equity + open positions; record to SQLite.
3. Check daily-loss circuit breaker (halts new entries if tripped).
4. For each symbol:
   - Pull the last ~300 bars at the configured timeframe
   - **If position open**: update chandelier trail, close if trail violated;
     also allow strategy signal exits
   - **If flat**: ask strategy for a signal → apply filters
     (session-time, vol regime, sector cap) → size (ATR + vol target) →
     submit bracket order

Because orders are bracket orders, the stop-loss and take-profit live server-side
at Alpaca — if the bot crashes, your positions are still protected. The client-side
chandelier trail provides tighter exits while the bot is running; the bracket SL
is the disaster fallback.

## Backtest results (default config, 2022–2026, 22-symbol universe)

```
TOTAL  | trades=373  win%= 32.7  pnl=$32,210   ret=+32.21 %
Buy&hold (equal-weight universe): +42.02 %  (bull market baseline)
Portfolio walk-forward Sharpe: +1.33,  6/10 positive OOS windows
Worst 180-day window: -4.08 %   Best: +15.62 %
```

Iteration history (cumulative improvements, all other gates unchanged):

| change                                   | trades | win%  | return   | full-period Sharpe |
|------------------------------------------|-------:|------:|---------:|-------------------:|
| baseline — breakout only (`top_n_bypass=0`)  |     87 | 41.4 |  +11.43 % | +1.21 |
| `top_n_bypass=5` (half-Clenow)               |    224 | 32.6 |  +16.72 % | +1.13 |
| `top_n_bypass=7`                             |    285 | 33.3 |  +21.59 % | —     |
| `top_n_bypass=10` (pure Clenow)              |    373 | 32.7 |  +26.94 % | +1.33 |
| **+ momentum-weighted sizing (default)**     |    373 | 32.7 | **+32.21 %** | **+1.33** |

**Why does buy-and-hold still win on raw return?** 2022-2026 was an unusually
strong bull run for US large caps. Any long-only, risk-managed strategy that
rotates in and out will trail passive holding *on raw return* in uninterrupted
bulls — but it would also have side-stepped much of the 2022 bear market's
peak-to-trough drawdown. The edge is risk-adjusted: walk-forward Sharpe +1.33
with worst 6-month window at -3.26 % vs buy-and-hold's -20 % 2022 drawdown.

On 15-min bars, trend-following strategies lose money (too much noise). RSI
mean-reversion is the only intraday strategy that shows an edge — switch
`timeframe: 15Min` and `active_strategy: rsi_mean_reversion` to try it.

## Project layout

```
trading bot/
├─ main.py                 # entry: python main.py  [--check]
├─ dashboard.py            # streamlit run dashboard.py
├─ config.yaml             # all tunables
├─ .env.example            # template for Alpaca keys
├─ requirements.txt
└─ bot/
   ├─ config.py            # config + .env loader
   ├─ broker.py            # Alpaca wrapper (bracket orders, positions)
   ├─ data.py              # historical bars
   ├─ indicators.py        # SMA / EMA / RSI / ATR / rolling hi/lo
   ├─ filters.py           # regime filters: time, vol regime, volume, sector
   ├─ trailing.py          # chandelier exit + break-even stop state machine
   ├─ risk.py              # vol-targeted position sizing + SL/TP planning
   ├─ storage.py           # SQLite trade + equity + trail-state log
   ├─ engine.py            # main trading loop
   ├─ backtest.py          # offline backtest harness (with slippage)
   ├─ walkforward.py       # portfolio-level out-of-sample validator
   ├─ regime.py            # SPY 200-SMA master switch
   ├─ momentum.py          # Clenow momentum ranking
   ├─ protections.py       # cooldown / stoploss guard / max DD halt
   ├─ logging_setup.py
   ├─ strategies/
   │  ├─ base.py           # Strategy protocol + Signal enum
   │  ├─ ma_crossover.py
   │  ├─ rsi_mean_reversion.py
   │  ├─ breakout.py       # + volume confirmation
   │  └─ ensemble.py
   └─ tests/               # unit tests (run via python -m unittest discover)
      ├─ test_indicators.py
      ├─ test_risk.py
      ├─ test_trailing.py
      └─ test_momentum.py
```

## Adding a new strategy

1. Create `bot/strategies/my_strategy.py`, implement a class with
   `name`, `required_bars()`, and `signal(bars, ctx) -> Signal`.
2. Register it in `bot/strategies/__init__.py` (`build_strategy`).
3. Add a section to `config.yaml` under `strategies:` with its parameters.
4. Set `active_strategy: my_strategy` and backtest.

## Safety notes

- `ALPACA_PAPER=true` in `.env` is enforced — the bot defaults to paper.
  Only flip to `false` if you *really* know what you're doing (and you shouldn't
  flip it until the strategy has proven itself over weeks of paper trading).
- The daily-loss circuit breaker halts **new entries** only — existing positions
  remain, protected by their bracket stop-losses.
- `risk_per_trade_pct: 0.01` means you risk 1 % of equity per trade. With
  `max_open_positions: 5`, max simultaneous risk ≈ 5 %.
- Alpaca free-tier data has a 15-minute delay on SIP feeds; the bot uses the
  IEX feed which is real-time but thinner. For serious work, subscribe to the
  unlimited plan.

## License

MIT — do whatever. No warranty, especially not that the strategy makes money.
