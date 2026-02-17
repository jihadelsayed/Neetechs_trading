# Trading Lab -- Requirements

## Overview

This project analyzes large-cap stocks and backtests a simple moving
average crossover strategy.

It downloads historical market data, generates trading signals, and
evaluates portfolio performance.

------------------------------------------------------------------------

## Python Version

-   Python 3.9 or higher recommended

Check version:

    python --version

------------------------------------------------------------------------

## Required Packages

Install dependencies using:

    pip install -r requirements.txt

### Required Libraries

-   pandas
-   numpy
-   yfinance
-   requests
-   beautifulsoup4
-   lxml
-   pytest

------------------------------------------------------------------------

## requirements.txt Content

    pandas>=2.0
    yfinance>=0.2.40
    numpy>=1.24
    pytest>=8.0
    requests>=2.31
    beautifulsoup4>=4.11.1
    lxml>=5.0

------------------------------------------------------------------------

## Project Structure

    Neetechs_trading/
    │
    ├── config.py
    ├── data_fetcher.py
    ├── strategy.py
    ├── backtest.py
    ├── main.py
    ├── requirements.txt
    ├── readme.md
    │
    ├── data/
    └── results/

------------------------------------------------------------------------

## Data Sources

-   Market data retrieved using `yfinance`
-   Data pulled from Yahoo Finance API

------------------------------------------------------------------------

## How to Run

1.  Install dependencies: pip install -r requirements.txt
    Editable install: `pip install -e .`

2.  Download data: python data_fetcher.py

3.  Run backtest: python main.py

    Optional entrypoints:

    -   python -m app.backtest --universe small
    -   python -m app.experiment --universe small --walk-forward
    -   python -m app.signal --universe nasdaq100 --top-n 10 --execution next_open --price-mode adj
    -   python -m app.live --mode paper --universe nasdaq100 --execution next_open
    -   python -m app.live --mode paper --universe nasdaq100 --flatten
    -   rg -n "walk" .
    -   python -m app.experiment --universe nasdaq100 --walk-forward --train-days 504 --test-days 126 --step-days 126
    -   metrics_summary.csv

    -   python -m app.signal --universe nasdaq100 --refresh-data --top-n 10 --execution next_open
    -   results/live_signal.csv

## Quickstart (Local)

```bash
pip install -e .
neetechs-health --universe small
neetechs-backtest --universe small --start 2024-01-01 --end 2025-01-01
```

## Docker

Build:

```bash
docker build -t neetechs-trading .
```

Run backtest:

```bash
docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/results:/app/results -v ${PWD}/logs:/app/logs neetechs-trading python -m app.backtest --universe small
```

Run experiment:

```bash
docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/results:/app/results -v ${PWD}/logs:/app/logs neetechs-trading python -m app.experiment --universe small --walk-forward
```

Run paper live loop:

```bash
docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/results:/app/results -v ${PWD}/logs:/app/logs neetechs-trading python -m app.live --mode paper --universe small --execution next_open
```

## CI

GitHub Actions runs ruff + pytest on push/PR (Python 3.10/3.11/3.12).

## Nightly Experiments

Workflow: `.github/workflows/nightly.yml`
Enable by setting repository variable `NIGHTLY_ENABLED=true`.

## Outputs

- `data/` cached market data
- `results/` backtests, experiments, robustness
- `logs/` live paper logs, reports, state
 - `results/live_signal.csv` latest signal snapshot

## Dashboard

Install optional deps:

```bash
pip install -e .[dashboard]
streamlit run app/dashboard.py
```

## Daily Paper Run

-   Command: `python -m app.live --mode paper --universe nasdaq100 --execution next_open`
-   Schedule: call once per day from Task Scheduler/cron
-   Outputs:
    -   logs/paper_trades.csv
    -   logs/paper_positions.csv
    -   logs/paper_equity.csv
    -   logs/reports/YYYY-MM-DD_report.md
    -   logs/state.json

## Live Broker (Optional, Disabled by Default)

To enable live broker placement (Alpaca), all of these are required:

-   `--mode live`
-   `--i-accept-real-trading YES`
-   env `LIVE_TRADING_ENABLED=true`

Environment variables:
-   `ALPACA_API_KEY`
-   `ALPACA_API_SECRET`
-   `ALPACA_BASE_URL` (optional, default paper endpoint)

Kill switch:
-   `KILL_SWITCH=true` blocks all orders and logs an alert.

------------------------------------------------------------------------

## Defaults

-   PRICE_MODE: adj (uses adjusted prices derived from Adj Close; Open is scaled by Adj Close / Close)
-   EXECUTION: next_open (signals on close[t], fills at open[t+1])
-   SLIPPAGE_MODE: bps (base 5 bps, per-turnover 0)
-   CASH_BUFFER: 1%
-   QQQ remains a regime filter only unless enabled

## Strategy Summary

The strategy implements a parsimonious momentum model:

- Blended momentum:
  - M12_1 = (Price[t-21] / Price[t-252] - 1)
  - M6_1 = (Price[t-21] / Price[t-126] - 1)
  - M_raw = 0.5 * M12_1 + 0.5 * M6_1
- Vol-adjusted score:
  - sigma_i = StdDev(daily_returns_i[t-126:t])
  - Score_i = M_raw / sigma_i
- Dispersion filter:
  - Dispersion_t = StdDev(Score across symbols)
  - Dispersion_med = RollingMedian(Dispersion[t-252:t])
  - Only rank when Dispersion_t > Dispersion_med; otherwise hold QQQ (or cash if unavailable)
- Selection: top TOP_N by Score_i
- Weights: inverse-vol weights, cap at 10% per name, then renormalize
- Turnover buffer: only exit if rank falls below N + 3
- Vol targeting: scale weights to target annual vol (default 0.15) using a 60-day window
- Regime filter: if QQQ < SMA200, hold QQQ

## Notes

-   This project is for educational purposes.
-   Results are not financial advice.
