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

2.  Download data: python data_fetcher.py

3.  Run backtest: python main.py

    Optional entrypoints:

    -   python -m app.backtest --universe small
    -   python -m app.experiment --universe small --walk-forward
    -   python -m app.live --mode paper --universe nasdaq100 --execution next_open
    -   python -m app.live --mode paper --universe nasdaq100 --flatten

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

## Notes

-   This project is for educational purposes.
-   Results are not financial advice.
