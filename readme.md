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
