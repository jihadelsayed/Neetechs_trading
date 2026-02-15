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

------------------------------------------------------------------------

## requirements.txt Content

    pandas>=2.0
    numpy>=1.24
    yfinance>=0.2.40

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

------------------------------------------------------------------------

## Notes

-   This project is for educational purposes.
-   No transaction costs or slippage included.
-   Results are not financial advice.
