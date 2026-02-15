# data_fetcher.py
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import yfinance as yf

from config import SYMBOLS, START_DATE, END_DATE


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def fetch_symbol(symbol: str) -> pd.DataFrame:
    """
    Download daily OHLCV data for a symbol and return as a DataFrame.
    """
    df = yf.download(
        symbol,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for symbol: {symbol}")

    # Normalize columns and index
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.columns = [c.strip().title() for c in df.columns]  # Open, High, Low, Close, Adj Close, Volume

    # Keep only standard columns if present
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep]

    return df


def save_symbol_csv(symbol: str, df: pd.DataFrame) -> Path:
    """
    Save a symbol's DataFrame to data/<SYMBOL>.csv and return path.
    """
    path = DATA_DIR / f"{symbol}.csv"
    df.to_csv(path, index=True)
    return path


def load_or_fetch_symbol(symbol: str, refresh: bool = False) -> pd.DataFrame:
    """
    Load from CSV if exists (unless refresh=True), otherwise download and save.
    """
    path = DATA_DIR / f"{symbol}.csv"

    if path.exists() and not refresh:
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.set_index("Date")
        return df

    df = fetch_symbol(symbol)
    save_symbol_csv(symbol, df)
    return df


def get_market_data(refresh: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Get data for all symbols in config.SYMBOLS.
    Returns dict: {symbol: dataframe}
    """
    data: Dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        data[sym] = load_or_fetch_symbol(sym, refresh=refresh)
    return data


if __name__ == "__main__":
    # Quick test run: downloads and saves CSVs
    market = get_market_data(refresh=False)
    for sym, df in market.items():
        print(sym, df.shape, "last date:", df.index.max().date())
