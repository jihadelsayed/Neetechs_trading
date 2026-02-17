from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

from tradinglab.config import DATA_DIR, START_DATE, END_DATE


DATA_DIR.mkdir(exist_ok=True)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().title() for c in df.columns]
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    return df[keep]


def _extract_symbol_frame(batch_df: pd.DataFrame, symbol: str) -> pd.DataFrame | None:
    if batch_df is None or batch_df.empty:
        return None

    if isinstance(batch_df.columns, pd.MultiIndex):
        # Expected shapes:
        # level 0 = field, level 1 = ticker (default)
        # or level 0 = ticker, level 1 = field (group_by='ticker')
        lvl0 = batch_df.columns.get_level_values(0)
        lvl1 = batch_df.columns.get_level_values(1)
        if symbol in lvl1:
            df = batch_df.xs(symbol, level=1, axis=1)
        elif symbol in lvl0:
            df = batch_df.xs(symbol, level=0, axis=1)
        else:
            return None
    else:
        # single symbol download
        df = batch_df

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return _normalize_columns(df)


def fetch_symbol(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    start = start_date or START_DATE
    end = end_date if end_date is not None else END_DATE
    kwargs = {
        "start": start,
        "interval": "1d",
        "auto_adjust": False,
        "progress": False,
    }
    if end is not None:
        kwargs["end"] = end
    df = yf.download(symbol, **kwargs)

    if df is None or df.empty:
        raise ValueError(f"No data returned for symbol: {symbol}")

    df = _extract_symbol_frame(df, symbol)
    if df is None or df.empty:
        raise ValueError(f"No usable data returned for symbol: {symbol}")
    return df


def fetch_symbols(
    symbols: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, pd.DataFrame]:
    if not symbols:
        return {}

    start = start_date or START_DATE
    end = end_date if end_date is not None else END_DATE
    joined = " ".join(symbols)
    kwargs = {
        "start": start,
        "interval": "1d",
        "auto_adjust": False,
        "progress": False,
        "group_by": "ticker",
    }
    if end is not None:
        kwargs["end"] = end
    df = yf.download(joined, **kwargs)

    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        sym_df = _extract_symbol_frame(df, sym)
        if sym_df is not None and not sym_df.empty:
            out[sym] = sym_df
    return out


def save_symbol_csv(symbol: str, df: pd.DataFrame) -> Path:
    path = DATA_DIR / f"{symbol}.csv"
    df.to_csv(path, index_label="Date")
    return path


def load_symbol_csv(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame | None:
    path = DATA_DIR / f"{symbol}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    if start_date:
        start_dt = pd.to_datetime(start_date)
        if df.index.min() > start_dt:
            return None
    if end_date:
        end_dt = pd.to_datetime(end_date)
        if df.index.max() < (end_dt - pd.Timedelta(days=1)):
            return None
    if start_date or end_date:
        df = df.loc[start_date:end_date]
    return df


def load_or_fetch_symbol(
    symbol: str,
    refresh: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    cached = None if refresh else load_symbol_csv(symbol, start_date=start_date, end_date=end_date)
    if cached is not None:
        return cached
    df = fetch_symbol(symbol, start_date=start_date, end_date=end_date)
    save_symbol_csv(symbol, df)
    return df


def load_or_fetch_symbols(
    symbols: Iterable[str],
    refresh: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    batch_size: int = 50,
) -> dict[str, pd.DataFrame]:
    symbols = list(dict.fromkeys([s.strip().upper() for s in symbols if s]))
    if not symbols:
        return {}

    out: dict[str, pd.DataFrame] = {}

    # Load cached first
    if not refresh:
        for sym in symbols:
            cached = load_symbol_csv(sym, start_date=start_date, end_date=end_date)
            if cached is not None:
                out[sym] = cached

    # Fetch missing in batches
    missing = [s for s in symbols if s not in out]
    if missing:
        for i in range(0, len(missing), batch_size):
            batch = missing[i : i + batch_size]
            fetched = fetch_symbols(batch, start_date=start_date, end_date=end_date)
            for sym, df in fetched.items():
                out[sym] = df
                save_symbol_csv(sym, df)

    # Final fallback: per symbol fetch for any still missing
    still_missing = [s for s in symbols if s not in out]
    for sym in still_missing:
        try:
            df = fetch_symbol(sym, start_date=start_date, end_date=end_date)
            out[sym] = df
            save_symbol_csv(sym, df)
        except Exception:
            continue

    return out
