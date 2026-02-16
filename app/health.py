from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Health checks")
    parser.add_argument("--universe", choices=["small", "nasdaq100"], default="nasdaq100")
    return parser


def run(argv: list[str] | None = None) -> int:
    _ensure_src_on_path()

    from tradinglab.data.tickers import nasdaq100_tickers
    from tradinglab.data.fetcher import load_or_fetch_symbols
    from tradinglab.config import REGIME_SYMBOL

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.universe == "small":
        symbols = ["AAPL", "MSFT", "NVDA", "AMZN", REGIME_SYMBOL]
    else:
        symbols = nasdaq100_tickers()
        if REGIME_SYMBOL not in symbols:
            symbols.append(REGIME_SYMBOL)

    price_dict = load_or_fetch_symbols(symbols, refresh=False)
    if not price_dict:
        print("Health check failed: no data")
        return 1

    latest_dates = []
    for sym, df in price_dict.items():
        if df.empty:
            print(f"Health check failed: empty data for {sym}")
            return 1
        if "Open" not in df.columns or "Close" not in df.columns:
            print(f"Health check failed: missing Open/Close for {sym}")
            return 1
        latest_dates.append(df.index.max())

    max_date = max(latest_dates)
    if any((max_date - d).days > 3 for d in latest_dates):
        print("Health check failed: stale data")
        return 1

    latest_prices = []
    for sym, df in price_dict.items():
        row = df.loc[max_date]
        if row.isna().any():
            print(f"Health check failed: NaNs in latest prices for {sym}")
            return 1
        latest_prices.append(row)

    print("Health check OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
