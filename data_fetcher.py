from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tradinglab.data.fetcher import load_or_fetch_symbols  # noqa: E402
from tradinglab.data.tickers import nasdaq100_tickers  # noqa: E402
from tradinglab.config import REGIME_SYMBOL  # noqa: E402


if __name__ == "__main__":
    symbols = nasdaq100_tickers()
    if REGIME_SYMBOL not in symbols:
        symbols.append(REGIME_SYMBOL)

    market = load_or_fetch_symbols(symbols, refresh=False)
    for sym, df in market.items():
        print(sym, df.shape, "last date:", df.index.max().date())
