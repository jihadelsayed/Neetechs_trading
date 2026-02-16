from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tradinglab.data.tickers import nasdaq100_tickers  # noqa: E402


if __name__ == "__main__":
    syms = nasdaq100_tickers()
    print("count:", len(syms))
    print(syms[:25])
