from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tradinglab.config import (  # noqa: F401
    START_DATE,
    END_DATE,
    INITIAL_CAPITAL,
    SHORT_WINDOW,
    LONG_WINDOW,
    MIN_VOL20,
    FEE_RATE,
    SLIPPAGE_RATE,
    REBALANCE,
    MOM_LOOKBACK,
    TOP_N,
    REGIME_SYMBOL,
    ALLOW_REGIME_TRADE,
    DATA_DIR,
    RESULTS_DIR,
)

__all__ = [
    "START_DATE",
    "END_DATE",
    "INITIAL_CAPITAL",
    "SHORT_WINDOW",
    "LONG_WINDOW",
    "MIN_VOL20",
    "FEE_RATE",
    "SLIPPAGE_RATE",
    "REBALANCE",
    "MOM_LOOKBACK",
    "TOP_N",
    "REGIME_SYMBOL",
    "ALLOW_REGIME_TRADE",
    "DATA_DIR",
    "RESULTS_DIR",
]
