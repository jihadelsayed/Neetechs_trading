from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tradinglab.engine.portfolio import (  # noqa: E402
    build_panel,
    run_portfolio,
    buy_hold_benchmark,
)

__all__ = ["build_panel", "run_portfolio", "buy_hold_benchmark"]
