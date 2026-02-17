from __future__ import annotations

from pathlib import Path

# Date range for historical data
START_DATE = "2023-01-01"
END_DATE = None

# Initial capital for backtesting
INITIAL_CAPITAL = 1000.0

# Moving average settings
SHORT_WINDOW = 20
LONG_WINDOW = 200

# Volatility gate
MIN_VOL20 = 0.8

# Realism
FEE_RATE = 0.001        # 0.10% per trade
SLIPPAGE_RATE = 0.0005  # 0.05% worse fill on buys/sells
SLIPPAGE_MODE = "bps"   # "bps" or "constant"
SLIPPAGE_BPS_BASE = 5.0
SLIPPAGE_BPS_PER_TURNOVER = 0.0
REBALANCE = "ME"        # month-end rebalance
MOM_LOOKBACK = 126
TOP_N = 15

# Regime filter
REGIME_SYMBOL = "QQQ"
ALLOW_REGIME_TRADE = False

# Execution + pricing
EXECUTION = "next_open"  # "next_open" or "same_close"
PRICE_MODE = "adj"       # "adj" or "raw"

# Risk controls (portfolio-level)
MAX_POSITION_WEIGHT = None
MAX_SECTOR_WEIGHT = None
MAX_TURNOVER_PER_REBALANCE = None
MAX_GROSS_EXPOSURE = 1.0
CASH_BUFFER = 0.01
MAX_ORDER_NOTIONAL = 10000.0
MAX_ORDERS_PER_RUN = 50

# Risk controls (position-level)
TRAILING_STOP_PCT = None
TIME_STOP_DAYS = None
TARGET_VOL = 0.15
VOL_LOOKBACK = 20

# Paths
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
