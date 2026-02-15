# config.py

# Big liquid companies (safe for beginners)
SYMBOLS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL"
]

# Date range for historical data
START_DATE = "2023-01-01"
END_DATE = "2026-01-01"

# Initial capital for backtesting
INITIAL_CAPITAL = 1000

# Risk per trade (percentage of capital)
RISK_PER_TRADE = 0.02  # 2%

# Moving average settings (we'll use later)
SHORT_WINDOW = 20
LONG_WINDOW = 50
