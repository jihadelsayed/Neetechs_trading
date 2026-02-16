from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tradinglab.config import INITIAL_CAPITAL, FEE_RATE, SLIPPAGE_RATE  # noqa: E402
from tradinglab.strategies.ma_crossover import generate_signals  # noqa: E402


def backtest(df: pd.DataFrame) -> pd.DataFrame:
    df = generate_signals(df).copy()

    capital = float(INITIAL_CAPITAL)
    shares = 0.0

    first_price = float(df["Close"].iloc[0])
    bh_shares = float(INITIAL_CAPITAL) / first_price

    portfolio_values = []
    cash_values = []
    share_values = []
    buyhold_values = []

    for _, row in df.iterrows():
        price = float(row["Close"])
        target_pos = int(row["Position"])

        if target_pos == 1 and shares == 0:
            buy_price = price * (1.0 + SLIPPAGE_RATE)
            shares = (capital * (1.0 - FEE_RATE)) / buy_price
            capital = 0.0
        elif target_pos == 0 and shares > 0:
            sell_price = price * (1.0 - SLIPPAGE_RATE)
            capital = (shares * sell_price) * (1.0 - FEE_RATE)
            shares = 0.0

        portfolio_value = capital + (shares * price)
        portfolio_values.append(portfolio_value)
        cash_values.append(capital)
        share_values.append(shares * price)
        buyhold_values.append(bh_shares * price)

    df["Cash"] = cash_values
    df["Stock_Value"] = share_values
    df["Portfolio_Value"] = portfolio_values
    df["BuyHold_Value"] = buyhold_values

    return df


def performance_summary(df: pd.DataFrame):
    start_value = df["Portfolio_Value"].iloc[0]
    end_value = df["Portfolio_Value"].iloc[-1]
    total_return = (end_value - start_value) / start_value * 100

    print(f"Starting Capital: ${start_value:,.2f}")
    print(f"Ending Value:     ${end_value:,.2f}")
    print(f"Total Return:     {total_return:.2f}%")


if __name__ == "__main__":
    df = pd.read_csv("data/AAPL.csv", parse_dates=["Date"]).set_index("Date")
    result = backtest(df)
    performance_summary(result)
