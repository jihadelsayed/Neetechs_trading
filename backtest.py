# backtest.py
from __future__ import annotations

import pandas as pd

from config import INITIAL_CAPITAL
from strategy import generate_signals


def backtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backtest moving average strategy on a single symbol DataFrame.
    Returns DataFrame with portfolio tracking columns.
    """

    df = generate_signals(df).copy()

    capital = INITIAL_CAPITAL
    shares = 0

    portfolio_values = []
    cash_values = []
    share_values = []

    for date, row in df.iterrows():
        price = row["Close"]
        signal = row["Signal"]

        # BUY signal
        if signal == 1 and shares == 0:
            shares = capital / price
            capital = 0

        # SELL signal
        elif signal == -1 and shares > 0:
            capital = shares * price
            shares = 0

        # Portfolio value = cash + shares value
        portfolio_value = capital + (shares * price)

        portfolio_values.append(portfolio_value)
        cash_values.append(capital)
        share_values.append(shares * price)

    df["Cash"] = cash_values
    df["Stock_Value"] = share_values
    df["Portfolio_Value"] = portfolio_values

    return df


def performance_summary(df: pd.DataFrame):
    start_value = df["Portfolio_Value"].iloc[0]
    end_value = df["Portfolio_Value"].iloc[-1]
    total_return = (end_value - start_value) / start_value * 100

    print(f"Starting Capital: ${start_value:,.2f}")
    print(f"Ending Value:     ${end_value:,.2f}")
    print(f"Total Return:     {total_return:.2f}%")
    

if __name__ == "__main__":
    # Test with one stock
    df = pd.read_csv("data/AAPL.csv", parse_dates=["Date"]).set_index("Date")
    result = backtest(df)
    performance_summary(result)
