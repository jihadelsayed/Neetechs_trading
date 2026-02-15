# strategy.py
from __future__ import annotations

import pandas as pd

from config import SHORT_WINDOW, LONG_WINDOW


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add indicators to a price DataFrame.
    Requires columns: Close (at minimum).
    """
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column")

    out = df.copy()

    # Simple moving averages
    out["SMA_Short"] = out["Close"].rolling(window=SHORT_WINDOW, min_periods=SHORT_WINDOW).mean()
    out["SMA_Long"] = out["Close"].rolling(window=LONG_WINDOW, min_periods=LONG_WINDOW).mean()

    return out


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a 'Signal' column:
      +1 when short SMA crosses above long SMA (buy)
      -1 when short SMA crosses below long SMA (sell)
       0 otherwise
    Also creates 'Position' which is the current held state (0/1).
    """
    out = add_indicators(df)

    out["Signal"] = 0

    # Need both SMAs available
    valid = out["SMA_Short"].notna() & out["SMA_Long"].notna()
    out.loc[valid & (out["SMA_Short"] > out["SMA_Long"]), "Signal"] = 1
    out.loc[valid & (out["SMA_Short"] < out["SMA_Long"]), "Signal"] = -1

    # Convert raw comparisons into "cross" signals
    # Cross happens when Signal changes from -1 to 1 (buy) or 1 to -1 (sell)
    out["Signal"] = out["Signal"].diff().fillna(0)

    # Normalize:
    # +2 means went from -1 to +1 => BUY
    # -2 means went from +1 to -1 => SELL
    out.loc[out["Signal"] > 0, "Signal"] = 1
    out.loc[out["Signal"] < 0, "Signal"] = -1

    # Position: 1 if holding, 0 if not (based on signals)
    position = []
    holding = 0
    for s in out["Signal"].astype(int).tolist():
        if s == 1:
            holding = 1
        elif s == -1:
            holding = 0
        position.append(holding)

    out["Position"] = position
    return out


if __name__ == "__main__":
    # quick sanity check with a single CSV if you want
    import pandas as pd

    df = pd.read_csv("data/AAPL.csv", parse_dates=["Date"]).set_index("Date")
    sig = generate_signals(df)
    print(sig[["Close", "SMA_Short", "SMA_Long", "Signal", "Position"]].tail(20))
