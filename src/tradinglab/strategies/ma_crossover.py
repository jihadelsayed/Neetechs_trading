from __future__ import annotations

import pandas as pd

from tradinglab.config import SHORT_WINDOW, LONG_WINDOW, MIN_VOL20


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column")

    out = df.copy()
    out["SMA_Short"] = out["Close"].rolling(window=SHORT_WINDOW, min_periods=SHORT_WINDOW).mean()
    out["SMA_Long"] = out["Close"].rolling(window=LONG_WINDOW, min_periods=LONG_WINDOW).mean()
    out["DailyPct"] = out["Close"].pct_change() * 100.0
    out["Vol20"] = out["DailyPct"].rolling(20, min_periods=20).std()
    return out


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = add_indicators(df)
    out["Signal"] = 0

    valid = out["SMA_Short"].notna() & out["SMA_Long"].notna()
    short = out["SMA_Short"]
    long = out["SMA_Long"]

    buy = valid & (short > long) & (short.shift(1) <= long.shift(1))
    sell = valid & (short < long) & (short.shift(1) >= long.shift(1))

    buy = buy & out["Vol20"].notna() & (out["Vol20"] >= MIN_VOL20)

    out.loc[buy, "Signal"] = 1
    out.loc[sell, "Signal"] = -1

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
