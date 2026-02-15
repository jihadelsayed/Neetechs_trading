# main.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

from config import SYMBOLS
from data_fetcher import get_market_data
from backtest import backtest


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def summarize(symbol: str, df: pd.DataFrame) -> dict:
    start = float(df["Portfolio_Value"].iloc[0])
    end = float(df["Portfolio_Value"].iloc[-1])
    total_return = (end - start) / start * 100.0

    # Simple max drawdown
    running_max = df["Portfolio_Value"].cummax()
    drawdown = (df["Portfolio_Value"] - running_max) / running_max
    max_drawdown = float(drawdown.min() * 100.0)

    return {
        "Symbol": symbol,
        "Start": round(start, 2),
        "End": round(end, 2),
        "Return_%": round(total_return, 2),
        "MaxDrawdown_%": round(max_drawdown, 2),
    }


def main(refresh_data: bool = False):
    market = get_market_data(refresh=refresh_data)

    rows = []
    for sym, df in market.items():
        bt = backtest(df)

        # Save per-symbol backtest output
        out_path = RESULTS_DIR / f"{sym}_backtest.csv"
        bt.to_csv(out_path)

        rows.append(summarize(sym, bt))

    summary_df = pd.DataFrame(rows).sort_values(by="Return_%", ascending=False)

    print("\n=== Backtest Summary (best -> worst) ===")
    print(summary_df.to_string(index=False))

    # Save summary too
    summary_df.to_csv(RESULTS_DIR / "summary.csv", index=False)
    print("\nSaved results/summary.csv and per-symbol backtests in results/")



if __name__ == "__main__":
    # Set refresh_data=True if you want to re-download latest data
    main(refresh_data=False)
