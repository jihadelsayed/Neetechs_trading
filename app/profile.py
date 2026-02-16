from __future__ import annotations

import cProfile
import pstats
import time
from pathlib import Path

import pandas as pd

from tradinglab.config import REGIME_SYMBOL
from tradinglab.data.fetcher import load_or_fetch_symbols
from tradinglab.data.tickers import nasdaq100_tickers
from tradinglab.engine.portfolio import build_price_panels, run_portfolio
from tradinglab.metrics.performance import compute_metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Profile backtest")
    parser.add_argument("--universe", choices=["small", "nasdaq100"], default="small")
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args()

    if args.universe == "small":
        symbols = ["AAPL", "MSFT", "NVDA", "AMZN", REGIME_SYMBOL]
    else:
        symbols = nasdaq100_tickers()
        if REGIME_SYMBOL not in symbols:
            symbols.append(REGIME_SYMBOL)

    t0 = time.perf_counter()
    price_dict = load_or_fetch_symbols(symbols, refresh=False, start_date=args.start, end_date=args.end)
    t1 = time.perf_counter()

    panel_close, _ = build_price_panels(price_dict)
    t2 = time.perf_counter()

    run = run_portfolio(price_dict)
    t3 = time.perf_counter()

    _ = compute_metrics("Portfolio", run.equity["Portfolio_Value"], run.trade_ledger)
    t4 = time.perf_counter()

    timing = {
        "data_load_s": t1 - t0,
        "panel_build_s": t2 - t1,
        "run_portfolio_s": t3 - t2,
        "metrics_s": t4 - t3,
        "total_s": t4 - t0,
    }

    profile_path = Path("logs") / "profile_stats.txt"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    profiler.enable()
    run_portfolio(price_dict)
    profiler.disable()

    with profile_path.open("w", encoding="utf-8") as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats("cumulative")
        stats.print_stats(50)

    timing_path = Path("logs") / "profile_timing.json"
    timing_path.write_text(pd.Series(timing).to_json())

    print(f"Profile saved: {profile_path}")
    print(f"Timing saved: {timing_path}")


if __name__ == "__main__":
    main()
