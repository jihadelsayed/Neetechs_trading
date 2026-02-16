from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from tradinglab.config import REGIME_SYMBOL
from tradinglab.data.fetcher import load_or_fetch_symbols
from tradinglab.data.tickers import nasdaq100_tickers
from tradinglab.engine.portfolio import build_price_panels, run_portfolio
from tradinglab.metrics.performance import compute_metrics


def run_bench(universe: str) -> Path:
    if universe == "small":
        symbols = ["AAPL", "MSFT", "NVDA", "AMZN", REGIME_SYMBOL]
    else:
        symbols = nasdaq100_tickers()
        if REGIME_SYMBOL not in symbols:
            symbols.append(REGIME_SYMBOL)

    t0 = time.perf_counter()
    price_dict = load_or_fetch_symbols(symbols, refresh=False)
    t1 = time.perf_counter()

    panel_close, _ = build_price_panels(price_dict)
    t2 = time.perf_counter()

    run = run_portfolio(price_dict)
    t3 = time.perf_counter()

    metrics = compute_metrics("Portfolio", run.equity["Portfolio_Value"], run.trade_ledger)
    t4 = time.perf_counter()

    result = {
        "data_load_s": t1 - t0,
        "panel_build_s": t2 - t1,
        "run_portfolio_s": t3 - t2,
        "metrics_s": t4 - t3,
        "total_s": t4 - t0,
        "metrics": metrics,
    }

    out_path = Path("logs") / "bench_baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    return out_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run benchmark")
    parser.add_argument("--universe", choices=["small", "nasdaq100"], default="small")
    args = parser.parse_args()

    path = run_bench(args.universe)
    print(f"Saved benchmark: {path}")


if __name__ == "__main__":
    main()
