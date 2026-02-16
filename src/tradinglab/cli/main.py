from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from tradinglab.config import (
    INITIAL_CAPITAL,
    RESULTS_DIR,
    REGIME_SYMBOL,
    ALLOW_REGIME_TRADE,
)
from tradinglab.data.fetcher import load_or_fetch_symbols
from tradinglab.data.tickers import nasdaq100_tickers
from tradinglab.engine.portfolio import run_portfolio, buy_hold_benchmark, buy_hold_single
from tradinglab.metrics.performance import compute_metrics


RESULTS_DIR.mkdir(exist_ok=True)


def main(
    refresh_data: bool = False,
    run_per_symbol: bool = False,
    symbols: Iterable[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    top_n: int | None = None,
    rebalance: str | None = None,
) -> None:
    if symbols is None:
        symbols = nasdaq100_tickers()

    symbols = list(dict.fromkeys([s.strip().upper() for s in symbols if s]))
    if REGIME_SYMBOL not in symbols:
        symbols.append(REGIME_SYMBOL)

    print("Universe size:", len(symbols))

    if top_n is not None or rebalance is not None:
        import tradinglab.engine.portfolio as portfolio

        if top_n is not None:
            portfolio.TOP_N = int(top_n)
        if rebalance is not None:
            portfolio.REBALANCE = rebalance

    price_dict = load_or_fetch_symbols(
        symbols,
        refresh=refresh_data,
        start_date=start_date,
        end_date=end_date,
    )

    if not price_dict:
        raise RuntimeError("No market data loaded. Everything got skipped.")

    missing = [s for s in symbols if s not in price_dict]
    if missing:
        print("Skipped symbols:", len(missing))

    # Portfolio rotation
    portfolio_run = run_portfolio(
        price_dict,
        regime_symbol=REGIME_SYMBOL,
        allow_regime_trade=ALLOW_REGIME_TRADE,
    )
    pf = portfolio_run.equity
    pf.to_csv(RESULTS_DIR / "portfolio.csv", index_label="Date")

    # Trade ledger + positions
    trade_ledger = portfolio_run.trade_ledger
    trade_ledger.to_csv(RESULTS_DIR / "trade_ledger.csv", index=False)

    positions = portfolio_run.positions
    positions.to_csv(RESULTS_DIR / "positions.csv", index_label="Date")

    # Buy & hold benchmark of same basket
    bh = buy_hold_benchmark(
        price_dict,
        regime_symbol=REGIME_SYMBOL,
        allow_regime_trade=ALLOW_REGIME_TRADE,
    )

    merged = pf.join(bh, how="inner")
    merged.to_csv(RESULTS_DIR / "portfolio_vs_bh.csv", index_label="Date")

    # QQQ buy & hold benchmark
    qqq_bh = None
    if REGIME_SYMBOL in price_dict:
        qqq_bh = buy_hold_single(price_dict, REGIME_SYMBOL)
        qqq_bh.to_csv(RESULTS_DIR / f"{REGIME_SYMBOL}_buyhold.csv", index_label="Date")

    # Metrics
    metrics_rows: List[dict] = []
    metrics_rows.append(compute_metrics("Portfolio", pf["Portfolio_Value"], trade_ledger))
    metrics_rows.append(compute_metrics("BuyHold_Basket", bh["BuyHold_Value"]))
    if qqq_bh is not None:
        metrics_rows.append(compute_metrics(f"{REGIME_SYMBOL}_BuyHold", qqq_bh.iloc[:, 0]))

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    metrics_df.to_json(RESULTS_DIR / "metrics.json", orient="records", indent=2)

    # Print summary
    pf_end = float(merged["Portfolio_Value"].iloc[-1])
    pf_dd = float(merged["Drawdown_%"].min())

    bh_end = float(merged["BuyHold_Value"].iloc[-1])
    bh_dd = float(merged["BuyHold_Drawdown_%"].min())

    print("\n=== Portfolio Rotation (Universe) ===")
    print(merged[["Portfolio_Value", "Return_%", "Drawdown_%"]].tail(5))

    print("\nPortfolio start:", float(INITIAL_CAPITAL))
    print("Portfolio end:", pf_end)
    print("Portfolio Max DD %:", pf_dd)

    print("\nBuy&Hold basket start:", float(INITIAL_CAPITAL))
    print("Buy&Hold basket end:", bh_end)
    print("Buy&Hold basket Max DD %:", bh_dd)

    print("\nSaved results/portfolio.csv and results/portfolio_vs_bh.csv")


if __name__ == "__main__":
    main(refresh_data=False, run_per_symbol=False)
