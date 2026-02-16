from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tradinglab.config import RESULTS_DIR, REGIME_SYMBOL, PRICE_MODE, EXECUTION
from tradinglab.data.fetcher import load_or_fetch_symbols
from tradinglab.data.tickers import nasdaq100_tickers
from tradinglab.engine.portfolio import run_portfolio, buy_hold_single
from tradinglab.metrics.performance import sharpe
from tradinglab.experiments.splits import walk_forward_splits
from tradinglab.robustness.rolling import rolling_metrics, rolling_summary
from tradinglab.robustness.monte_carlo import bootstrap_paths, monte_carlo_report
from tradinglab.robustness.stability import parameter_stability
from tradinglab.robustness.regimes import regime_metrics
from tradinglab.robustness.allocation import allocation_study
from tradinglab.robustness.warnings import overfit_warnings


def _build_turnover_series(trade_ledger: pd.DataFrame, equity: pd.Series) -> pd.Series:
    if trade_ledger.empty:
        return pd.Series(index=equity.index, data=0.0)
    daily = trade_ledger.groupby("fill_date")["order_notional"].sum().reindex(equity.index).fillna(0.0)
    avg_equity = equity.rolling(20).mean().replace(0.0, np.nan)
    turnover = daily / avg_equity
    return turnover.fillna(0.0)


def _build_exposure_series(exposures: pd.DataFrame) -> pd.Series:
    if exposures.empty:
        return pd.Series(dtype=float)
    return exposures["Gross_Exposure_Pct"].fillna(0.0)


def _grid_params() -> list[dict]:
    grid = []
    for top_n in [5, 10, 15, 20]:
        for mom in [126, 189, 252]:
            for reb in ["W-FRI", "ME"]:
                for lw in [150, 200, 250]:
                    grid.append(
                        {
                            "top_n": top_n,
                            "mom_lookback": mom,
                            "rebalance": reb,
                            "long_window": lw,
                        }
                    )
    return grid


def _apply_params(run_kwargs: dict, params: dict) -> dict:
    out = run_kwargs.copy()
    out.update(params)
    return out


def run_robustness(
    symbols: list[str] | None,
    refresh_data: bool,
    start_date: str | None,
    end_date: str | None,
    walk_forward: bool,
    train_days: int,
    test_days: int,
    step_days: int,
    monte_carlo_paths: int,
    allocation_study_flag: bool,
    seed: int,
) -> Path:
    if symbols is None:
        symbols = nasdaq100_tickers()

    symbols = list(dict.fromkeys([s.strip().upper() for s in symbols if s]))
    if REGIME_SYMBOL not in symbols:
        symbols.append(REGIME_SYMBOL)

    price_dict = load_or_fetch_symbols(
        symbols,
        refresh=refresh_data,
        start_date=start_date,
        end_date=end_date,
    )
    if not price_dict:
        raise RuntimeError("No market data loaded.")

    out_dir = RESULTS_DIR / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Main strategy run
    run = run_portfolio(
        price_dict,
        execution=EXECUTION,
        price_mode=PRICE_MODE,
    )

    equity = run.equity["Portfolio_Value"]
    qqq = buy_hold_single(price_dict, REGIME_SYMBOL, price_mode=PRICE_MODE)
    qqq_series = qqq.iloc[:, 0]

    turnover_series = _build_turnover_series(run.trade_ledger, equity)
    exposure_series = _build_exposure_series(run.exposures)

    roll = rolling_metrics(equity, qqq_series, turnover_series, exposure_series)
    roll.to_csv(out_dir / "rolling_metrics.csv", index_label="Date")

    bench_cagr = (qqq_series / qqq_series.shift(252) - 1.0).reindex(roll.index)
    summary = rolling_summary(roll, bench_cagr)
    pd.DataFrame([summary]).to_csv(out_dir / "rolling_summary.csv", index=False)

    # Monte Carlo
    returns = equity.pct_change().dropna()
    paths_df, summary_df = bootstrap_paths(returns, paths=monte_carlo_paths, seed=seed)
    paths_df.to_csv(out_dir / "monte_carlo_paths.csv", index=True)
    summary_df.to_csv(out_dir / "monte_carlo_summary.csv", index=False)
    report = monte_carlo_report(summary_df)
    (out_dir / "monte_carlo_report.json").write_text(json.dumps(report, indent=2))

    # Parameter stability
    params_grid = _grid_params()
    stability_df = parameter_stability(
        price_dict,
        params_grid=params_grid,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        execution=EXECUTION,
        price_mode=PRICE_MODE,
        seed=seed,
    )
    stability_df.to_csv(out_dir / "parameter_stability.csv", index=False)

    # Regime segmentation
    regime_df = regime_metrics(price_dict, equity, REGIME_SYMBOL, PRICE_MODE)
    regime_df.to_csv(out_dir / "regime_metrics.csv", index=False)

    # Allocation study
    if allocation_study_flag:
        alloc_df = allocation_study(equity, qqq_series, target_vol=0.10, drawdown_threshold=0.15)
        alloc_df.to_csv(out_dir / "allocation_metrics.csv", index=False)

    # Overfit warnings
    warnings = []
    if walk_forward:
        panel_idx = pd.DataFrame(index=equity.index)
        splits = walk_forward_splits(panel_idx, train_days, test_days, step_days)
        grid_rows = []
        best_rows = []
        for split_id, split in enumerate(splits, start=1):
            test_slice = {
                sym: df.loc[split["test_start"] : split["test_end"]]
                for sym, df in price_dict.items()
                if not df.loc[split["test_start"] : split["test_end"]].empty
            }
            if not test_slice:
                continue
            best_sharpe = None
            best_param_id = None
            for param_id, param in enumerate(params_grid):
                run_test = run_portfolio(
                    test_slice,
                    execution=EXECUTION,
                    price_mode=PRICE_MODE,
                    slippage_mode="constant",
                    top_n=param["top_n"],
                    mom_lookback=param["mom_lookback"],
                    rebalance=param["rebalance"],
                    long_window=param["long_window"],
                )
                s = sharpe(run_test.equity["Portfolio_Value"])
                grid_rows.append({"split_id": split_id, "param_id": param_id, "test_sharpe": s})
                if best_sharpe is None or s > best_sharpe:
                    best_sharpe = s
                    best_param_id = param_id
            best_rows.append({"split_id": split_id, "param_id": best_param_id, "test_sharpe": best_sharpe})

        grid_df = pd.DataFrame(grid_rows)
        best_df = pd.DataFrame(best_rows)
        warnings = overfit_warnings(grid_df, best_df, equity)

    if warnings:
        (out_dir / "overfit_warnings.txt").write_text("\n".join(warnings))
    else:
        (out_dir / "overfit_warnings.txt").write_text("No warnings.")

    return out_dir
