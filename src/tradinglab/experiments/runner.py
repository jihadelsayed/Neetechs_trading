from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from tradinglab.config import RESULTS_DIR, REGIME_SYMBOL, PRICE_MODE, EXECUTION
from tradinglab.data.fetcher import load_or_fetch_symbols
from tradinglab.data.tickers import nasdaq100_tickers
from tradinglab.engine.portfolio import run_portfolio, buy_hold_benchmark, buy_hold_single
from tradinglab.metrics.performance import compute_metrics
from tradinglab.experiments.splits import split_by_date, walk_forward_splits


def _slice_price_dict(price_dict: dict[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    sliced: dict[str, pd.DataFrame] = {}
    for sym, df in price_dict.items():
        sliced_df = df.loc[start:end]
        if not sliced_df.empty:
            sliced[sym] = sliced_df
    return sliced


def run_experiment(
    symbols: list[str] | None,
    refresh_data: bool,
    start_date: str | None,
    end_date: str | None,
    split_date: str | None,
    walk_forward: bool,
    train_days: int,
    test_days: int,
    step_days: int,
    grid_search: bool = False,
    max_combos: int | None = None,
    jobs: int = 1,
    execution: str | None = None,
    price_mode: str | None = None,
    max_position_weight: float | None = None,
    trailing_stop: float | None = None,
    time_stop: int | None = None,
    target_vol: float | None = None,
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

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / "experiments" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    config_used = {
        "symbols": symbols,
        "refresh_data": refresh_data,
        "start_date": start_date,
        "end_date": end_date,
        "split_date": split_date,
        "walk_forward": walk_forward,
        "train_days": train_days,
        "test_days": test_days,
        "step_days": step_days,
        "execution": execution,
        "price_mode": price_mode,
        "max_position_weight": max_position_weight,
        "trailing_stop": trailing_stop,
        "time_stop": time_stop,
        "target_vol": target_vol,
    }
    (out_dir / "config_used.json").write_text(json.dumps(config_used, indent=2))

    panel = None
    for df in price_dict.values():
        panel = df
        break

    if walk_forward:
        # use close panels from one symbol for date index
        sample = next(iter(price_dict.values()))
        date_index = sample.index
        panel_idx = pd.DataFrame(index=date_index)
        splits = walk_forward_splits(panel_idx, train_days, test_days, step_days)
    else:
        if split_date is None:
            raise ValueError("split_date is required for fixed split")
        sample = next(iter(price_dict.values()))
        panel_idx = pd.DataFrame(index=sample.index)
        train_idx, test_idx = split_by_date(panel_idx, split_date)
        splits = [
            {
                "train_start": train_idx.index[0],
                "train_end": train_idx.index[-1],
                "test_start": test_idx.index[0],
                "test_end": test_idx.index[-1],
            }
        ]

    per_split_rows: list[dict] = []
    equity_rows: list[dict] = []
    ledger_rows: list[dict] = []

    grid_train_rows: list[dict] = []
    grid_test_rows: list[dict] = []
    grid_selected_rows: list[dict] = []

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

    def _score_run(run) -> float:
        from tradinglab.metrics.performance import sharpe

        return sharpe(run.equity["Portfolio_Value"])

    params_grid = _grid_params() if grid_search else []
    if max_combos is not None and params_grid:
        params_grid = params_grid[: max_combos]

    for split_id, split in enumerate(splits, start=1):
        train_slice = _slice_price_dict(price_dict, split["train_start"], split["train_end"])
        test_slice = _slice_price_dict(price_dict, split["test_start"], split["test_end"])

        if grid_search and train_slice:
            def _run_param(param: dict) -> dict:
                run_train = run_portfolio(
                    train_slice,
                    execution=execution or EXECUTION,
                    price_mode=price_mode or PRICE_MODE,
                    slippage_mode="constant",
                    top_n=param["top_n"],
                    mom_lookback=param["mom_lookback"],
                    rebalance=param["rebalance"],
                    long_window=param["long_window"],
                )
                return {"param": param, "train_sharpe": _score_run(run_train)}

            if jobs > 1:
                from joblib import Parallel, delayed

                results = Parallel(n_jobs=max(1, jobs))(delayed(_run_param)(p) for p in params_grid)
            else:
                results = [_run_param(p) for p in params_grid]

            for res in results:
                row = {"split_id": split_id, **res["param"], "train_sharpe": res["train_sharpe"]}
                grid_train_rows.append(row)

            top = sorted(results, key=lambda x: x["train_sharpe"], reverse=True)[:3]
            for selected in top:
                grid_selected_rows.append(
                    {"split_id": split_id, **selected["param"], "train_sharpe": selected["train_sharpe"]}
                )

                if test_slice:
                    run_test = run_portfolio(
                        test_slice,
                        execution=execution or EXECUTION,
                        price_mode=price_mode or PRICE_MODE,
                        slippage_mode="constant",
                        top_n=selected["param"]["top_n"],
                        mom_lookback=selected["param"]["mom_lookback"],
                        rebalance=selected["param"]["rebalance"],
                        long_window=selected["param"]["long_window"],
                    )
                    grid_test_rows.append(
                        {
                            "split_id": split_id,
                            **selected["param"],
                            "test_sharpe": _score_run(run_test),
                        }
                    )

        for label, dataset in [("train", train_slice), ("test", test_slice)]:
            if not dataset:
                continue

            run = run_portfolio(
                dataset,
                execution=execution or EXECUTION,
                price_mode=price_mode or PRICE_MODE,
                max_position_weight=max_position_weight,
                trailing_stop_pct=trailing_stop,
                time_stop_days=time_stop,
                target_vol=target_vol,
            )

            pf_metrics = compute_metrics(f"Portfolio_{label}", run.equity["Portfolio_Value"], run.trade_ledger)
            pf_metrics["split_id"] = split_id
            pf_metrics["dataset"] = label
            per_split_rows.append(pf_metrics)

            bh = buy_hold_benchmark(dataset, price_mode=price_mode or PRICE_MODE)
            bh_metrics = compute_metrics(f"BuyHold_{label}", bh["BuyHold_Value"])
            bh_metrics["split_id"] = split_id
            bh_metrics["dataset"] = label
            per_split_rows.append(bh_metrics)

            if REGIME_SYMBOL in dataset:
                qqq = buy_hold_single(dataset, REGIME_SYMBOL, price_mode=price_mode or PRICE_MODE)
                qqq_metrics = compute_metrics(f"{REGIME_SYMBOL}_{label}", qqq.iloc[:, 0])
                qqq_metrics["split_id"] = split_id
                qqq_metrics["dataset"] = label
                per_split_rows.append(qqq_metrics)

            if label == "test":
                merged = run.equity[["Portfolio_Value"]].join(bh, how="inner")
                if REGIME_SYMBOL in dataset:
                    merged = merged.join(qqq, how="left")

                for dt, row in merged.iterrows():
                    equity_rows.append(
                        {
                            "split_id": split_id,
                            "date": dt,
                            "portfolio_value": row["Portfolio_Value"],
                            "buyhold_value": row["BuyHold_Value"],
                            "qqq_value": row.get(f"{REGIME_SYMBOL}_BuyHold", None),
                        }
                    )

                if not run.trade_ledger.empty:
                    ledger = run.trade_ledger.copy()
                    ledger.insert(0, "split_id", split_id)
                    ledger_rows.append(ledger)

    per_split_df = pd.DataFrame(per_split_rows)
    per_split_df.to_csv(out_dir / "per_split_metrics.csv", index=False)

    if not per_split_df.empty:
        summary = per_split_df.select_dtypes(include=["number"]).groupby(
            [per_split_df["Label"], per_split_df["dataset"]]
        ).mean()
        summary.reset_index().to_csv(out_dir / "metrics_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "metrics_summary.csv", index=False)

    equity_df = pd.DataFrame(equity_rows)
    equity_df.to_csv(out_dir / "equity_curves.csv", index=False)

    if ledger_rows:
        ledger_df = pd.concat(ledger_rows, ignore_index=True)
        ledger_df.to_csv(out_dir / "trade_ledger.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "trade_ledger.csv", index=False)

    if grid_search:
        pd.DataFrame(grid_train_rows).to_csv(out_dir / "grid_results_train.csv", index=False)
        pd.DataFrame(grid_selected_rows).to_csv(out_dir / "grid_selected.csv", index=False)
        pd.DataFrame(grid_test_rows).to_csv(out_dir / "grid_results_test.csv", index=False)

    return out_dir
