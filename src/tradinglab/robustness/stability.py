from __future__ import annotations

import pandas as pd

from tradinglab.experiments.splits import walk_forward_splits
from tradinglab.engine.portfolio import run_portfolio, build_price_panels
from tradinglab.metrics.performance import sharpe


def parameter_stability(
    price_dict: dict[str, pd.DataFrame],
    params_grid: list[dict],
    train_days: int,
    test_days: int,
    step_days: int,
    execution: str,
    price_mode: str,
    seed: int,
) -> pd.DataFrame:
    panel_close, _ = build_price_panels(price_dict, price_mode=price_mode)
    if panel_close.empty:
        return pd.DataFrame()

    panel_idx = pd.DataFrame(index=panel_close.index)
    splits = walk_forward_splits(panel_idx, train_days, test_days, step_days)

    results = []

    for param in params_grid:
        test_sharpes = []
        for split in splits:
            test_slice = {
                sym: df.loc[split["test_start"] : split["test_end"]]
                for sym, df in price_dict.items()
                if not df.loc[split["test_start"] : split["test_end"]].empty
            }
            if not test_slice:
                continue

            run = run_portfolio(
                test_slice,
                execution=execution,
                price_mode=price_mode,
                max_position_weight=None,
                trailing_stop_pct=None,
                time_stop_days=None,
                target_vol=None,
                slippage_mode="constant",
                top_n=param["top_n"],
                mom_lookback=param["mom_lookback"],
                rebalance=param["rebalance"],
                long_window=param["long_window"],
            )
            test_sharpes.append(sharpe(run.equity["Portfolio_Value"]))

        if not test_sharpes:
            continue

        mean_sharpe = float(pd.Series(test_sharpes).mean())
        std_sharpe = float(pd.Series(test_sharpes).std(ddof=0))
        stability = mean_sharpe - std_sharpe

        row = {**param}
        row.update(
            {
                "mean_test_sharpe": mean_sharpe,
                "std_test_sharpe": std_sharpe,
                "stability_score": stability,
            }
        )
        results.append(row)

    return pd.DataFrame(results)
