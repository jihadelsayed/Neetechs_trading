from __future__ import annotations

import pandas as pd


def overfit_warnings(
    test_sharpes: pd.DataFrame,
    split_best: pd.DataFrame,
    equity: pd.Series,
) -> list[str]:
    warnings = []

    if not test_sharpes.empty:
        median = float(test_sharpes["test_sharpe"].median())
        best = float(test_sharpes["test_sharpe"].max())
        if median != 0 and best > 2.0 * median:
            warnings.append("Best Sharpe > 2x median Sharpe: potential overfit")

    if not split_best.empty:
        for i in range(len(split_best) - 1):
            current = split_best.iloc[i]
            next_row = split_best.iloc[i + 1]
            if current["param_id"] == next_row["param_id"] and next_row["test_sharpe"] < 0:
                warnings.append("Top params collapse in next window: instability detected")
                break

    if equity is not None and not equity.empty:
        yearly = equity.resample("YE").last().pct_change().dropna()
        total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        if total > 0 and not yearly.empty:
            if yearly.max() > 0.4 * total:
                warnings.append("One year contributes >40% of total return: concentration risk")

    return warnings
