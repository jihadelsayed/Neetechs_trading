from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_paths(
    returns: pd.Series,
    paths: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if returns.empty:
        return pd.DataFrame(), pd.DataFrame()

    rng = np.random.default_rng(seed)
    rets = returns.dropna().values
    n = len(rets)

    equity_paths = []
    summary_rows = []

    for i in range(paths):
        sampled = rng.choice(rets, size=n, replace=True)
        equity = (1.0 + sampled).cumprod()
        equity = pd.Series(equity)
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        max_dd = float(drawdown.min())

        worst_1y = float((equity / equity.shift(252) - 1.0).min()) if n > 252 else float("nan")
        summary_rows.append(
            {
                "path_id": i,
                "final_equity": float(equity.iloc[-1]),
                "max_drawdown": max_dd,
                "worst_1y": worst_1y,
            }
        )
        equity_paths.append(equity)

    paths_df = pd.DataFrame(equity_paths).T
    paths_df.index.name = "day"
    paths_df.columns = [f"path_{i}" for i in range(paths)]

    summary_df = pd.DataFrame(summary_rows)
    return paths_df, summary_df


def monte_carlo_report(summary_df: pd.DataFrame) -> dict:
    if summary_df.empty:
        return {}

    p5_terminal = float(np.percentile(summary_df["final_equity"], 5))
    p95_dd = float(np.percentile(summary_df["max_drawdown"], 95))
    prob_loss = float((summary_df["final_equity"] < 1.0).mean())

    return {
        "P5_Terminal_Equity": p5_terminal,
        "P95_MaxDD": p95_dd,
        "Prob_Losing_Money": prob_loss,
    }
