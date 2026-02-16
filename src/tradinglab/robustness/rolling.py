from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_max_drawdown(series: pd.Series) -> pd.Series:
    running_max = series.cummax()
    drawdown = (series - running_max) / running_max
    return drawdown


def rolling_metrics(
    equity: pd.Series,
    benchmark: pd.Series,
    turnover_series: pd.Series,
    exposure_series: pd.Series,
    window: int = 252,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "equity": equity,
            "benchmark": benchmark,
            "turnover": turnover_series,
            "exposure": exposure_series,
        }
    ).dropna()

    returns = df["equity"].pct_change()
    bench_returns = df["benchmark"].pct_change()

    rolling_sharpe = (
        returns.rolling(window).mean() / returns.rolling(window).std(ddof=0)
    ) * np.sqrt(252)

    rolling_cagr = (df["equity"].rolling(window).apply(
        lambda x: (x.iloc[-1] / x.iloc[0]) ** (252 / window) - 1.0, raw=False
    ))

    rolling_dd = df["equity"].rolling(window).apply(
        lambda x: _rolling_max_drawdown(pd.Series(x)).min(), raw=False
    )

    rolling_corr = returns.rolling(window).corr(bench_returns)

    cov = returns.rolling(window).cov(bench_returns)
    var = bench_returns.rolling(window).var(ddof=0)
    rolling_beta = cov / var

    rolling_turnover = df["turnover"].rolling(window).mean()
    rolling_exposure = df["exposure"].rolling(window).mean()

    out = pd.DataFrame(
        {
            "Rolling_Sharpe": rolling_sharpe,
            "Rolling_CAGR": rolling_cagr,
            "Rolling_MaxDD": rolling_dd,
            "Rolling_Corr_QQQ": rolling_corr,
            "Rolling_Beta_QQQ": rolling_beta,
            "Rolling_Turnover": rolling_turnover,
            "Rolling_Exposure": rolling_exposure,
        }
    ).dropna()

    return out


def rolling_summary(rolling_df: pd.DataFrame, benchmark_cagr: pd.Series) -> dict:
    if rolling_df.empty:
        return {}

    positive_sharpe = float((rolling_df["Rolling_Sharpe"] > 0).mean())
    beating_qqq = float((rolling_df["Rolling_CAGR"] > benchmark_cagr.loc[rolling_df.index]).mean())
    dd_worse = float((rolling_df["Rolling_MaxDD"] < -0.30).mean())

    return {
        "Pct_Positive_Sharpe": positive_sharpe,
        "Pct_Beating_QQQ": beating_qqq,
        "Pct_DD_Worse_30": dd_worse,
    }
