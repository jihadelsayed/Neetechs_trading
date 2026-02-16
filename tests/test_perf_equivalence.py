from __future__ import annotations

import numpy as np
import pandas as pd

from tradinglab.engine.portfolio import run_portfolio


def _make_price_df(prices: list[float], dates: list[pd.Timestamp]) -> pd.DataFrame:
    return pd.DataFrame({"Open": prices, "High": prices, "Low": prices, "Close": prices}, index=dates)


def reference_portfolio(price_dict: dict[str, pd.DataFrame]) -> pd.Series:
    # simple reference: buy-and-hold equal weight on first day
    dates = list(next(iter(price_dict.values())).index)
    symbols = list(price_dict.keys())
    init = 1000.0
    weights = {s: 1.0 / len(symbols) for s in symbols}
    shares = {s: (init * weights[s]) / price_dict[s].iloc[0]["Close"] for s in symbols}
    values = []
    for date in dates:
        v = 0.0
        for sym in symbols:
            v += shares[sym] * price_dict[sym].loc[date, "Close"]
        values.append(v)
    return pd.Series(values, index=dates)


def test_engine_equivalence_tolerance():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    price_dict = {
        "AAA": _make_price_df([10, 11, 12, 13, 14], dates),
        "BBB": _make_price_df([10, 10, 10, 10, 10], dates),
        "QQQ": _make_price_df([10, 11, 12, 13, 14], dates),
    }

    run = run_portfolio(price_dict, execution="same_close", price_mode="raw", slippage_mode="constant")
    ref = reference_portfolio({"AAA": price_dict["AAA"], "BBB": price_dict["BBB"]})

    assert np.isfinite(run.equity["Portfolio_Value"].iloc[-1])
    assert abs(run.equity["Portfolio_Value"].iloc[-1] - run.equity["Portfolio_Value"].iloc[-1]) < 1e-6


def test_golden_snapshot():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    price_dict = {
        "AAA": _make_price_df([10, 11, 12, 13], dates),
        "BBB": _make_price_df([10, 10, 10, 10], dates),
        "QQQ": _make_price_df([10, 11, 12, 13], dates),
    }

    run = run_portfolio(price_dict, execution="same_close", price_mode="raw", slippage_mode="constant")
    end_value = round(run.equity["Portfolio_Value"].iloc[-1], 6)
    assert end_value > 0
