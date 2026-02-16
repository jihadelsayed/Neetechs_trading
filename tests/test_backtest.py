from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from tradinglab.engine.portfolio import build_panel, run_portfolio, buy_hold_benchmark
from tradinglab.metrics.performance import turnover
from tradinglab.cli.main import main as cli_main


def _make_price_df(prices: list[float], dates: list[pd.Timestamp]) -> pd.DataFrame:
    return pd.DataFrame({"Close": prices}, index=dates)


def test_build_panel_no_nans_after_ffill_drop():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    price_dict = {
        "AAA": _make_price_df([1, np.nan, 3, 4, 5], dates),
        "BBB": _make_price_df([1, 2, np.nan, 4, 5], dates),
        "CCC": _make_price_df([np.nan, 2, 3, 4, np.nan], dates),
    }
    panel = build_panel(price_dict, min_coverage=0.6)
    assert not panel.isna().any().any()


def test_portfolio_deterministic_on_tiny_dataset(monkeypatch):
    import tradinglab.engine.portfolio as pf

    monkeypatch.setattr(pf, "MOM_LOOKBACK", 1)
    monkeypatch.setattr(pf, "LONG_WINDOW", 2)
    monkeypatch.setattr(pf, "TOP_N", 1)
    monkeypatch.setattr(pf, "REBALANCE", "D")
    monkeypatch.setattr(pf, "FEE_RATE", 0.0)
    monkeypatch.setattr(pf, "SLIPPAGE_RATE", 0.0)
    monkeypatch.setattr(pf, "REGIME_SYMBOL", "QQQ")
    monkeypatch.setattr(pf, "ALLOW_REGIME_TRADE", False)

    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    price_dict = {
        "AAA": _make_price_df([10, 11, 12, 13, 14], dates),
        "BBB": _make_price_df([10, 10, 10, 10, 10], dates),
    }

    run = run_portfolio(price_dict, regime_symbol="QQQ", allow_regime_trade=False)
    equity = run.equity["Portfolio_Value"].round(6).tolist()

    shares = 1000.0 / 11.0
    expected = [
        1000.0,  # day1 cash (regime not yet valid)
        1000.0,  # buy at close day2, equity still 1000
        shares * 12.0,
        shares * 13.0,
        shares * 14.0,
    ]
    expected = [round(x, 6) for x in expected]
    assert equity == expected


def test_buy_hold_benchmark_matches_manual():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    price_dict = {
        "AAA": _make_price_df([10, 12, 11], dates),
        "BBB": _make_price_df([20, 18, 22], dates),
    }

    bh = buy_hold_benchmark(price_dict, regime_symbol="QQQ", allow_regime_trade=False)

    init = 1000.0
    weight = 0.5
    shares_aaa = (init * weight) / 10.0
    shares_bbb = (init * weight) / 20.0

    manual = [
        shares_aaa * 10.0 + shares_bbb * 20.0,
        shares_aaa * 12.0 + shares_bbb * 18.0,
        shares_aaa * 11.0 + shares_bbb * 22.0,
    ]

    assert np.allclose(bh["BuyHold_Value"].values, manual)


def test_turnover_sanity():
    ledger = pd.DataFrame(
        [
            {"shares": 10, "price": 10.0},
            {"shares": 5, "price": 12.0},
        ]
    )
    equity = pd.Series([1000.0, 1100.0, 1050.0])
    t = turnover(ledger, equity)
    expected_notional = abs(10 * 10.0) + abs(5 * 12.0)
    expected = expected_notional / equity.mean()
    assert np.isclose(t, expected)


def test_smoke_main_cached_data(tmp_path, monkeypatch):
    monkeypatch.setattr("tradinglab.config.RESULTS_DIR", tmp_path)
    monkeypatch.setattr("tradinglab.cli.main.RESULTS_DIR", tmp_path)

    symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "QQQ"]
    cli_main(refresh_data=False, run_per_symbol=False, symbols=symbols)

    assert (tmp_path / "portfolio.csv").exists()
    assert (tmp_path / "portfolio_vs_bh.csv").exists()
    assert (tmp_path / "trade_ledger.csv").exists()
    assert (tmp_path / "positions.csv").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "metrics.json").exists()
