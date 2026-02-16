from __future__ import annotations

import numpy as np
import pandas as pd

from tradinglab.engine.portfolio import run_portfolio


def _make_ohlc(prices: list[float], dates: list[pd.Timestamp]) -> pd.DataFrame:
    return pd.DataFrame({"Open": prices, "Close": prices}, index=dates)


def test_no_lookahead_next_open(monkeypatch):
    import tradinglab.engine.portfolio as pf

    monkeypatch.setattr(pf, "MOM_LOOKBACK", 1)
    monkeypatch.setattr(pf, "LONG_WINDOW", 2)
    monkeypatch.setattr(pf, "TOP_N", 1)
    monkeypatch.setattr(pf, "REBALANCE", "D")
    monkeypatch.setattr(pf, "FEE_RATE", 0.0)
    monkeypatch.setattr(pf, "SLIPPAGE_RATE", 0.0)
    monkeypatch.setattr(pf, "SLIPPAGE_MODE", "constant")
    monkeypatch.setattr(pf, "REGIME_SYMBOL", "QQQ")
    monkeypatch.setattr(pf, "ALLOW_REGIME_TRADE", False)

    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    price_dict = {
        "AAA": _make_ohlc([10, 11, 12, 13, 14], dates),
        "BBB": _make_ohlc([10, 10, 10, 10, 10], dates),
        "QQQ": _make_ohlc([10, 11, 12, 13, 14], dates),
    }

    run = run_portfolio(
        price_dict,
        execution="next_open",
        price_mode="raw",
        slippage_mode="constant",
    )

    ledger = run.trade_ledger
    assert not ledger.empty
    assert (ledger["fill_date"] > ledger["signal_date"]).all()


def test_max_position_weight(monkeypatch):
    import tradinglab.engine.portfolio as pf

    monkeypatch.setattr(pf, "MOM_LOOKBACK", 1)
    monkeypatch.setattr(pf, "LONG_WINDOW", 2)
    monkeypatch.setattr(pf, "TOP_N", 2)
    monkeypatch.setattr(pf, "REBALANCE", "D")
    monkeypatch.setattr(pf, "FEE_RATE", 0.0)
    monkeypatch.setattr(pf, "SLIPPAGE_RATE", 0.0)
    monkeypatch.setattr(pf, "SLIPPAGE_MODE", "constant")
    monkeypatch.setattr(pf, "REGIME_SYMBOL", "QQQ")
    monkeypatch.setattr(pf, "ALLOW_REGIME_TRADE", False)

    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    price_dict = {
        "AAA": _make_ohlc([10, 11, 12, 13], dates),
        "BBB": _make_ohlc([9, 9.5, 10, 10.5], dates),
        "QQQ": _make_ohlc([10, 11, 12, 13], dates),
    }

    run = run_portfolio(
        price_dict,
        execution="same_close",
        price_mode="raw",
        max_position_weight=0.2,
        cash_buffer=0.0,
        slippage_mode="constant",
    )
    last = run.positions.iloc[-1]
    close = run.panel_close.iloc[-1]
    equity = run.equity["Portfolio_Value"].iloc[-1]

    w_aaa = (last["AAA"] * close["AAA"]) / equity
    w_bbb = (last["BBB"] * close["BBB"]) / equity
    assert w_aaa <= 0.2005
    assert w_bbb <= 0.2005


def test_max_turnover_per_rebalance(monkeypatch):
    import tradinglab.engine.portfolio as pf

    monkeypatch.setattr(pf, "MOM_LOOKBACK", 1)
    monkeypatch.setattr(pf, "LONG_WINDOW", 2)
    monkeypatch.setattr(pf, "TOP_N", 1)
    monkeypatch.setattr(pf, "REBALANCE", "D")
    monkeypatch.setattr(pf, "FEE_RATE", 0.0)
    monkeypatch.setattr(pf, "SLIPPAGE_RATE", 0.0)
    monkeypatch.setattr(pf, "SLIPPAGE_MODE", "constant")
    monkeypatch.setattr(pf, "REGIME_SYMBOL", "QQQ")
    monkeypatch.setattr(pf, "ALLOW_REGIME_TRADE", False)

    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    price_dict = {
        "AAA": _make_ohlc([10, 11, 12, 13], dates),
        "BBB": _make_ohlc([20, 19, 18, 17], dates),
        "QQQ": _make_ohlc([10, 11, 12, 13], dates),
    }

    run = run_portfolio(
        price_dict,
        execution="same_close",
        price_mode="raw",
        max_turnover_per_rebalance=0.1,
        cash_buffer=0.0,
        slippage_mode="constant",
    )

    ledger = run.trade_ledger
    assert not ledger.empty
    nav = run.equity["Portfolio_Value"].iloc[0]
    first_signal = ledger["signal_date"].min()
    traded = ledger.loc[ledger["signal_date"] == first_signal, "order_notional"].sum()
    assert traded <= nav * 0.11


def test_max_gross_exposure_and_cash_buffer(monkeypatch):
    import tradinglab.engine.portfolio as pf

    monkeypatch.setattr(pf, "MOM_LOOKBACK", 1)
    monkeypatch.setattr(pf, "LONG_WINDOW", 2)
    monkeypatch.setattr(pf, "TOP_N", 2)
    monkeypatch.setattr(pf, "REBALANCE", "D")
    monkeypatch.setattr(pf, "FEE_RATE", 0.0)
    monkeypatch.setattr(pf, "SLIPPAGE_RATE", 0.0)
    monkeypatch.setattr(pf, "SLIPPAGE_MODE", "constant")
    monkeypatch.setattr(pf, "REGIME_SYMBOL", "QQQ")
    monkeypatch.setattr(pf, "ALLOW_REGIME_TRADE", False)

    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    price_dict = {
        "AAA": _make_ohlc([10, 11, 12, 13], dates),
        "BBB": _make_ohlc([9, 9.5, 10, 10.5], dates),
        "QQQ": _make_ohlc([10, 11, 12, 13], dates),
    }

    run = run_portfolio(
        price_dict,
        execution="same_close",
        price_mode="raw",
        max_gross_exposure=0.5,
        cash_buffer=0.2,
        slippage_mode="constant",
    )

    exposures = run.exposures
    assert exposures["Gross_Exposure_Pct"].max() <= 0.5005
    assert exposures["Cash_Pct"].min() >= 0.199


def test_trailing_and_time_stop(monkeypatch):
    import tradinglab.engine.portfolio as pf

    monkeypatch.setattr(pf, "MOM_LOOKBACK", 1)
    monkeypatch.setattr(pf, "LONG_WINDOW", 2)
    monkeypatch.setattr(pf, "TOP_N", 1)
    monkeypatch.setattr(pf, "REBALANCE", "D")
    monkeypatch.setattr(pf, "FEE_RATE", 0.0)
    monkeypatch.setattr(pf, "SLIPPAGE_RATE", 0.0)
    monkeypatch.setattr(pf, "SLIPPAGE_MODE", "constant")
    monkeypatch.setattr(pf, "REGIME_SYMBOL", "QQQ")
    monkeypatch.setattr(pf, "ALLOW_REGIME_TRADE", False)

    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    price_dict = {
        "AAA": _make_ohlc([10, 12, 14, 12, 11], dates),
        "QQQ": _make_ohlc([10, 11, 12, 13, 14], dates),
    }

    run_trail = run_portfolio(
        price_dict,
        execution="same_close",
        price_mode="raw",
        trailing_stop_pct=0.15,
        slippage_mode="constant",
    )
    assert (run_trail.trade_ledger["action"] == "SELL").any()

    run_time = run_portfolio(
        price_dict,
        execution="same_close",
        price_mode="raw",
        time_stop_days=2,
        slippage_mode="constant",
    )
    assert (run_time.trade_ledger["action"] == "SELL").any()


def test_target_vol_scaling(monkeypatch):
    import tradinglab.engine.portfolio as pf

    monkeypatch.setattr(pf, "MOM_LOOKBACK", 1)
    monkeypatch.setattr(pf, "LONG_WINDOW", 2)
    monkeypatch.setattr(pf, "TOP_N", 2)
    monkeypatch.setattr(pf, "REBALANCE", "D")
    monkeypatch.setattr(pf, "FEE_RATE", 0.0)
    monkeypatch.setattr(pf, "SLIPPAGE_RATE", 0.0)
    monkeypatch.setattr(pf, "SLIPPAGE_MODE", "constant")
    monkeypatch.setattr(pf, "REGIME_SYMBOL", "QQQ")
    monkeypatch.setattr(pf, "ALLOW_REGIME_TRADE", False)

    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    price_dict = {
        "AAA": _make_ohlc([10, 12, 11, 13, 12, 14], dates),
        "BBB": _make_ohlc([20, 18, 19, 17, 18, 16], dates),
        "QQQ": _make_ohlc([10, 11, 12, 13, 14, 15], dates),
    }

    run = run_portfolio(
        price_dict,
        execution="same_close",
        price_mode="raw",
        target_vol=0.05,
        vol_lookback=3,
        slippage_mode="constant",
    )
    assert run.exposures["Gross_Exposure_Pct"].max() <= 1.0
