from __future__ import annotations

import os
import json
import tempfile
from pathlib import Path

import pandas as pd

from app.live.runner import run_live_cycle
from app.brokers.base import Order


def _mock_history():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0],
            "High": [10.5, 11.5, 12.5],
            "Low": [9.5, 10.5, 11.5],
            "Close": [10.0, 11.0, 12.0],
            "Adj Close": [10.0, 11.0, 12.0],
        },
        index=dates,
    )
    return {"AAA": df, "QQQ": df}


def test_live_gating_refuses(monkeypatch, tmp_path):
    monkeypatch.setattr("tradinglab.data.fetcher.load_or_fetch_symbols", lambda *a, **k: _mock_history())
    orders = run_live_cycle(
        universe="small",
        refresh_data=False,
        execution="next_open",
        price_mode="raw",
        dry_run=False,
        state_path=tmp_path / "state.json",
        flatten=False,
        mode="live",
        confirm=False,
        accept_real_trading="NO",
        flatten_on_kill=False,
    )
    assert orders == []


def test_kill_switch_blocks(monkeypatch, tmp_path):
    monkeypatch.setattr("tradinglab.data.fetcher.load_or_fetch_symbols", lambda *a, **k: _mock_history())
    os.environ["KILL_SWITCH"] = "true"
    orders = run_live_cycle(
        universe="small",
        refresh_data=False,
        execution="next_open",
        price_mode="raw",
        dry_run=False,
        state_path=tmp_path / "state.json",
        flatten=False,
        mode="paper",
        confirm=False,
        accept_real_trading="",
        flatten_on_kill=False,
    )
    assert orders == []
    os.environ.pop("KILL_SWITCH", None)


def test_pending_orders_requires_confirm(monkeypatch, tmp_path):
    monkeypatch.setattr("tradinglab.data.fetcher.load_or_fetch_symbols", lambda *a, **k: _mock_history())
    os.environ["LIVE_TRADING_ENABLED"] = "true"
    orders = run_live_cycle(
        universe="small",
        refresh_data=False,
        execution="next_open",
        price_mode="raw",
        dry_run=False,
        state_path=tmp_path / "state.json",
        flatten=False,
        mode="live",
        confirm=False,
        accept_real_trading="YES",
        flatten_on_kill=False,
    )
    pending = list(Path("logs").glob("pending_orders_*.json"))
    assert pending
    os.environ.pop("LIVE_TRADING_ENABLED", None)


def test_safety_checks_block(monkeypatch, tmp_path):
    monkeypatch.setattr("tradinglab.data.fetcher.load_or_fetch_symbols", lambda *a, **k: _mock_history())
    monkeypatch.setattr("app.live.runner.MAX_ORDER_NOTIONAL", 1.0)
    orders = run_live_cycle(
        universe="small",
        refresh_data=False,
        execution="next_open",
        price_mode="raw",
        dry_run=False,
        state_path=tmp_path / "state.json",
        flatten=False,
        mode="paper",
        confirm=False,
        accept_real_trading="",
        flatten_on_kill=False,
    )
    assert orders == []
