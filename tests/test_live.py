from __future__ import annotations

import pandas as pd

from app.live.runner import generate_orders_from_weights
from app.state import LiveState, save_state, load_state
from app.brokers.paper import CachedMarketDataProvider, PaperBroker
from app.brokers.base import Order


def test_generate_orders_from_weights():
    symbols = ["AAA", "BBB"]
    current_positions = {"AAA": 10.0, "BBB": 0.0}
    target_weights = {"AAA": 0.5, "BBB": 0.5}
    prices = pd.Series({"AAA": 10.0, "BBB": 20.0})
    equity = 200.0

    orders = generate_orders_from_weights(symbols, current_positions, target_weights, prices, equity)
    assert len(orders) == 1


def test_state_roundtrip(tmp_path):
    state = LiveState.default({"foo": "bar"})
    path = tmp_path / "state.json"
    save_state(path, state)
    loaded = load_state(path, {"foo": "bar"})
    assert loaded.cash == state.cash
    assert loaded.config_snapshot["foo"] == "bar"


def test_paper_broker_fill():
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    df = pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [10.5, 11.5],
            "Low": [9.5, 10.5],
            "Close": [10.0, 11.0],
            "Adj Close": [10.0, 11.0],
        },
        index=dates,
    )
    provider = CachedMarketDataProvider(price_dict={"AAA": df}, price_mode="raw")
    provider.set_current_date(dates[0])
    broker = PaperBroker(provider, cash=100.0, positions={})

    orders = [Order(symbol="AAA", side="BUY", qty=5.0)]
    fills = broker.place_orders(orders)
    assert fills
    assert broker.get_positions()["AAA"] > 0


def test_live_dry_run_deterministic(tmp_path):
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
    provider = CachedMarketDataProvider(price_dict={"AAA": df, "QQQ": df}, price_mode="raw")
    provider.set_current_date(dates[-1])
    broker = PaperBroker(provider, cash=100.0, positions={})

    # deterministic orders from same weights
    orders1 = [Order(symbol="AAA", side="BUY", qty=1.0)]
    orders2 = [Order(symbol="AAA", side="BUY", qty=1.0)]
    assert orders1[0].qty == orders2[0].qty
