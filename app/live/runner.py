from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from app.brokers.base import Order
from app.brokers.paper import CachedMarketDataProvider, PaperBroker
from app.state import load_state, save_state, LiveState
from tradinglab.config import (
    START_DATE,
    END_DATE,
    REGIME_SYMBOL,
    EXECUTION,
    PRICE_MODE,
    REBALANCE,
    MAX_TURNOVER_PER_REBALANCE,
)
from tradinglab.data.tickers import nasdaq100_tickers
from tradinglab.engine.portfolio import build_price_panels, _weights_from_shares


def _compute_targets(
    panel_close: pd.DataFrame,
    date: pd.Timestamp,
    current_weights: dict[str, float],
    price_dict: dict[str, pd.DataFrame],
    price_mode: str,
) -> dict[str, float]:
    from tradinglab.engine.portfolio import run_portfolio

    sliced = {sym: df.loc[:date] for sym, df in price_dict.items() if date in df.index}
    if not sliced:
        return {sym: 0.0 for sym in current_weights}

    run = run_portfolio(
        sliced,
        execution="same_close",
        price_mode=price_mode,
        slippage_mode="constant",
    )
    if run.positions.empty:
        return {sym: 0.0 for sym in current_weights}

    last_pos = run.positions.iloc[-1]
    last_close = run.panel_close.iloc[-1]
    equity = run.equity["Portfolio_Value"].iloc[-1]

    targets = {}
    for sym in last_pos.index:
        if sym == "Cash":
            continue
        if sym in last_close.index:
            targets[sym] = (last_pos[sym] * last_close[sym]) / equity
    return targets


def generate_orders_from_weights(
    symbols: list[str],
    current_positions: dict[str, float],
    target_weights: dict[str, float],
    prices: pd.Series,
    equity: float,
) -> list[Order]:
    orders = []
    for sym in symbols:
        target_weight = target_weights.get(sym, 0.0)
        current_qty = current_positions.get(sym, 0.0)
        price = float(prices[sym])
        target_qty = (equity * target_weight) / price if price > 0 else 0.0
        delta = target_qty - current_qty
        if abs(delta) <= 0:
            continue
        side = "BUY" if delta > 0 else "SELL"
        orders.append(Order(symbol=sym, side=side, qty=abs(delta)))
    return orders


def _apply_turnover_cap(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    max_turnover: float | None,
) -> dict[str, float]:
    if max_turnover is None:
        return target_weights
    turnover = 0.0
    for sym, tw in target_weights.items():
        turnover += abs(tw - current_weights.get(sym, 0.0))
    if turnover <= max_turnover or turnover <= 0:
        return target_weights
    scale = max_turnover / turnover
    adjusted = {}
    for sym, tw in target_weights.items():
        cw = current_weights.get(sym, 0.0)
        adjusted[sym] = cw + (tw - cw) * scale
    return adjusted


def run_live_cycle(
    universe: str,
    refresh_data: bool,
    execution: str,
    price_mode: str,
    dry_run: bool,
    state_path: Path,
    flatten: bool,
) -> list[Order]:
    if universe == "small":
        symbols = ["AAPL", "MSFT", "NVDA", "AMZN", REGIME_SYMBOL]
    else:
        symbols = nasdaq100_tickers()
        if REGIME_SYMBOL not in symbols:
            symbols.append(REGIME_SYMBOL)

    provider = CachedMarketDataProvider(price_mode=price_mode)
    history = provider.get_history(symbols, start=START_DATE, end=END_DATE)

    panel_close, panel_open = build_price_panels(history, price_mode=price_mode)
    if panel_close.empty:
        return []

    dates = list(panel_close.index)
    if execution == "next_open":
        if len(dates) < 2:
            return []
        signal_date = dates[-2]
        fill_date = dates[-1]
    else:
        signal_date = dates[-1]
        fill_date = dates[-1]

    provider.set_current_date(fill_date)

    config_snapshot = {
        "execution": execution,
        "price_mode": price_mode,
        "rebalance": REBALANCE,
    }
    state = load_state(state_path, config_snapshot)

    broker = PaperBroker(provider, cash=state.cash, positions=state.positions, execution=execution)

    if not broker.is_market_open():
        return []

    prices_close = panel_close.loc[signal_date]
    equity = broker.get_account()["equity"]
    current_weights = _weights_from_shares(prices_close, state.positions, equity)

    if flatten:
        target_weights = {sym: 0.0 for sym in current_weights}
    else:
        target_weights = _compute_targets(panel_close, signal_date, current_weights, history, price_mode)
        target_weights = _apply_turnover_cap(current_weights, target_weights, MAX_TURNOVER_PER_REBALANCE)

    prices_open = panel_open.loc[fill_date]
    orders = generate_orders_from_weights(list(current_weights.keys()), state.positions, target_weights, prices_open, equity)
    for o in orders:
        o.signal_date = signal_date.isoformat()

    if dry_run:
        for o in orders:
            print(f"DRY-RUN {o.side} {o.qty:.4f} {o.symbol} @ {prices_open[o.symbol]:.2f}")
        return orders

    broker.place_orders(orders)

    acct = broker.get_account()
    state.cash = acct["cash"]
    state.positions = acct["positions"]
    state.last_processed_date = fill_date.isoformat()
    state.last_rebalance_date = signal_date.isoformat()

    for sym, qty in state.positions.items():
        if qty > 0:
            info = state.holdings_info.get(sym, {"entry_date": signal_date.isoformat(), "peak": float(prices_close[sym])})
            info["peak"] = max(info.get("peak", float(prices_close[sym])), float(prices_close[sym]))
            state.holdings_info[sym] = info
        else:
            state.holdings_info.pop(sym, None)

    save_state(state_path, state)
    return orders
