from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import pandas as pd

from app.brokers.base import Order
from app.brokers.paper import CachedMarketDataProvider, PaperBroker
from app.state import load_state, save_state, LiveState
from app.alerts import send_alert
from app.reporting import write_daily_report
from app.config_validate import validate_config, write_config_snapshot
from tradinglab.config import (
    START_DATE,
    REGIME_SYMBOL,
    EXECUTION,
    PRICE_MODE,
    REBALANCE,
    MAX_TURNOVER_PER_REBALANCE,
    MAX_ORDER_NOTIONAL,
    MAX_ORDERS_PER_RUN,
    MAX_POSITION_WEIGHT,
)
from tradinglab.data.tickers import nasdaq100_tickers
from tradinglab.engine.portfolio import build_price_panels, _weights_from_shares
from tradinglab.engine.portfolio import buy_hold_single


def _compute_targets(
    panel_close: pd.DataFrame,
    date: pd.Timestamp,
    current_weights: dict[str, float],
    price_dict: dict[str, pd.DataFrame],
    price_mode: str,
    top_n: int | None = None,
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
        top_n=top_n,
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


def compute_target_weights(
    panel_close: pd.DataFrame,
    signal_date: pd.Timestamp,
    current_weights: dict[str, float],
    price_dict: dict[str, pd.DataFrame],
    price_mode: str,
    max_turnover: float | None = MAX_TURNOVER_PER_REBALANCE,
    top_n: int | None = None,
) -> tuple[dict[str, float], bool]:
    target_weights = _compute_targets(panel_close, signal_date, current_weights, price_dict, price_mode, top_n=top_n)
    capped = _apply_turnover_cap(current_weights, target_weights, max_turnover)
    return capped, capped != target_weights


def run_live_cycle(
    universe: str,
    refresh_data: bool,
    execution: str,
    price_mode: str,
    dry_run: bool,
    state_path: Path,
    flatten: bool,
    mode: str = "paper",
    confirm: bool = False,
    accept_real_trading: str = "",
    flatten_on_kill: bool = False,
) -> list[Order]:
    end_date = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    if universe == "small":
        symbols = ["AAPL", "MSFT", "NVDA", "AMZN", REGIME_SYMBOL]
    else:
        symbols = nasdaq100_tickers()
        if REGIME_SYMBOL not in symbols:
            symbols.append(REGIME_SYMBOL)

    errors = validate_config(execution, price_mode, REBALANCE)
    if errors:
        raise ValueError("; ".join(errors))

    provider = CachedMarketDataProvider(price_mode=price_mode)
    history = provider.get_history(symbols, start=START_DATE, end=end_date)

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
    write_config_snapshot(Path("logs/state_config.json"), config_snapshot)

    if mode == "live":
        live_enabled = os.getenv("LIVE_TRADING_ENABLED") == "true"
        if accept_real_trading != "YES" or not live_enabled:
            print("Live trading not enabled; refusing to trade.")
            return []
        if confirm:
            from app.brokers.alpaca import AlpacaBroker

            broker = AlpacaBroker()
        else:
            broker = PaperBroker(provider, cash=state.cash, positions=state.positions, execution=execution)
    else:
        broker = PaperBroker(provider, cash=state.cash, positions=state.positions, execution=execution)

    if state.last_processed_date == fill_date.isoformat():
        print("Already processed:", fill_date.date())
        return []

    if state.in_progress and state.last_run_date == fill_date.isoformat():
        print("Detected incomplete run; aborting to avoid duplicates.")
        return []

    if not broker.is_market_open():
        return []

    prices_close = panel_close.loc[signal_date]
    equity = broker.get_account()["equity"]
    if state.strategy_start_equity is None:
        state.strategy_start_equity = equity
    start_equity = equity
    current_weights = _weights_from_shares(prices_close, state.positions, equity)

    risk_triggers: list[str] = []
    rejects: list[str] = []

    if flatten:
        target_weights = {sym: 0.0 for sym in current_weights}
    else:
        target_weights, turnover_capped = compute_target_weights(
            panel_close,
            signal_date,
            current_weights,
            history,
            price_mode,
            max_turnover=MAX_TURNOVER_PER_REBALANCE,
        )
        if turnover_capped:
            risk_triggers.append("turnover_cap_applied")

    if MAX_POSITION_WEIGHT is not None:
        for sym, w in target_weights.items():
            if w > MAX_POSITION_WEIGHT:
                msg = f"Target weight exceeds MAX_POSITION_WEIGHT for {sym}"
                rejects.append(msg)
                print(msg)
                return []

    prices_open = panel_open.loc[fill_date]
    orders = generate_orders_from_weights(list(current_weights.keys()), state.positions, target_weights, prices_open, equity)
    for o in orders:
        o.signal_date = signal_date.isoformat()

    if dry_run:
        for o in orders:
            print(f"DRY-RUN {o.side} {o.qty:.4f} {o.symbol} @ {prices_open[o.symbol]:.2f}")
        return orders

    allowed = set(symbols)
    for o in orders:
        if o.symbol not in allowed:
            msg = f"Order symbol not allowed: {o.symbol}"
            rejects.append(msg)
            print(msg)
            return []

    # pre-flight safety checks
    if len(orders) > MAX_ORDERS_PER_RUN:
        msg = "Order count exceeds MAX_ORDERS_PER_RUN"
        rejects.append(msg)
        print(msg)
        return []

    buy_notional = 0.0
    for o in orders:
        notional = o.qty * float(prices_open[o.symbol])
        if notional > MAX_ORDER_NOTIONAL:
            msg = f"Order too large for {o.symbol}"
            rejects.append(msg)
            print(msg)
            return []
        if o.side == "BUY":
            buy_notional += notional

    if buy_notional > state.cash and state.cash > 0:
        msg = "Buy notional exceeds cash"
        rejects.append(msg)
        print(msg)
        return []

    pending_path = Path("logs") / f"pending_orders_{fill_date.date()}.json"
    pending_path.write_text(json.dumps([o.__dict__ for o in orders], indent=2))

    if mode == "live" and not confirm:
        print(f"Pending orders written to {pending_path}. Use --confirm to place.")
        return orders

    if os.getenv("KILL_SWITCH") == "true":
        send_alert("KILL_SWITCH active: no orders placed.")
        if flatten_on_kill and mode == "live":
            broker.cancel_all()
            positions = broker.get_positions()
            flatten_orders = []
            for sym, qty in positions.items():
                if qty > 0 and sym in prices_open.index:
                    flatten_orders.append(Order(symbol=sym, side="SELL", qty=qty))
            if flatten_orders:
                pending_path.write_text(json.dumps([o.__dict__ for o in flatten_orders], indent=2))
                broker.place_orders(flatten_orders)
        return []

    state.in_progress = True
    state.last_run_date = fill_date.isoformat()
    save_state(state_path, state)

    broker.place_orders(orders)

    acct = broker.get_account()
    state.cash = acct["cash"]
    state.positions = acct["positions"]
    state.last_processed_date = fill_date.isoformat()
    state.last_rebalance_date = signal_date.isoformat()
    state.last_equity = acct["equity"]
    state.in_progress = False

    for sym, qty in state.positions.items():
        if qty > 0:
            info = state.holdings_info.get(sym, {"entry_date": signal_date.isoformat(), "peak": float(prices_close[sym])})
            info["peak"] = max(info.get("peak", float(prices_close[sym])), float(prices_close[sym]))
            state.holdings_info[sym] = info
        else:
            state.holdings_info.pop(sym, None)

    save_state(state_path, state)

    # reporting
    prices_for_report = panel_open.loc[fill_date]
    cash_pct = state.cash / acct["equity"] if acct["equity"] > 0 else 0.0
    gross_exposure = sum(
        abs(qty * float(prices_for_report[sym])) for sym, qty in state.positions.items() if sym in prices_for_report.index
    )
    gross_exposure_pct = gross_exposure / acct["equity"] if acct["equity"] > 0 else 0.0
    turnover = sum(abs(o.qty * float(prices_for_report[o.symbol])) for o in orders) / acct["equity"] if acct["equity"] > 0 else 0.0

    qqq_bh = 0.0
    if REGIME_SYMBOL in history:
        qqq = buy_hold_single(history, REGIME_SYMBOL, price_mode=price_mode)
        if fill_date in qqq.index:
            qqq_bh = float(qqq.loc[fill_date].iloc[0])

    strategy_since_start = acct["equity"] - (state.strategy_start_equity or acct["equity"])

    report_path = Path("logs/reports") / f"{fill_date.date()}_report.md"
    write_daily_report(
        report_path,
        date=str(fill_date.date()),
        start_equity=start_equity,
        end_equity=acct["equity"],
        positions=state.positions,
        prices=prices_for_report,
        cash_pct=cash_pct,
        gross_exposure=gross_exposure_pct,
        turnover=turnover,
        risk_triggers=risk_triggers,
        qqq_bh=qqq_bh,
        strategy_since_start=strategy_since_start,
        broker_summary=broker.get_account() if hasattr(broker, "get_account") else None,
        order_status="placed",
        rejects=rejects,
        pending_path=str(pending_path),
    )

    return orders
