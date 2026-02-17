from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Callable

import numpy as np
import pandas as pd

from tradinglab.config import (
    INITIAL_CAPITAL,
    LONG_WINDOW,
    MOM_LOOKBACK,
    TOP_N,
    REBALANCE,
    FEE_RATE,
    SLIPPAGE_RATE,
    SLIPPAGE_MODE,
    SLIPPAGE_BPS_BASE,
    SLIPPAGE_BPS_PER_TURNOVER,
    REGIME_SYMBOL,
    ALLOW_REGIME_TRADE,
    PRICE_MODE,
    EXECUTION,
    MAX_POSITION_WEIGHT,
    MAX_SECTOR_WEIGHT,
    MAX_TURNOVER_PER_REBALANCE,
    MAX_GROSS_EXPOSURE,
    CASH_BUFFER,
    TRAILING_STOP_PCT,
    TIME_STOP_DAYS,
    TARGET_VOL,
    VOL_LOOKBACK,
)

MOM_SKIP_DAYS = 21
MOM_LONG_DAYS = 252
MOM_MID_DAYS = 126
DISPERSION_WINDOW = 252
TURNOVER_BUFFER = 3
WEIGHT_CAP = 0.10
PORTFOLIO_VOL_LOOKBACK = 60


@dataclass
class PortfolioRun:
    equity: pd.DataFrame
    trade_ledger: pd.DataFrame
    positions: pd.DataFrame
    exposures: pd.DataFrame
    panel_close: pd.DataFrame
    panel_open: pd.DataFrame
    tradable_symbols: list[str]


def _select_price_series(df: pd.DataFrame, price_mode: str) -> tuple[pd.Series | None, pd.Series | None]:
    if price_mode == "adj":
        adj_close = df["Adj Close"] if "Adj Close" in df.columns else None
        close = df["Close"] if "Close" in df.columns else None
        open_px = df["Open"] if "Open" in df.columns else None

        if adj_close is None:
            if close is None:
                return None, None
            close_series = close.copy()
            open_series = open_px.copy() if open_px is not None else close.copy()
            return open_series, close_series

        close_series = adj_close.copy()
        if open_px is None or close is None:
            open_series = close_series.copy()
        else:
            adj_factor = adj_close / close.replace(0.0, np.nan)
            open_series = (open_px * adj_factor).copy()
        return open_series, close_series

    if "Close" not in df.columns:
        return None, None
    close_series = df["Close"].copy()
    open_series = df["Open"].copy() if "Open" in df.columns else close_series.copy()
    return open_series, close_series


def build_price_panels(
    price_dict: dict[str, pd.DataFrame],
    price_mode: str = PRICE_MODE,
    min_coverage: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    opens: dict[str, pd.Series] = {}
    closes: dict[str, pd.Series] = {}

    for sym in sorted(price_dict.keys()):
        df = price_dict[sym]
        open_series, close_series = _select_price_series(df, price_mode)
        if close_series is None:
            continue
        opens[sym] = open_series
        closes[sym] = close_series

    panel_close = pd.DataFrame(closes).sort_index()
    if panel_close.empty:
        return panel_close, panel_close

    min_non_na = max(1, int(len(panel_close.columns) * min_coverage))
    panel_close = panel_close.dropna(thresh=min_non_na)
    panel_close = panel_close.ffill()
    panel_close = panel_close.dropna(how="any")

    panel_open = pd.DataFrame(opens).sort_index()
    panel_open = panel_open.reindex(panel_close.index).ffill()
    panel_open = panel_open.dropna(how="any")

    # align close to open in case open dropped more rows
    panel_close = panel_close.loc[panel_open.index]
    return panel_close, panel_open


def build_panel(price_dict: dict[str, pd.DataFrame], min_coverage: float = 0.8) -> pd.DataFrame:
    panel_close, _ = build_price_panels(price_dict, price_mode=PRICE_MODE, min_coverage=min_coverage)
    return panel_close


def blended_momentum_score(
    panel_close: pd.DataFrame,
    skip_days: int = MOM_SKIP_DAYS,
    long_days: int = MOM_LONG_DAYS,
    mid_days: int = MOM_MID_DAYS,
    vol_lookback: int = MOM_MID_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    m12_1 = panel_close.shift(skip_days) / panel_close.shift(long_days) - 1.0
    m6_1 = panel_close.shift(skip_days) / panel_close.shift(mid_days) - 1.0
    m_raw = 0.5 * m12_1 + 0.5 * m6_1
    sigma = panel_close.pct_change().rolling(vol_lookback).std(ddof=0)
    score = m_raw / sigma
    return score, sigma


def dispersion_series(score: pd.DataFrame, window: int = DISPERSION_WINDOW) -> tuple[pd.Series, pd.Series]:
    dispersion = score.std(axis=1)
    dispersion_med = dispersion.rolling(window).median()
    return dispersion, dispersion_med


def inverse_vol_weights(sigmas: pd.Series, cap: float = WEIGHT_CAP) -> dict[str, float]:
    sigmas = sigmas.replace([np.inf, -np.inf], np.nan).dropna()
    sigmas = sigmas[sigmas > 0]
    if sigmas.empty:
        return {}
    inv = 1.0 / sigmas
    weights = inv / inv.sum()
    if cap is not None:
        weights = weights.clip(upper=cap)
    total = weights.sum()
    if total > 0:
        weights = weights / total
    return weights.to_dict()


def select_with_turnover_buffer(
    scores: pd.Series,
    current_holdings: list[str],
    top_n: int,
    buffer: int = TURNOVER_BUFFER,
    exclude_symbols: set[str] | None = None,
) -> list[str]:
    exclude_symbols = exclude_symbols or set()
    scores = scores.dropna()
    scores = scores[~scores.index.isin(exclude_symbols)]
    if scores.empty:
        return []
    ranked = scores.sort_values(ascending=False)
    top = ranked.head(top_n).index.tolist()
    if not current_holdings:
        return top
    threshold = top_n + buffer
    ranks = pd.Series(range(1, len(ranked) + 1), index=ranked.index)
    keep = [sym for sym in current_holdings if sym in ranks.index and ranks[sym] <= threshold]
    merged = list(dict.fromkeys(top + keep))
    return merged


def market_regime_ok(panel_close: pd.DataFrame, regime_symbol: str = REGIME_SYMBOL) -> pd.Series:
    if regime_symbol in panel_close.columns:
        px = panel_close[regime_symbol]
    else:
        px = panel_close.mean(axis=1)
    sma200 = px.rolling(LONG_WINDOW).mean()
    return px > sma200


def _portfolio_value(prices: pd.Series, cash: float, shares: dict[str, float]) -> float:
    value = cash
    for sym, sh in shares.items():
        value += sh * float(prices[sym])
    return value


def _weights_from_shares(prices: pd.Series, shares: dict[str, float], equity: float) -> dict[str, float]:
    if equity <= 0:
        return {sym: 0.0 for sym in shares}
    weights = {}
    for sym, sh in shares.items():
        weights[sym] = (sh * float(prices[sym])) / equity
    return weights


def _apply_sector_cap(
    weights: dict[str, float],
    sector_map: dict[str, str],
    max_sector_weight: float,
) -> dict[str, float]:
    if max_sector_weight is None:
        return weights
    sector_weights: dict[str, float] = {}
    for sym, w in weights.items():
        sector = sector_map.get(sym, "Unknown")
        sector_weights[sector] = sector_weights.get(sector, 0.0) + w

    adjusted = weights.copy()
    for sector, total_w in sector_weights.items():
        if total_w <= max_sector_weight or total_w <= 0:
            continue
        scale = max_sector_weight / total_w
        for sym, w in weights.items():
            if sector_map.get(sym, "Unknown") == sector:
                adjusted[sym] = w * scale
    return adjusted


def _apply_weight_caps(
    weights: dict[str, float],
    max_position_weight: float | None,
    max_gross_exposure: float | None,
    cash_buffer: float | None,
) -> dict[str, float]:
    adjusted = weights.copy()

    if max_position_weight is not None:
        for sym in adjusted:
            adjusted[sym] = min(adjusted[sym], max_position_weight)

    total = sum(max(0.0, w) for w in adjusted.values())

    if max_gross_exposure is not None and total > max_gross_exposure:
        scale = max_gross_exposure / total
        for sym in adjusted:
            adjusted[sym] *= scale
        total = sum(max(0.0, w) for w in adjusted.values())

    if cash_buffer is not None:
        cap = max(0.0, 1.0 - cash_buffer)
        if total > cap and total > 0:
            scale = cap / total
            for sym in adjusted:
                adjusted[sym] *= scale

    return adjusted


def _apply_turnover_cap(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    max_turnover: float | None,
) -> dict[str, float]:
    if max_turnover is None:
        return target_weights

    turnover = 0.0
    for sym, tw in target_weights.items():
        cw = current_weights.get(sym, 0.0)
        turnover += abs(tw - cw)

    if turnover <= max_turnover or turnover <= 0:
        return target_weights

    scale = max_turnover / turnover
    adjusted = {}
    for sym, tw in target_weights.items():
        cw = current_weights.get(sym, 0.0)
        adjusted[sym] = cw + (tw - cw) * scale
    return adjusted


def _vol_scale(series: pd.Series, target_vol: float, lookback: int) -> float:
    if target_vol is None:
        return 1.0
    if series is None or series.empty:
        return 1.0
    window = series.tail(lookback)
    if window.empty:
        return 1.0
    vol = window.std(ddof=0) * np.sqrt(252)
    if vol <= 0 or np.isnan(vol):
        return 1.0
    return min(1.0, float(target_vol) / float(vol))


def _slippage_rate(notional: float, nav: float, slippage_mode: str) -> float:
    if slippage_mode == "constant":
        return float(SLIPPAGE_RATE)

    base_bps = float(SLIPPAGE_BPS_BASE)
    per_turnover = float(SLIPPAGE_BPS_PER_TURNOVER)
    turnover = 0.0 if nav <= 0 else abs(notional) / nav
    total_bps = base_bps + per_turnover * turnover
    return total_bps / 10000.0


def _execute_order(
    order: dict,
    prices: pd.Series,
    cash: float,
    shares: dict[str, float],
    nav: float,
    cash_buffer: float,
    slippage_mode: str,
) -> tuple[float, dict[str, float], dict | None]:
    symbol = order["symbol"]
    signal_date = order["signal_date"]
    fill_date = order["fill_date"]
    order_shares = float(order["shares"])
    action = "BUY" if order_shares > 0 else "SELL"

    if order_shares == 0.0:
        return cash, shares, None

    price_mid = float(prices[symbol])
    nav = max(nav, 0.0)

    if action == "SELL" and abs(order_shares) > shares.get(symbol, 0.0):
        order_shares = -shares.get(symbol, 0.0)
    trade_notional = abs(order_shares) * price_mid
    slip_rate = _slippage_rate(trade_notional, nav, slippage_mode)

    if action == "BUY":
        cash_buffer_amount = max(0.0, cash_buffer * nav)
        available_cash = max(0.0, cash - cash_buffer_amount)

        fee = trade_notional * FEE_RATE
        slippage_cost = trade_notional * slip_rate
        total_cost = trade_notional + fee + slippage_cost

        if total_cost > available_cash and total_cost > 0:
            scale = available_cash / total_cost
            order_shares *= scale
            trade_notional = abs(order_shares) * price_mid
            slip_rate = _slippage_rate(trade_notional, nav, slippage_mode)
            fee = trade_notional * FEE_RATE
            slippage_cost = trade_notional * slip_rate
            total_cost = trade_notional + fee + slippage_cost

        if order_shares <= 0 or total_cost <= 0:
            return cash, shares, None

        fill_price = price_mid * (1.0 + slip_rate)
        cash -= total_cost
        shares[symbol] += order_shares
    else:
        fee = trade_notional * FEE_RATE
        slippage_cost = trade_notional * slip_rate
        proceeds = trade_notional - fee - slippage_cost

        fill_price = price_mid * (1.0 - slip_rate)
        shares[symbol] += order_shares
        cash += proceeds

    equity_after = _portfolio_value(prices, cash, shares)
    ledger_row = {
        "signal_date": signal_date,
        "fill_date": fill_date,
        "symbol": symbol,
        "action": action,
        "shares": abs(order_shares),
        "fill_price": fill_price,
        "order_notional": trade_notional,
        "fees": fee,
        "slippage_cost": slippage_cost,
        "cash_after": cash,
        "equity_after": equity_after,
    }

    return cash, shares, ledger_row


def run_portfolio(
    price_dict: dict[str, pd.DataFrame],
    regime_symbol: str = REGIME_SYMBOL,
    allow_regime_trade: bool = ALLOW_REGIME_TRADE,
    price_mode: str = PRICE_MODE,
    execution: str = EXECUTION,
    top_n: int | None = None,
    mom_lookback: int | None = None,
    rebalance: str | None = None,
    long_window: int | None = None,
    max_position_weight: float | None = MAX_POSITION_WEIGHT,
    max_sector_weight: float | None = MAX_SECTOR_WEIGHT,
    sector_map: dict[str, str] | None = None,
    max_turnover_per_rebalance: float | None = MAX_TURNOVER_PER_REBALANCE,
    max_gross_exposure: float | None = MAX_GROSS_EXPOSURE,
    cash_buffer: float | None = CASH_BUFFER,
    trailing_stop_pct: float | None = TRAILING_STOP_PCT,
    time_stop_days: int | None = TIME_STOP_DAYS,
    target_vol: float | None = TARGET_VOL,
    vol_lookback: int = VOL_LOOKBACK,
    slippage_mode: str = SLIPPAGE_MODE,
    log_fn: Callable[[str], None] | None = None,
) -> PortfolioRun:
    local_top_n = int(top_n) if top_n is not None else TOP_N
    local_mom = int(mom_lookback) if mom_lookback is not None else MOM_LOOKBACK
    local_rebalance = rebalance or REBALANCE
    local_long = int(long_window) if long_window is not None else LONG_WINDOW

    panel_close, panel_open = build_price_panels(price_dict, price_mode=price_mode)
    if panel_close.empty:
        raise ValueError("Price panel is empty. No symbols with usable data.")

    tradable_symbols = list(panel_close.columns)
    panel_close = panel_close[tradable_symbols]
    panel_open = panel_open[tradable_symbols]

    close_values = panel_close.to_numpy()
    open_values = panel_open.to_numpy()
    dates = list(panel_close.index)
    n_days, n_syms = close_values.shape

    regime_ok = market_regime_ok(panel_close, regime_symbol=regime_symbol).to_numpy()
    score_df, sigma_df = blended_momentum_score(panel_close)
    dispersion, dispersion_med = dispersion_series(score_df)
    score = score_df.to_numpy()
    sigma = sigma_df.to_numpy()

    rebal_dates = panel_close.resample(local_rebalance).last().index
    rebal_mask = np.array([d in set(rebal_dates) for d in dates])

    cash = float(INITIAL_CAPITAL)
    shares = np.zeros(n_syms, dtype=float)
    holding_info: dict[int, dict] = {}

    ledger_rows: list[dict] = []
    positions_rows: list[dict] = []
    exposure_rows: list[dict] = []
    value_rows: list[dict] = []
    portfolio_values: list[float] = []

    pending_orders: dict[int, list[tuple[int, float, pd.Timestamp]]] = {}

    def portfolio_value(prices: np.ndarray) -> float:
        return float(cash + np.dot(shares, prices))

    def queue_orders(signal_idx: int, fill_idx: int, order_shares: np.ndarray) -> None:
        orders = []
        for i in range(n_syms):
            sh = float(order_shares[i])
            if abs(sh) <= 0:
                continue
            orders.append((i, sh, dates[signal_idx]))
        if orders:
            pending_orders.setdefault(fill_idx, []).extend(orders)

    def _recent_portfolio_vol() -> float | None:
        if len(portfolio_values) < PORTFOLIO_VOL_LOOKBACK + 1:
            return None
        window = np.array(portfolio_values[-(PORTFOLIO_VOL_LOOKBACK + 1):], dtype=float)
        rets = np.diff(window) / window[:-1]
        if rets.size == 0:
            return None
        vol = np.std(rets, ddof=0) * np.sqrt(252)
        return float(vol)

    def compute_targets(idx: int) -> np.ndarray:
        current_equity = portfolio_value(close_values[idx])
        current_weights = np.zeros(n_syms) if current_equity <= 0 else (shares * close_values[idx]) / current_equity

        if not bool(regime_ok[idx]):
            if regime_symbol in tradable_symbols:
                targets = np.zeros(n_syms)
                targets[tradable_symbols.index(regime_symbol)] = 1.0
                if log_fn is not None:
                    log_fn(f"Regime off on {dates[idx].date()}; holding {regime_symbol}.")
                return targets
            return np.zeros(n_syms)

        if not rebal_mask[idx]:
            return current_weights

        if not np.isfinite(dispersion.iloc[idx]) or not np.isfinite(dispersion_med.iloc[idx]):
            return current_weights
        if dispersion.iloc[idx] <= dispersion_med.iloc[idx]:
            if log_fn is not None:
                log_fn(f"Dispersion filter failed on {dates[idx].date()}; holding {regime_symbol}.")
            if regime_symbol in tradable_symbols:
                targets = np.zeros(n_syms)
                targets[tradable_symbols.index(regime_symbol)] = 1.0
                return targets
            return np.zeros(n_syms)

        score_row = pd.Series(score[idx], index=tradable_symbols)
        sigma_row = pd.Series(sigma[idx], index=tradable_symbols)
        exclude = set()
        if not allow_regime_trade and regime_symbol in tradable_symbols:
            exclude.add(regime_symbol)

        finite_mask = score_row.replace([np.inf, -np.inf], np.nan).notna()
        sigma_mask = sigma_row.replace([np.inf, -np.inf], np.nan).notna() & (sigma_row > 0)
        eligible_mask = finite_mask & sigma_mask & (~score_row.index.isin(exclude))
        eligible_scores = score_row[eligible_mask]
        if eligible_scores.empty:
            if regime_symbol in tradable_symbols:
                targets = np.zeros(n_syms)
                targets[tradable_symbols.index(regime_symbol)] = 1.0
                if log_fn is not None:
                    log_fn(f"No eligible symbols on rebalance date {dates[idx].date()}; holding {regime_symbol}.")
                return targets
            if log_fn is not None:
                log_fn(f"No eligible symbols on rebalance date {dates[idx].date()}; staying in cash.")
            return np.zeros(n_syms)

        current_syms = [tradable_symbols[i] for i in range(n_syms) if shares[i] > 0]
        selected = select_with_turnover_buffer(
            eligible_scores,
            current_syms,
            local_top_n,
            buffer=TURNOVER_BUFFER,
            exclude_symbols=set(),
        )
        if not selected:
            if regime_symbol in tradable_symbols:
                targets = np.zeros(n_syms)
                targets[tradable_symbols.index(regime_symbol)] = 1.0
                if log_fn is not None:
                    log_fn(f"No eligible symbols on rebalance date {dates[idx].date()}; holding {regime_symbol}.")
                return targets
            if log_fn is not None:
                log_fn(f"No eligible symbols on rebalance date {dates[idx].date()}; staying in cash.")
            return np.zeros(n_syms)

        weights_map = inverse_vol_weights(sigma_row[selected], cap=WEIGHT_CAP)
        if not weights_map:
            if log_fn is not None:
                log_fn(f"No eligible symbols on rebalance date {dates[idx].date()}; staying in cash.")
            return np.zeros(n_syms)

        targets = np.zeros(n_syms)
        for sym, w in weights_map.items():
            if sym in tradable_symbols:
                targets[tradable_symbols.index(sym)] = w

        realized_vol = _recent_portfolio_vol()
        if target_vol is not None and realized_vol is not None and realized_vol > 0:
            scale = float(target_vol) / float(realized_vol)
            targets *= scale

        total = targets.sum()
        if total > 1.0 and total > 0:
            targets /= total

        if max_position_weight is not None:
            targets = np.minimum(targets, max_position_weight)

        total = targets.sum()
        if max_gross_exposure is not None and total > max_gross_exposure and total > 0:
            targets *= max_gross_exposure / total
            total = targets.sum()
        if cash_buffer is not None:
            cap = max(0.0, 1.0 - cash_buffer)
            if total > cap and total > 0:
                targets *= cap / total

        turnover = float(np.abs(targets - current_weights).sum())
        if max_turnover_per_rebalance is not None and turnover > max_turnover_per_rebalance and turnover > 0:
            scale = max_turnover_per_rebalance / turnover
            targets = current_weights + (targets - current_weights) * scale
        return targets

    def apply_exit_overrides(idx: int, targets: np.ndarray) -> np.ndarray:
        adjusted = targets.copy()
        for i in range(n_syms):
            if shares[i] <= 0:
                continue
            info = holding_info.get(i)
            if info is None:
                continue
            peak = info["peak"]
            entry_idx = info["entry_idx"]
            holding_days = idx - entry_idx + 1
            exit_due = False
            if trailing_stop_pct is not None:
                if close_values[idx][i] <= peak * (1.0 - trailing_stop_pct):
                    exit_due = True
            if time_stop_days is not None and holding_days >= time_stop_days:
                exit_due = True
            if exit_due:
                adjusted[i] = 0.0
        return adjusted

    def order_shares_from_targets(idx: int, targets: np.ndarray) -> np.ndarray:
        equity = portfolio_value(close_values[idx])
        if equity <= 0:
            return np.zeros(n_syms)
        target_shares = (equity * targets) / close_values[idx]
        return target_shares - shares

    def scale_buys_for_cash(idx: int, order_shares: np.ndarray) -> np.ndarray:
        equity = portfolio_value(close_values[idx])
        cash_buffer_amount = max(0.0, (cash_buffer or 0.0) * equity)
        available_cash = max(0.0, cash - cash_buffer_amount)
        buy_notional = float(np.sum(np.where(order_shares > 0, order_shares * close_values[idx], 0.0)))
        if buy_notional <= 0:
            return order_shares
        if buy_notional <= available_cash:
            return order_shares
        scale = available_cash / buy_notional
        adjusted = order_shares.copy()
        adjusted = np.where(adjusted > 0, adjusted * scale, adjusted)
        return adjusted

    def execute_order(idx: int, sym_idx: int, sh: float, signal_date: pd.Timestamp, fill_date: pd.Timestamp, prices: np.ndarray) -> None:
        nonlocal cash, shares
        if sh == 0.0:
            return
        action = "BUY" if sh > 0 else "SELL"
        price_mid = float(prices[sym_idx])
        nav = float(cash + np.dot(shares, prices))
        trade_notional = abs(sh) * price_mid
        slip_rate = _slippage_rate(trade_notional, nav, slippage_mode)
        if action == "SELL" and abs(sh) > shares[sym_idx]:
            sh = -shares[sym_idx]
            trade_notional = abs(sh) * price_mid
            slip_rate = _slippage_rate(trade_notional, nav, slippage_mode)

        if action == "BUY":
            cash_buffer_amount = max(0.0, (cash_buffer or 0.0) * nav)
            available_cash = max(0.0, cash - cash_buffer_amount)
            fee = trade_notional * FEE_RATE
            slippage_cost = trade_notional * slip_rate
            total_cost = trade_notional + fee + slippage_cost
            if total_cost > available_cash and total_cost > 0:
                scale = available_cash / total_cost
                sh *= scale
                trade_notional = abs(sh) * price_mid
                slip_rate = _slippage_rate(trade_notional, nav, slippage_mode)
                fee = trade_notional * FEE_RATE
                slippage_cost = trade_notional * slip_rate
                total_cost = trade_notional + fee + slippage_cost
            if sh <= 0 or total_cost <= 0:
                return
            fill_price = price_mid * (1.0 + slip_rate)
            cash -= total_cost
            shares[sym_idx] += sh
        else:
            fee = trade_notional * FEE_RATE
            slippage_cost = trade_notional * slip_rate
            proceeds = trade_notional - fee - slippage_cost
            fill_price = price_mid * (1.0 - slip_rate)
            shares[sym_idx] += sh
            cash += proceeds

        equity_after = cash + np.dot(shares, prices)
        ledger_rows.append(
            {
                "signal_date": signal_date,
                "fill_date": fill_date,
                "symbol": tradable_symbols[sym_idx],
                "action": action,
                "shares": abs(sh),
                "fill_price": fill_price,
                "order_notional": trade_notional,
                "fees": fee,
                "slippage_cost": slippage_cost,
                "cash_after": cash,
                "equity_after": equity_after,
            }
        )

    for idx in range(n_days):
        if execution == "next_open" and idx in pending_orders:
            for sym_idx, sh, signal_date in pending_orders[idx]:
                execute_order(idx, sym_idx, sh, signal_date, dates[idx], open_values[idx])
            pending_orders.pop(idx, None)

        if execution == "same_close":
            targets = compute_targets(idx)
            targets = apply_exit_overrides(idx, targets)
            order_shares = order_shares_from_targets(idx, targets)
            order_shares = scale_buys_for_cash(idx, order_shares)
            for sym_idx in range(n_syms):
                sh = float(order_shares[sym_idx])
                if sh != 0:
                    execute_order(idx, sym_idx, sh, dates[idx], dates[idx], close_values[idx])
        else:
            if idx < n_days - 1:
                targets = compute_targets(idx)
                targets = apply_exit_overrides(idx, targets)
                order_shares = order_shares_from_targets(idx, targets)
                order_shares = scale_buys_for_cash(idx, order_shares)
                queue_orders(idx, idx + 1, order_shares)

        for sym_idx in range(n_syms):
            if shares[sym_idx] > 0:
                info = holding_info.get(sym_idx)
                if info is None:
                    holding_info[sym_idx] = {"entry_idx": idx, "peak": float(close_values[idx][sym_idx])}
                else:
                    info["peak"] = max(info["peak"], float(close_values[idx][sym_idx]))
            else:
                holding_info.pop(sym_idx, None)

        equity_close = float(cash + np.dot(shares, close_values[idx]))
        portfolio_values.append(equity_close)
        gross_exposure = float(np.sum(np.abs(shares * close_values[idx])))
        holdings = int(np.sum(shares > 0))

        value_rows.append({"Date": dates[idx], "Portfolio_Value": equity_close})
        positions_rows.append(
            {"Date": dates[idx], "Cash": cash, **{tradable_symbols[i]: shares[i] for i in range(n_syms)}}
        )
        exposure_rows.append(
            {
                "Date": dates[idx],
                "Cash_Pct": cash / equity_close if equity_close > 0 else 0.0,
                "Gross_Exposure_Pct": gross_exposure / equity_close if equity_close > 0 else 0.0,
                "Holdings": holdings,
            }
        )

    equity_df = pd.DataFrame(value_rows).set_index("Date")
    equity_df["Return_%"] = equity_df["Portfolio_Value"].pct_change() * 100.0
    running_max = equity_df["Portfolio_Value"].cummax()
    equity_df["Drawdown_%"] = (equity_df["Portfolio_Value"] - running_max) / running_max * 100.0

    ledger_df = pd.DataFrame(ledger_rows)
    if not ledger_df.empty:
        ledger_df = ledger_df.sort_values(["fill_date", "symbol", "action"]).reset_index(drop=True)

    positions_df = pd.DataFrame(positions_rows).set_index("Date")
    exposures_df = pd.DataFrame(exposure_rows).set_index("Date")

    return PortfolioRun(
        equity=equity_df,
        trade_ledger=ledger_df,
        positions=positions_df,
        exposures=exposures_df,
        panel_close=panel_close,
        panel_open=panel_open,
        tradable_symbols=tradable_symbols,
    )


def buy_hold_benchmark(
    price_dict: dict[str, pd.DataFrame],
    regime_symbol: str = REGIME_SYMBOL,
    allow_regime_trade: bool = ALLOW_REGIME_TRADE,
    price_mode: str = PRICE_MODE,
) -> pd.DataFrame:
    panel_close, _ = build_price_panels(price_dict, price_mode=price_mode)
    if panel_close.empty:
        raise ValueError("Price panel is empty. No symbols with Close data.")

    symbols = [s for s in panel_close.columns if allow_regime_trade or s != regime_symbol]
    if not symbols:
        raise ValueError("No tradable symbols for buy-hold benchmark.")

    init = float(INITIAL_CAPITAL)
    weight = 1.0 / len(symbols)

    first = panel_close.iloc[0]
    shares = {sym: (init * weight) / float(first[sym]) for sym in symbols}

    values = []
    for date in panel_close.index:
        v = 0.0
        for sym in symbols:
            v += shares[sym] * float(panel_close.loc[date, sym])
        values.append({"Date": date, "BuyHold_Value": v})

    out = pd.DataFrame(values).set_index("Date")
    running_max = out["BuyHold_Value"].cummax()
    out["BuyHold_Drawdown_%"] = (out["BuyHold_Value"] - running_max) / running_max * 100.0
    return out


def buy_hold_single(
    price_dict: dict[str, pd.DataFrame],
    symbol: str,
    price_mode: str = PRICE_MODE,
) -> pd.DataFrame:
    if symbol not in price_dict:
        raise ValueError(f"Missing data for benchmark symbol: {symbol}")

    df = price_dict[symbol]
    open_series, close_series = _select_price_series(df, price_mode)
    if close_series is None:
        raise ValueError(f"Close column missing for benchmark symbol: {symbol}")

    series = close_series.dropna()
    if series.empty:
        raise ValueError(f"No usable prices for benchmark symbol: {symbol}")

    init = float(INITIAL_CAPITAL)
    shares = init / float(series.iloc[0])

    values = (series * shares).to_frame(name=f"{symbol}_BuyHold")
    return values
