from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def _daily_returns(series: pd.Series) -> pd.Series:
    return series.pct_change().dropna()


def cagr(series: pd.Series) -> float:
    if series is None or series.empty:
        return float("nan")
    start = float(series.iloc[0])
    end = float(series.iloc[-1])
    if start <= 0:
        return float("nan")
    days = (series.index[-1] - series.index[0]).days
    years = days / 365.25
    if years <= 0:
        return float("nan")
    return (end / start) ** (1.0 / years) - 1.0


def annual_vol(series: pd.Series) -> float:
    rets = _daily_returns(series)
    if rets.empty:
        return float("nan")
    return float(rets.std(ddof=0) * np.sqrt(TRADING_DAYS))


def sharpe(series: pd.Series) -> float:
    rets = _daily_returns(series)
    if rets.empty:
        return float("nan")
    vol = rets.std(ddof=0)
    if vol == 0:
        return float("nan")
    return float((rets.mean() / vol) * np.sqrt(TRADING_DAYS))


def sortino(series: pd.Series) -> float:
    rets = _daily_returns(series)
    if rets.empty:
        return float("nan")
    downside = rets[rets < 0]
    if downside.empty:
        return float("nan")
    downside_vol = downside.std(ddof=0)
    if downside_vol == 0:
        return float("nan")
    return float((rets.mean() / downside_vol) * np.sqrt(TRADING_DAYS))


def max_drawdown(series: pd.Series) -> float:
    if series is None or series.empty:
        return float("nan")
    running_max = series.cummax()
    drawdown = (series - running_max) / running_max
    return float(drawdown.min())


def calmar(series: pd.Series) -> float:
    mdd = max_drawdown(series)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return float(cagr(series) / abs(mdd))


def turnover(trade_ledger: pd.DataFrame, equity_series: pd.Series) -> float:
    if trade_ledger is None or trade_ledger.empty:
        return 0.0
    if "order_notional" in trade_ledger.columns:
        notional = trade_ledger["order_notional"].abs().sum()
    else:
        notional = (trade_ledger["shares"].abs() * trade_ledger["fill_price"].abs()).sum()
    avg_equity = float(equity_series.mean()) if equity_series is not None and not equity_series.empty else 0.0
    if avg_equity == 0:
        return float("nan")
    return float(notional / avg_equity)


def trade_win_rate(trade_ledger: pd.DataFrame) -> float:
    if trade_ledger is None or trade_ledger.empty:
        return float("nan")

    wins = 0
    total = 0

    lots: dict[str, deque] = {}

    for _, row in trade_ledger.iterrows():
        symbol = row["symbol"]
        action = row["action"]
        shares = float(row["shares"])
        price = float(row["fill_price"])
        fee = float(row["fees"])
        slippage_cost = float(row.get("slippage_cost", 0.0))

        if symbol not in lots:
            lots[symbol] = deque()

        if action == "BUY":
            if shares <= 0:
                continue
            cost_per_share = (shares * price + fee + slippage_cost) / shares
            lots[symbol].append([shares, cost_per_share])
        elif action == "SELL":
            if shares <= 0:
                continue
            proceeds_per_share = (shares * price - fee - slippage_cost) / shares
            remaining = shares
            pnl = 0.0

            while remaining > 0 and lots[symbol]:
                lot_shares, lot_cost = lots[symbol][0]
                matched = min(remaining, lot_shares)
                pnl += (proceeds_per_share - lot_cost) * matched
                lot_shares -= matched
                remaining -= matched
                if lot_shares <= 0:
                    lots[symbol].popleft()
                else:
                    lots[symbol][0][0] = lot_shares

            if remaining <= 0:
                total += 1
                if pnl > 0:
                    wins += 1

    if total == 0:
        return float("nan")
    return float(wins / total)


def compute_metrics(
    label: str,
    series: pd.Series,
    trade_ledger: pd.DataFrame | None = None,
) -> dict:
    metrics = {
        "Label": label,
        "CAGR": cagr(series),
        "Annual_Vol": annual_vol(series),
        "Sharpe": sharpe(series),
        "Sortino": sortino(series),
        "Max_DD": max_drawdown(series),
        "Calmar": calmar(series),
    }

    if trade_ledger is not None:
        metrics["Turnover"] = turnover(trade_ledger, series)
        metrics["Win_Rate"] = trade_win_rate(trade_ledger)
    else:
        metrics["Turnover"] = 0.0
        metrics["Win_Rate"] = float("nan")

    return metrics
