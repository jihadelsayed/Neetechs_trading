from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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
    REGIME_SYMBOL,
    ALLOW_REGIME_TRADE,
)


@dataclass
class PortfolioRun:
    equity: pd.DataFrame
    trade_ledger: pd.DataFrame
    positions: pd.DataFrame
    panel: pd.DataFrame
    tradable_symbols: list[str]


def build_panel(price_dict: dict[str, pd.DataFrame], min_coverage: float = 0.8) -> pd.DataFrame:
    closes: dict[str, pd.Series] = {}
    for sym in sorted(price_dict.keys()):
        df = price_dict[sym]
        if "Close" not in df.columns:
            continue
        closes[sym] = df["Close"].copy()

    panel = pd.DataFrame(closes).sort_index()
    if panel.empty:
        return panel

    min_non_na = max(1, int(len(panel.columns) * min_coverage))
    panel = panel.dropna(thresh=min_non_na)
    panel = panel.ffill()
    panel = panel.dropna(how="any")
    return panel


def momentum_scores(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.pct_change(MOM_LOOKBACK)


def trend_filter(panel: pd.DataFrame) -> pd.DataFrame:
    sma = panel.rolling(LONG_WINDOW).mean()
    return panel > sma


def market_regime_ok(panel: pd.DataFrame, regime_symbol: str = REGIME_SYMBOL) -> pd.Series:
    if regime_symbol in panel.columns:
        px = panel[regime_symbol]
    else:
        px = panel.mean(axis=1)
    sma200 = px.rolling(LONG_WINDOW).mean()
    return px > sma200


def _portfolio_value(prices: pd.Series, cash: float, shares: dict[str, float]) -> float:
    value = cash
    for sym, sh in shares.items():
        value += sh * float(prices[sym])
    return value


def run_portfolio(
    price_dict: dict[str, pd.DataFrame],
    regime_symbol: str = REGIME_SYMBOL,
    allow_regime_trade: bool = ALLOW_REGIME_TRADE,
) -> PortfolioRun:
    panel = build_panel(price_dict)
    if panel.empty:
        raise ValueError("Price panel is empty. No symbols with Close data.")

    regime_ok = market_regime_ok(panel, regime_symbol=regime_symbol)
    mom = momentum_scores(panel)
    trend_ok = trend_filter(panel)

    rebal_dates = panel.resample(REBALANCE).last().index
    rebal_dates = set([d for d in rebal_dates if d in panel.index])

    tradable_symbols = [s for s in panel.columns if allow_regime_trade or s != regime_symbol]

    cash = float(INITIAL_CAPITAL)
    shares = {s: 0.0 for s in tradable_symbols}

    ledger_rows: list[dict] = []
    positions_rows: list[dict] = []
    value_rows: list[dict] = []

    for date in panel.index:
        prices = panel.loc[date]

        if not bool(regime_ok.loc[date]):
            # liquidate all positions
            for sym in sorted(tradable_symbols):
                if shares[sym] > 0:
                    exec_price = float(prices[sym]) * (1.0 - SLIPPAGE_RATE)
                    fee = (shares[sym] * exec_price) * FEE_RATE
                    proceeds = (shares[sym] * exec_price) - fee
                    cash += proceeds

                    shares_sold = shares[sym]
                    shares[sym] = 0.0

                    equity_after = _portfolio_value(prices, cash, shares)
                    ledger_rows.append(
                        {
                            "date": date,
                            "symbol": sym,
                            "action": "SELL",
                            "shares": shares_sold,
                            "price": exec_price,
                            "fee": fee,
                            "slippage": SLIPPAGE_RATE,
                            "cash_after": cash,
                            "equity_after": equity_after,
                        }
                    )

            value_rows.append({"Date": date, "Portfolio_Value": _portfolio_value(prices, cash, shares)})
            positions_rows.append({"Date": date, "Cash": cash, **shares})
            continue

        if date in rebal_dates:
            score_row = mom.loc[date].copy()
            ok_row = trend_ok.loc[date].copy()

            score_row = score_row.loc[tradable_symbols]
            ok_row = ok_row.loc[tradable_symbols]

            score_row = score_row.where(ok_row).dropna()
            top = score_row.sort_values(ascending=False).head(TOP_N).index.tolist()

            targets = {sym: (1.0 / len(top) if sym in top and len(top) > 0 else 0.0) for sym in tradable_symbols}

            # sell non-targets
            for sym in sorted(tradable_symbols):
                if targets[sym] == 0.0 and shares[sym] > 0:
                    exec_price = float(prices[sym]) * (1.0 - SLIPPAGE_RATE)
                    fee = (shares[sym] * exec_price) * FEE_RATE
                    proceeds = (shares[sym] * exec_price) - fee
                    cash += proceeds

                    shares_sold = shares[sym]
                    shares[sym] = 0.0

                    equity_after = _portfolio_value(prices, cash, shares)
                    ledger_rows.append(
                        {
                            "date": date,
                            "symbol": sym,
                            "action": "SELL",
                            "shares": shares_sold,
                            "price": exec_price,
                            "fee": fee,
                            "slippage": SLIPPAGE_RATE,
                            "cash_after": cash,
                            "equity_after": equity_after,
                        }
                    )

            equity = _portfolio_value(prices, cash, shares)

            for sym in sorted(tradable_symbols):
                target_weight = targets[sym]
                if target_weight <= 0.0:
                    continue
                target_value = equity * target_weight
                current_value = shares[sym] * float(prices[sym])
                diff = target_value - current_value

                if diff > 0 and cash > 0:
                    exec_price = float(prices[sym]) * (1.0 + SLIPPAGE_RATE)
                    spend = min(cash, diff)
                    fee = spend * FEE_RATE
                    buy_shares = (spend - fee) / exec_price

                    if buy_shares > 0:
                        shares[sym] += buy_shares
                        cash -= spend

                        equity_after = _portfolio_value(prices, cash, shares)
                        ledger_rows.append(
                            {
                                "date": date,
                                "symbol": sym,
                                "action": "BUY",
                                "shares": buy_shares,
                                "price": exec_price,
                                "fee": fee,
                                "slippage": SLIPPAGE_RATE,
                                "cash_after": cash,
                                "equity_after": equity_after,
                            }
                        )

        value_rows.append({"Date": date, "Portfolio_Value": _portfolio_value(prices, cash, shares)})
        positions_rows.append({"Date": date, "Cash": cash, **shares})

    equity_df = pd.DataFrame(value_rows).set_index("Date")
    equity_df["Return_%"] = equity_df["Portfolio_Value"].pct_change() * 100.0
    running_max = equity_df["Portfolio_Value"].cummax()
    equity_df["Drawdown_%"] = (equity_df["Portfolio_Value"] - running_max) / running_max * 100.0

    ledger_df = pd.DataFrame(ledger_rows)
    if not ledger_df.empty:
        ledger_df = ledger_df.sort_values(["date", "symbol", "action"]).reset_index(drop=True)

    positions_df = pd.DataFrame(positions_rows).set_index("Date")

    return PortfolioRun(
        equity=equity_df,
        trade_ledger=ledger_df,
        positions=positions_df,
        panel=panel,
        tradable_symbols=tradable_symbols,
    )


def buy_hold_benchmark(
    price_dict: dict[str, pd.DataFrame],
    regime_symbol: str = REGIME_SYMBOL,
    allow_regime_trade: bool = ALLOW_REGIME_TRADE,
) -> pd.DataFrame:
    panel = build_panel(price_dict)
    if panel.empty:
        raise ValueError("Price panel is empty. No symbols with Close data.")

    symbols = [s for s in panel.columns if allow_regime_trade or s != regime_symbol]
    if not symbols:
        raise ValueError("No tradable symbols for buy-hold benchmark.")

    init = float(INITIAL_CAPITAL)
    weight = 1.0 / len(symbols)

    first = panel.iloc[0]
    shares = {sym: (init * weight) / float(first[sym]) for sym in symbols}

    values = []
    for date in panel.index:
        v = 0.0
        for sym in symbols:
            v += shares[sym] * float(panel.loc[date, sym])
        values.append({"Date": date, "BuyHold_Value": v})

    out = pd.DataFrame(values).set_index("Date")
    running_max = out["BuyHold_Value"].cummax()
    out["BuyHold_Drawdown_%"] = (out["BuyHold_Value"] - running_max) / running_max * 100.0
    return out


def buy_hold_single(price_dict: dict[str, pd.DataFrame], symbol: str) -> pd.DataFrame:
    if symbol not in price_dict:
        raise ValueError(f"Missing data for benchmark symbol: {symbol}")

    df = price_dict[symbol]
    if "Close" not in df.columns:
        raise ValueError(f"Close column missing for benchmark symbol: {symbol}")

    series = df["Close"].copy()
    series = series.dropna()
    if series.empty:
        raise ValueError(f"No usable prices for benchmark symbol: {symbol}")

    init = float(INITIAL_CAPITAL)
    shares = init / float(series.iloc[0])

    values = (series * shares).to_frame(name=f"{symbol}_BuyHold")
    return values
