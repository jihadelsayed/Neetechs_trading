from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.brokers.base import MarketDataProvider, Broker, Order, Fill
from tradinglab.config import (
    PRICE_MODE,
    EXECUTION,
    FEE_RATE,
    SLIPPAGE_MODE,
    SLIPPAGE_RATE,
    SLIPPAGE_BPS_BASE,
    SLIPPAGE_BPS_PER_TURNOVER,
    CASH_BUFFER,
    DATA_DIR,
)
from tradinglab.data.fetcher import load_or_fetch_symbols


class CachedMarketDataProvider(MarketDataProvider):
    def __init__(self, price_dict: dict[str, pd.DataFrame] | None = None, price_mode: str = PRICE_MODE):
        self._price_dict = price_dict or {}
        self.price_mode = price_mode
        self.current_date: pd.Timestamp | None = None

    def set_current_date(self, date: pd.Timestamp) -> None:
        self.current_date = pd.to_datetime(date)

    def get_history(self, symbols: list[str], start: str | None, end: str | None) -> dict[str, pd.DataFrame]:
        if self._price_dict:
            out = {}
            for sym, df in self._price_dict.items():
                out[sym] = df.loc[start:end]
            return out
        DATA_DIR.mkdir(exist_ok=True)
        self._price_dict = load_or_fetch_symbols(symbols, refresh=False, start_date=start, end_date=end)
        return self._price_dict

    def get_latest_prices(self, symbols: list[str]) -> pd.DataFrame:
        if self.current_date is None:
            raise RuntimeError("current_date not set")

        if not symbols:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close"])

        rows = []
        for sym in symbols:
            df = self._price_dict.get(sym)
            if df is None or self.current_date not in df.index:
                continue
            row = df.loc[self.current_date].copy()

            if self.price_mode == "adj" and "Adj Close" in df.columns and "Close" in df.columns:
                adj = float(row["Adj Close"])
                close = float(row["Close"]) if float(row["Close"]) != 0 else adj
                factor = adj / close if close != 0 else 1.0
                for col in ["Open", "High", "Low", "Close"]:
                    if col in row.index:
                        row[col] = float(row[col]) * factor
                row["Adj Close"] = adj
            rows.append({"Symbol": sym, **row.to_dict()})

        if not rows:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close"])
        return pd.DataFrame(rows).set_index("Symbol")


class PaperBroker(Broker):
    def __init__(
        self,
        provider: CachedMarketDataProvider,
        cash: float = 0.0,
        positions: dict[str, float] | None = None,
        execution: str = EXECUTION,
        price_mode: str = PRICE_MODE,
        log_dir: Path = Path("logs"),
    ):
        self.provider = provider
        self.cash = float(cash)
        self.positions = positions or {}
        self.execution = execution
        self.price_mode = price_mode
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)

    def is_market_open(self) -> bool:
        return self.provider.current_date is not None

    def get_positions(self) -> dict[str, float]:
        return dict(self.positions)

    def get_account(self) -> dict:
        equity = self.cash
        if self.positions:
            prices = self.provider.get_latest_prices(list(self.positions.keys()))
            for sym, qty in self.positions.items():
                if sym in prices.index:
                    equity += qty * float(prices.loc[sym, "Open"])
        return {"cash": self.cash, "equity": equity, "positions": self.get_positions()}

    def cancel_all(self) -> None:
        return None

    def _slippage_rate(self, notional: float, nav: float) -> float:
        if SLIPPAGE_MODE == "constant":
            return float(SLIPPAGE_RATE)
        base_bps = float(SLIPPAGE_BPS_BASE)
        per_turnover = float(SLIPPAGE_BPS_PER_TURNOVER)
        turnover = 0.0 if nav <= 0 else abs(notional) / nav
        total_bps = base_bps + per_turnover * turnover
        return total_bps / 10000.0

    def place_orders(self, orders: list[Order]) -> list[Fill]:
        if not orders:
            return []

        symbols = list({o.symbol for o in orders})
        prices = self.provider.get_latest_prices(symbols)
        if prices.empty:
            return []

        fills: list[Fill] = []
        account = self.get_account()
        nav = float(account["equity"])

        for order in orders:
            if order.symbol not in prices.index:
                continue
            if order.qty <= 0:
                continue

            mid = float(prices.loc[order.symbol, "Open"])
            notional = order.qty * mid
            slip = self._slippage_rate(notional, nav)
            fee = notional * FEE_RATE
            slippage_cost = notional * slip

            if order.side == "BUY":
                total_cost = notional + fee + slippage_cost
                cash_buffer_amt = max(0.0, CASH_BUFFER * nav)
                available = max(0.0, self.cash - cash_buffer_amt)
                if total_cost > available and total_cost > 0:
                    scale = available / total_cost
                    notional *= scale
                    fee = notional * FEE_RATE
                    slippage_cost = notional * slip
                    total_cost = notional + fee + slippage_cost
                if total_cost <= 0:
                    continue
                qty = notional / mid
                self.cash -= total_cost
                self.positions[order.symbol] = self.positions.get(order.symbol, 0.0) + qty
                fill_price = mid * (1.0 + slip)
            else:
                qty = min(order.qty, self.positions.get(order.symbol, 0.0))
                notional = qty * mid
                fee = notional * FEE_RATE
                slippage_cost = notional * slip
                proceeds = notional - fee - slippage_cost
                self.cash += proceeds
                self.positions[order.symbol] = self.positions.get(order.symbol, 0.0) - qty
                fill_price = mid * (1.0 - slip)

            ts = datetime.utcnow().isoformat()
            fill = Fill(
                symbol=order.symbol,
                side=order.side,
                qty=qty,
                price=fill_price,
                fee=fee,
                slippage_est=slippage_cost,
                timestamp=ts,
            )
            fills.append(fill)
            self._log_trade(order, fill)

        self._log_positions()
        self._log_equity()
        return fills

    def _log_trade(self, order: Order, fill: Fill) -> None:
        path = self.log_dir / "paper_trades.csv"
        row = {
            "timestamp": fill.timestamp,
            "symbol": fill.symbol,
            "side": fill.side,
            "qty": fill.qty,
            "price": fill.price,
            "fee": fill.fee,
            "slippage_est": fill.slippage_est,
            "signal_date": order.signal_date,
        }
        self._append_csv(path, row)

    def _log_positions(self) -> None:
        path = self.log_dir / "paper_positions.csv"
        row = {"timestamp": datetime.utcnow().isoformat(), **self.positions}
        self._append_csv(path, row)

    def _log_equity(self) -> None:
        path = self.log_dir / "paper_equity.csv"
        acct = self.get_account()
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "cash": acct["cash"],
            "equity": acct["equity"],
        }
        self._append_csv(path, row)

    def _append_csv(self, path: Path, row: dict) -> None:
        df = pd.DataFrame([row])
        if path.exists():
            df.to_csv(path, mode="a", index=False, header=False)
        else:
            df.to_csv(path, index=False)
