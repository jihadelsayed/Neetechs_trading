from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass
class Order:
    symbol: str
    side: str  # "BUY" or "SELL"
    qty: float
    order_type: str = "market"
    tif: str = "day"
    signal_date: str | None = None


@dataclass
class Fill:
    symbol: str
    side: str
    qty: float
    price: float
    fee: float
    slippage_est: float
    timestamp: str


class MarketDataProvider(Protocol):
    def get_latest_prices(self, symbols: list[str]) -> pd.DataFrame:
        ...

    def get_history(self, symbols: list[str], start: str | None, end: str | None) -> dict[str, pd.DataFrame]:
        ...


class Broker(Protocol):
    def get_account(self) -> dict:
        ...

    def place_orders(self, orders: list[Order]) -> list[Fill]:
        ...

    def get_positions(self) -> dict[str, float]:
        ...

    def cancel_all(self) -> None:
        ...

    def is_market_open(self) -> bool:
        ...
