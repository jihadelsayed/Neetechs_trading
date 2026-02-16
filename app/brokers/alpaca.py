from __future__ import annotations

import os
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any

import requests
import pandas as pd

from app.brokers.base import Broker, Order, Fill


class AlpacaBroker(Broker):
    def __init__(self) -> None:
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.api_secret = os.getenv("ALPACA_API_SECRET")
        self.base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        if not self.api_key or not self.api_secret:
            raise RuntimeError("Missing Alpaca credentials")

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }

    def _request(self, method: str, path: str, json_body: dict | None = None) -> Any:
        url = f"{self.base_url}{path}"
        for attempt in range(5):
            try:
                resp = requests.request(method, url, headers=self._headers(), json=json_body, timeout=30)
                if resp.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                if attempt == 4:
                    raise RuntimeError(f"Alpaca API error: {exc}") from exc
                time.sleep(2 ** attempt)
        raise RuntimeError("Alpaca API failed")

    def get_account(self) -> dict:
        data = self._request("GET", "/v2/account")
        return {
            "cash": float(data.get("cash", 0.0)),
            "equity": float(data.get("equity", 0.0)),
        }

    def get_positions(self) -> dict[str, float]:
        data = self._request("GET", "/v2/positions")
        out = {}
        for pos in data:
            out[pos["symbol"]] = float(pos["qty"])
        return out

    def get_latest_prices(self, symbols: list[str]) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close"])
        joined = ",".join(symbols)
        data = self._request("GET", f"/v2/stocks/quotes/latest?symbols={joined}")
        rows = []
        for sym, quote in data.get("quotes", {}).items():
            rows.append(
                {
                    "Symbol": sym,
                    "Open": quote.get("ap"),
                    "High": quote.get("ap"),
                    "Low": quote.get("bp"),
                    "Close": quote.get("ap"),
                    "Adj Close": quote.get("ap"),
                }
            )
        return pd.DataFrame(rows).set_index("Symbol")

    def place_orders(self, orders: list[Order]) -> list[Fill]:
        fills: list[Fill] = []
        for order in orders:
            body = {
                "symbol": order.symbol,
                "qty": str(order.qty),
                "side": order.side.lower(),
                "type": order.order_type,
                "time_in_force": order.tif,
            }
            data = self._request("POST", "/v2/orders", json_body=body)
            fills.append(
                Fill(
                    symbol=order.symbol,
                    side=order.side,
                    qty=order.qty,
                    price=float(data.get("filled_avg_price") or 0.0),
                    fee=0.0,
                    slippage_est=0.0,
                    timestamp=datetime.utcnow().isoformat(),
                )
            )
        return fills

    def cancel_all(self) -> None:
        self._request("DELETE", "/v2/orders")

    def is_market_open(self) -> bool:
        data = self._request("GET", "/v2/clock")
        return bool(data.get("is_open", False))
