from __future__ import annotations

from typing import Callable

import pandas as pd

from tradinglab.engine.portfolio import _select_price_series


def filter_price_dict_by_history(
    price_dict: dict[str, pd.DataFrame],
    price_mode: str,
    min_history_days: int,
    keep_symbols: set[str] | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    keep_symbols = keep_symbols or set()
    kept: dict[str, pd.DataFrame] = {}
    dropped: list[str] = []

    for sym, df in price_dict.items():
        if sym in keep_symbols:
            kept[sym] = df
            continue

        _, close_series = _select_price_series(df, price_mode)
        if close_series is None:
            dropped.append(sym)
            continue

        length = int(close_series.dropna().shape[0])
        if length < min_history_days:
            dropped.append(sym)
            continue

        kept[sym] = df

    if log_fn is not None:
        log_fn(f"Dropped {len(dropped)} symbols for insufficient history (<{min_history_days} days).")
    return kept, dropped


def build_close_panel(
    price_dict: dict[str, pd.DataFrame],
    price_mode: str,
) -> pd.DataFrame:
    closes: dict[str, pd.Series] = {}
    for sym, df in price_dict.items():
        _, close_series = _select_price_series(df, price_mode)
        if close_series is None:
            continue
        closes[sym] = close_series

    panel_close = pd.DataFrame(closes).sort_index()
    if panel_close.empty:
        return panel_close

    panel_close = panel_close.ffill()
    panel_close = panel_close.dropna(how="any")
    return panel_close


def prepare_panel_with_history_filter(
    price_dict: dict[str, pd.DataFrame],
    price_mode: str,
    min_history_days: int,
    keep_symbols: set[str] | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, list[str]]:
    keep_symbols = keep_symbols or set()
    filtered, dropped = filter_price_dict_by_history(
        price_dict,
        price_mode=price_mode,
        min_history_days=min_history_days,
        keep_symbols=keep_symbols,
        log_fn=log_fn,
    )
    panel_close = build_close_panel(filtered, price_mode=price_mode)

    if not panel_close.empty and len(panel_close) < min_history_days and len(filtered) > 1:
        def _start_date(df: pd.DataFrame) -> pd.Timestamp | None:
            _, close_series = _select_price_series(df, price_mode)
            if close_series is None:
                return None
            series = close_series.dropna()
            if series.empty:
                return None
            return series.index[0]

        start_dates = {sym: _start_date(df) for sym, df in filtered.items()}
        extra_dropped: list[str] = []

        while len(panel_close) < min_history_days and len(filtered) > 1:
            valid_dates = [dt for dt in start_dates.values() if dt is not None]
            if not valid_dates:
                break
            latest_start = max(valid_dates)
            to_drop = [
                sym
                for sym, dt in start_dates.items()
                if dt == latest_start and sym not in keep_symbols
            ]
            if not to_drop:
                break
            for sym in to_drop:
                filtered.pop(sym, None)
                start_dates.pop(sym, None)
                extra_dropped.append(sym)
            panel_close = build_close_panel(filtered, price_mode=price_mode)
            if panel_close.empty:
                break

        if extra_dropped and log_fn is not None:
            log_fn(f"Dropped {len(extra_dropped)} symbols to meet overlap history requirement.")
        dropped.extend(extra_dropped)

    return filtered, panel_close, dropped
