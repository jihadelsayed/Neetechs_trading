from __future__ import annotations

import pandas as pd

from tradinglab.engine.portfolio import build_price_panels
from tradinglab.metrics.performance import compute_metrics


def regime_metrics(
    price_dict: dict[str, pd.DataFrame],
    equity: pd.Series,
    regime_symbol: str,
    price_mode: str,
) -> pd.DataFrame:
    panel_close, _ = build_price_panels(price_dict, price_mode=price_mode)
    if panel_close.empty or regime_symbol not in panel_close.columns:
        return pd.DataFrame()

    qqq = panel_close[regime_symbol]
    qqq_sma = qqq.rolling(200).mean()
    qqq_ret = qqq.pct_change()
    vol = qqq_ret.rolling(20).std()
    vol_thresh = float(vol.median())

    regimes = {
        "Bull": qqq > qqq_sma,
        "Bear": qqq < qqq_sma,
        "HighVol": vol >= vol_thresh,
        "LowVol": vol < vol_thresh,
    }

    rows = []
    for name, mask in regimes.items():
        idx = equity.index.intersection(mask.index)
        mask = mask.loc[idx]
        series = equity.loc[idx][mask]
        if series.empty:
            continue
        metrics = compute_metrics(name, series)
        rows.append(metrics)

    return pd.DataFrame(rows)
