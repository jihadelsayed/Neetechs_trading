from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.signal import generate_signal


def _make_df(start: str, periods: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="D")
    values = np.linspace(100.0, 120.0, periods)
    return pd.DataFrame(
        {
            "Open": values,
            "Close": values,
            "Adj Close": values,
        },
        index=idx,
    )


def test_signal_smoke(tmp_path: Path):
    price_dict = {
        "AAPL": _make_df("2022-01-01", 600),
        "MSFT": _make_df("2022-01-01", 600),
        "NVDA": _make_df("2022-01-01", 600),
        "AMZN": _make_df("2022-01-01", 600),
        "QQQ": _make_df("2022-01-01", 600),
    }
    out_path = tmp_path / "live_signal.csv"
    df = generate_signal(
        universe="small",
        top_n=2,
        execution="next_open",
        price_mode="adj",
        refresh_data=False,
        positions_file=None,
        price_dict=price_dict,
        output_path=out_path,
    )
    assert out_path.exists()
    assert not df.empty
    assert "symbol" in df.columns
