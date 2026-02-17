from __future__ import annotations

import numpy as np
import pandas as pd

from tradinglab.data.panel import prepare_panel_with_history_filter
from tradinglab.experiments.splits import walk_forward_splits


def _make_df(start: str, periods: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="D")
    values = np.linspace(100.0, 100.0 + periods - 1, periods)
    return pd.DataFrame(
        {
            "Open": values,
            "Close": values,
            "Adj Close": values,
        },
        index=idx,
    )


def test_panel_filter_drops_short_history():
    price_dict = {
        "AAA": _make_df("2020-01-01", 300),
        "BBB": _make_df("2020-07-01", 100),
        "CCC": _make_df("2020-03-01", 200),
    }

    filtered, panel_close, dropped = prepare_panel_with_history_filter(
        price_dict,
        price_mode="adj",
        min_history_days=180,
    )

    assert "BBB" in dropped
    assert "AAA" in filtered
    assert "CCC" in filtered
    assert not panel_close.empty
    assert panel_close.isna().sum().sum() == 0


def test_walk_forward_splits_non_empty():
    dates = pd.date_range("2020-01-01", periods=1000, freq="D")
    panel_idx = pd.DataFrame(index=dates)
    splits = walk_forward_splits(panel_idx, train_days=200, test_days=50, step_days=50)
    assert len(splits) > 0
    first = splits[0]
    assert first["train_start"] == dates[0]
    assert first["train_end"] == dates[199]
    assert first["test_start"] == dates[200]
