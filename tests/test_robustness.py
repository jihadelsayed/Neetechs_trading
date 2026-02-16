from __future__ import annotations

import numpy as np
import pandas as pd

from tradinglab.robustness.monte_carlo import bootstrap_paths
from tradinglab.robustness.rolling import rolling_metrics
from tradinglab.robustness.regimes import regime_metrics
from tradinglab.robustness.allocation import vol_target_exposure


def test_monte_carlo_paths_shape():
    returns = pd.Series([0.01, -0.005, 0.002, 0.0, 0.004])
    paths_df, summary_df = bootstrap_paths(returns, paths=10, seed=42)
    assert len(summary_df) == 10
    assert paths_df.shape[0] == len(returns)
    assert paths_df.shape[1] == 10


def test_rolling_metrics_alignment():
    dates = pd.date_range("2024-01-01", periods=300, freq="D")
    equity = pd.Series(np.linspace(1.0, 2.0, 300), index=dates)
    benchmark = pd.Series(np.linspace(1.0, 1.8, 300), index=dates)
    turnover = pd.Series(0.0, index=dates)
    exposure = pd.Series(1.0, index=dates)

    roll = rolling_metrics(equity, benchmark, turnover, exposure, window=252)
    assert not roll.empty
    assert roll.index[0] == dates[252]


def test_regime_segmentation_logic():
    dates = pd.date_range("2024-01-01", periods=260, freq="D")
    qqq = pd.DataFrame({"Close": np.linspace(100, 150, 260)}, index=dates)
    price_dict = {"QQQ": qqq}
    equity = pd.Series(np.linspace(1.0, 1.2, 260), index=dates)

    reg = regime_metrics(price_dict, equity, regime_symbol="QQQ", price_mode="raw")
    assert "Bull" in reg["Label"].values


def test_allocation_exposure_cap():
    returns = pd.Series([0.01, -0.01, 0.02, -0.005, 0.0, 0.01])
    exposure = vol_target_exposure(returns, target_vol=0.1)
    assert exposure.max() <= 1.0
