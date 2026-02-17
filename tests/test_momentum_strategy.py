from __future__ import annotations

import numpy as np
import pandas as pd

from tradinglab.engine.portfolio import (
    blended_momentum_score,
    inverse_vol_weights,
    dispersion_series,
    select_with_turnover_buffer,
)


def test_blended_score_calculation():
    idx = pd.date_range("2020-01-01", periods=300, freq="D")
    prices = pd.Series(np.linspace(100.0, 200.0, 300), index=idx)
    panel = pd.DataFrame({"AAA": prices})

    score, sigma = blended_momentum_score(panel)
    t = idx[260]
    m12_1 = panel.loc[t - pd.Timedelta(days=21), "AAA"] / panel.loc[t - pd.Timedelta(days=252), "AAA"] - 1.0
    m6_1 = panel.loc[t - pd.Timedelta(days=21), "AAA"] / panel.loc[t - pd.Timedelta(days=126), "AAA"] - 1.0
    m_raw = 0.5 * m12_1 + 0.5 * m6_1
    sigma_t = panel["AAA"].pct_change().rolling(126).std(ddof=0).loc[t]
    score_t = score.loc[t, "AAA"]
    assert np.isclose(score_t, m_raw / sigma_t, rtol=1e-6, atol=1e-6)


def test_inverse_vol_weights_with_cap():
    sigmas = pd.Series({"A": 0.01, "B": 0.5, "C": 0.5})
    weights = inverse_vol_weights(sigmas, cap=0.10)
    inv = 1.0 / sigmas
    expected = inv / inv.sum()
    expected = expected.clip(upper=0.10)
    expected = expected / expected.sum()
    for k in weights:
        assert np.isclose(weights[k], expected[k], rtol=1e-6, atol=1e-6)


def test_dispersion_filter_behavior():
    idx = pd.date_range("2020-01-01", periods=300, freq="D")
    score = pd.DataFrame(
        {
            "A": np.ones(300),
            "B": np.ones(300),
            "C": np.ones(300),
        },
        index=idx,
    )
    dispersion, med = dispersion_series(score)
    assert (dispersion <= med).iloc[-1]


def test_turnover_buffer_behavior():
    scores = pd.Series({"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0, "E": 1.0})
    current = ["D"]
    selected = select_with_turnover_buffer(scores, current, top_n=2, buffer=3)
    assert "D" in selected
