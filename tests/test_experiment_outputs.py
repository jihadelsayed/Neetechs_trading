from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tradinglab.experiments import runner as exp_runner


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


def test_experiment_outputs_non_empty(tmp_path, monkeypatch):
    price_dict = {
        "AAA": _make_df("2020-01-01", 1200),
        "BBB": _make_df("2020-01-01", 1200),
        "QQQ": _make_df("2020-01-01", 1200),
    }

    monkeypatch.setattr(exp_runner, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(exp_runner, "load_or_fetch_symbols", lambda *args, **kwargs: price_dict)

    out_dir = exp_runner.run_experiment(
        universe="small",
        symbols=["AAA", "BBB", "QQQ"],
        refresh_data=False,
        start_date="2020-01-01",
        end_date="2023-06-01",
        split_date=None,
        walk_forward=True,
        train_days=200,
        test_days=50,
        step_days=50,
        grid_search=False,
        max_combos=None,
        jobs=1,
        execution="next_open",
        price_mode="adj",
        max_position_weight=None,
        trailing_stop=None,
        time_stop=None,
        target_vol=None,
    )

    assert out_dir.exists()
    config_path = out_dir / "config_used.json"
    equity_path = out_dir / "equity_curves.csv"
    per_split_path = out_dir / "per_split_metrics.csv"
    summary_path = out_dir / "metrics_summary.csv"
    ledger_path = out_dir / "trade_ledger.csv"

    for path in [config_path, equity_path, per_split_path, summary_path, ledger_path]:
        assert path.exists()

    with config_path.open() as f:
        cfg = json.load(f)
    assert cfg["symbols"]

    equity_df = pd.read_csv(equity_path)
    assert not equity_df.empty
    assert "portfolio_value" in equity_df.columns

    per_split_df = pd.read_csv(per_split_path)
    assert not per_split_df.empty
    assert "split_id" in per_split_df.columns

    summary_df = pd.read_csv(summary_path)
    assert not summary_df.empty
    assert "Sharpe_mean" in summary_df.columns
    assert "Sharpe_median" in summary_df.columns

    ledger_df = pd.read_csv(ledger_path)
    assert "symbol" in ledger_df.columns
