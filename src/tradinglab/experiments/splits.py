from __future__ import annotations

import pandas as pd


def split_by_date(panel: pd.DataFrame, split_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if panel.empty:
        return panel.copy(), panel.copy()
    split_dt = pd.to_datetime(split_date)
    train = panel.loc[:split_dt]
    test = panel.loc[panel.index > split_dt]
    return train, test


def walk_forward_splits(
    panel: pd.DataFrame,
    train_days: int,
    test_days: int,
    step_days: int,
) -> list[dict]:
    if panel.index.empty:
        return []

    dates = list(panel.index)
    total = len(dates)
    splits: list[dict] = []

    start_idx = 0
    while True:
        train_end = start_idx + train_days - 1
        test_start = train_end + 1
        test_end = test_start + test_days - 1

        if test_end >= total:
            break

        splits.append(
            {
                "train_start": dates[start_idx],
                "train_end": dates[train_end],
                "test_start": dates[test_start],
                "test_end": dates[test_end],
            }
        )
        start_idx += step_days

    return splits
