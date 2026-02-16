from __future__ import annotations

import json
from pathlib import Path

from tradinglab.config import EXECUTION, PRICE_MODE, REBALANCE


ALLOWED_EXECUTION = {"next_open", "same_close"}
ALLOWED_PRICE_MODE = {"adj", "raw"}
ALLOWED_REBALANCE = {"ME", "W-FRI"}


def validate_config(execution: str, price_mode: str, rebalance: str) -> list[str]:
    errors = []
    if execution not in ALLOWED_EXECUTION:
        errors.append(f"Invalid execution: {execution}")
    if price_mode not in ALLOWED_PRICE_MODE:
        errors.append(f"Invalid price_mode: {price_mode}")
    if rebalance not in ALLOWED_REBALANCE:
        errors.append(f"Invalid rebalance: {rebalance}")
    return errors


def write_config_snapshot(path: Path, snapshot: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2))
