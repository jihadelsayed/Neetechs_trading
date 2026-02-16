from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from tradinglab.config import INITIAL_CAPITAL


@dataclass
class LiveState:
    cash: float
    positions: dict[str, float]
    last_rebalance_date: str | None
    last_processed_date: str | None
    holdings_info: dict[str, dict]
    config_snapshot: dict
    in_progress: bool
    last_run_date: str | None
    last_equity: float | None
    strategy_start_equity: float | None

    @staticmethod
    def default(config_snapshot: dict) -> "LiveState":
        return LiveState(
            cash=float(INITIAL_CAPITAL),
            positions={},
            last_rebalance_date=None,
            last_processed_date=None,
            holdings_info={},
            config_snapshot=config_snapshot,
            in_progress=False,
            last_run_date=None,
            last_equity=None,
            strategy_start_equity=None,
        )


def load_state(path: Path, config_snapshot: dict) -> LiveState:
    if not path.exists():
        return LiveState.default(config_snapshot)
    raw = json.loads(path.read_text())
    return LiveState(
        cash=raw.get("cash", float(INITIAL_CAPITAL)),
        positions=raw.get("positions", {}),
        last_rebalance_date=raw.get("last_rebalance_date"),
        last_processed_date=raw.get("last_processed_date"),
        holdings_info=raw.get("holdings_info", {}),
        config_snapshot=raw.get("config_snapshot", config_snapshot),
        in_progress=raw.get("in_progress", False),
        last_run_date=raw.get("last_run_date"),
        last_equity=raw.get("last_equity"),
        strategy_start_equity=raw.get("strategy_start_equity"),
    )


def save_state(path: Path, state: LiveState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), indent=2))
