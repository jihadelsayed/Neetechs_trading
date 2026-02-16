from __future__ import annotations

from pathlib import Path

import pandas as pd

from app import __version__


def write_daily_report(
    path: Path,
    date: str,
    start_equity: float,
    end_equity: float,
    positions: dict[str, float],
    prices: pd.Series,
    cash_pct: float,
    gross_exposure: float,
    turnover: float,
    risk_triggers: list[str],
    qqq_bh: float,
    strategy_since_start: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pnl = end_equity - start_equity
    pnl_pct = (pnl / start_equity) if start_equity > 0 else 0.0

    weights = {}
    for sym, qty in positions.items():
        if sym in prices.index:
            weights[sym] = qty * float(prices[sym])
    total = sum(weights.values()) + max(0.0, end_equity - sum(weights.values()))
    weights = {k: (v / total) for k, v in weights.items()} if total > 0 else {}

    top = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:20]

    lines = []
    lines.append(f"# Daily Report {date}")
    lines.append("")
    lines.append(f"Version: {__version__}")
    lines.append("")
    lines.append("## Equity")
    lines.append(f"- Start equity: {start_equity:.2f}")
    lines.append(f"- End equity: {end_equity:.2f}")
    lines.append(f"- Daily PnL: {pnl:.2f} ({pnl_pct:.2%})")
    lines.append("")
    lines.append("## Exposure")
    lines.append(f"- Cash %: {cash_pct:.2%}")
    lines.append(f"- Gross exposure %: {gross_exposure:.2%}")
    lines.append(f"- Turnover: {turnover:.2%}")
    lines.append("")
    lines.append("## Positions (Top 20 by weight)")
    for sym, w in top:
        lines.append(f"- {sym}: {w:.2%}")
    if not top:
        lines.append("- None")
    lines.append("")
    lines.append("## Risk Triggers")
    if risk_triggers:
        for trigger in risk_triggers:
            lines.append(f"- {trigger}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Comparison Snapshot")
    lines.append(f"- QQQ buy&hold since start: {qqq_bh:.2f}")
    lines.append(f"- Strategy since start: {strategy_since_start:.2f}")

    path.write_text("\n".join(lines), encoding="utf-8")
