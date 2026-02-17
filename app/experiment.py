from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    v = value.strip().lower()
    if v in {"1", "true", "yes", "y", "t"}:
        return True
    if v in {"0", "false", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run backtest experiments")
    parser.add_argument(
        "--refresh-data",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help="Refresh cached market data (true/false)",
    )
    parser.add_argument(
        "--universe",
        choices=["small", "nasdaq100"],
        default="nasdaq100",
        help="Universe size",
    )
    parser.add_argument("--start", dest="start_date", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", dest="end_date", help="End date YYYY-MM-DD")
    parser.add_argument("--split-date", dest="split_date", help="Split date YYYY-MM-DD")
    parser.add_argument(
        "--walk-forward",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help="Enable walk-forward splits",
    )
    parser.add_argument("--train-days", type=int, default=504)
    parser.add_argument("--test-days", type=int, default=126)
    parser.add_argument("--step-days", type=int, default=63)
    parser.add_argument("--grid-search", action="store_true", default=False)
    parser.add_argument("--max-combos", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--execution",
        choices=["next_open", "same_close"],
        help="Execution timing",
    )
    parser.add_argument(
        "--price-mode",
        choices=["adj", "raw"],
        help="Price mode",
    )
    parser.add_argument("--max-position-weight", type=float, help="Max position weight")
    parser.add_argument("--trailing-stop", type=float, help="Trailing stop pct")
    parser.add_argument("--time-stop", type=int, help="Max holding days")
    parser.add_argument("--target-vol", type=float, help="Target annualized vol")
    parser.add_argument("--seed", type=int, help="Random seed")
    return parser


def run(argv: list[str] | None = None) -> None:
    _ensure_src_on_path()

    from tradinglab.experiments.runner import run_experiment

    parser = build_parser()
    args = parser.parse_args(argv)

    start_date = args.start_date or "2015-01-01"
    end_date = args.end_date or pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

    if args.universe == "small":
        symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "QQQ"]
    else:
        symbols = None

    run_experiment(
        universe=args.universe,
        symbols=symbols,
        refresh_data=bool(args.refresh_data),
        start_date=start_date,
        end_date=end_date,
        split_date=args.split_date,
        walk_forward=bool(args.walk_forward),
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        grid_search=bool(args.grid_search),
        max_combos=args.max_combos,
        jobs=args.jobs,
        execution=args.execution,
        price_mode=args.price_mode,
        max_position_weight=args.max_position_weight,
        trailing_stop=args.trailing_stop,
        time_stop=args.time_stop,
        target_vol=args.target_vol,
    )


if __name__ == "__main__":
    run()
