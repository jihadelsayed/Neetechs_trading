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
    parser = argparse.ArgumentParser(description="Run robustness diagnostics")
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
    parser.add_argument(
        "--walk-forward",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help="Enable walk-forward stability analysis",
    )
    parser.add_argument("--train-days", type=int, default=504)
    parser.add_argument("--test-days", type=int, default=126)
    parser.add_argument("--step-days", type=int, default=63)
    parser.add_argument("--monte-carlo", type=int, default=1000, dest="monte_carlo")
    parser.add_argument(
        "--allocation-study",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help="Run allocation policy study",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def run(argv: list[str] | None = None) -> None:
    _ensure_src_on_path()

    from tradinglab.robustness.runner import run_robustness

    parser = build_parser()
    args = parser.parse_args(argv)

    start_date = args.start_date or "2015-01-01"
    end_date = args.end_date or pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

    if args.universe == "small":
        symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "QQQ"]
    else:
        symbols = None

    run_robustness(
        universe=args.universe,
        symbols=symbols,
        refresh_data=bool(args.refresh_data),
        start_date=start_date,
        end_date=end_date,
        walk_forward=bool(args.walk_forward),
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        monte_carlo_paths=args.monte_carlo,
        allocation_study_flag=bool(args.allocation_study),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    run()
