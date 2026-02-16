from __future__ import annotations

import argparse
import sys
from pathlib import Path


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
    parser = argparse.ArgumentParser(description="Run backtest")
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
    parser.add_argument("--top-n", dest="top_n", type=int, help="Top N holdings")
    parser.add_argument(
        "--rebalance",
        choices=["ME", "W-FRI"],
        help="Rebalance frequency",
    )
    return parser


def run(argv: list[str] | None = None) -> None:
    _ensure_src_on_path()

    from tradinglab.cli.main import main

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.universe == "small":
        symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "QQQ"]
    else:
        symbols = None

    main(
        refresh_data=bool(args.refresh_data),
        run_per_symbol=False,
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        top_n=args.top_n,
        rebalance=args.rebalance,
    )


if __name__ == "__main__":
    run()
