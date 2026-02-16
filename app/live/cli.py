from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
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
    parser = argparse.ArgumentParser(description="Run live (paper) loop")
    parser.add_argument("--mode", choices=["paper"], default="paper")
    parser.add_argument("--universe", choices=["small", "nasdaq100"], default="nasdaq100")
    parser.add_argument(
        "--dry-run",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
    )
    parser.add_argument(
        "--refresh-data",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
    )
    parser.add_argument("--execution", choices=["next_open", "same_close"], default="next_open")
    parser.add_argument("--price-mode", choices=["adj", "raw"], default="adj")
    parser.add_argument("--state-path", default="logs/state.json")
    parser.add_argument("--flatten", action="store_true")
    parser.add_argument("--run-once", action="store_true", default=True)
    return parser


def run(argv: list[str] | None = None) -> None:
    _ensure_src_on_path()
    from app.live.runner import run_live_cycle

    parser = build_parser()
    args = parser.parse_args(argv)

    state_path = Path(args.state_path)
    run_live_cycle(
        universe=args.universe,
        refresh_data=bool(args.refresh_data),
        execution=args.execution,
        price_mode=args.price_mode,
        dry_run=bool(args.dry_run),
        state_path=state_path,
        flatten=bool(args.flatten),
    )


if __name__ == "__main__":
    run()
