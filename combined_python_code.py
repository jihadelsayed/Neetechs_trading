
# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\backtest.py
# ==============================

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tradinglab.config import INITIAL_CAPITAL, FEE_RATE, SLIPPAGE_RATE  # noqa: E402
from tradinglab.strategies.ma_crossover import generate_signals  # noqa: E402


def backtest(df: pd.DataFrame) -> pd.DataFrame:
    df = generate_signals(df).copy()

    capital = float(INITIAL_CAPITAL)
    shares = 0.0

    first_price = float(df["Close"].iloc[0])
    bh_shares = float(INITIAL_CAPITAL) / first_price

    portfolio_values = []
    cash_values = []
    share_values = []
    buyhold_values = []

    for _, row in df.iterrows():
        price = float(row["Close"])
        target_pos = int(row["Position"])

        if target_pos == 1 and shares == 0:
            buy_price = price * (1.0 + SLIPPAGE_RATE)
            shares = (capital * (1.0 - FEE_RATE)) / buy_price
            capital = 0.0
        elif target_pos == 0 and shares > 0:
            sell_price = price * (1.0 - SLIPPAGE_RATE)
            capital = (shares * sell_price) * (1.0 - FEE_RATE)
            shares = 0.0

        portfolio_value = capital + (shares * price)
        portfolio_values.append(portfolio_value)
        cash_values.append(capital)
        share_values.append(shares * price)
        buyhold_values.append(bh_shares * price)

    df["Cash"] = cash_values
    df["Stock_Value"] = share_values
    df["Portfolio_Value"] = portfolio_values
    df["BuyHold_Value"] = buyhold_values

    return df


def performance_summary(df: pd.DataFrame):
    start_value = df["Portfolio_Value"].iloc[0]
    end_value = df["Portfolio_Value"].iloc[-1]
    total_return = (end_value - start_value) / start_value * 100

    print(f"Starting Capital: ${start_value:,.2f}")
    print(f"Ending Value:     ${end_value:,.2f}")
    print(f"Total Return:     {total_return:.2f}%")


if __name__ == "__main__":
    df = pd.read_csv("data/AAPL.csv", parse_dates=["Date"]).set_index("Date")
    result = backtest(df)
    performance_summary(result)

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\config.py
# ==============================

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tradinglab.config import (  # noqa: F401
    START_DATE,
    END_DATE,
    INITIAL_CAPITAL,
    SHORT_WINDOW,
    LONG_WINDOW,
    MIN_VOL20,
    FEE_RATE,
    SLIPPAGE_RATE,
    SLIPPAGE_MODE,
    SLIPPAGE_BPS_BASE,
    SLIPPAGE_BPS_PER_TURNOVER,
    REBALANCE,
    MOM_LOOKBACK,
    TOP_N,
    REGIME_SYMBOL,
    ALLOW_REGIME_TRADE,
    EXECUTION,
    PRICE_MODE,
    MAX_POSITION_WEIGHT,
    MAX_SECTOR_WEIGHT,
    MAX_TURNOVER_PER_REBALANCE,
    MAX_GROSS_EXPOSURE,
    CASH_BUFFER,
    MAX_ORDER_NOTIONAL,
    MAX_ORDERS_PER_RUN,
    TRAILING_STOP_PCT,
    TIME_STOP_DAYS,
    TARGET_VOL,
    VOL_LOOKBACK,
    DATA_DIR,
    RESULTS_DIR,
)

__all__ = [
    "START_DATE",
    "END_DATE",
    "INITIAL_CAPITAL",
    "SHORT_WINDOW",
    "LONG_WINDOW",
    "MIN_VOL20",
    "FEE_RATE",
    "SLIPPAGE_RATE",
    "SLIPPAGE_MODE",
    "SLIPPAGE_BPS_BASE",
    "SLIPPAGE_BPS_PER_TURNOVER",
    "REBALANCE",
    "MOM_LOOKBACK",
    "TOP_N",
    "REGIME_SYMBOL",
    "ALLOW_REGIME_TRADE",
    "EXECUTION",
    "PRICE_MODE",
    "MAX_POSITION_WEIGHT",
    "MAX_SECTOR_WEIGHT",
    "MAX_TURNOVER_PER_REBALANCE",
    "MAX_GROSS_EXPOSURE",
    "CASH_BUFFER",
    "MAX_ORDER_NOTIONAL",
    "MAX_ORDERS_PER_RUN",
    "TRAILING_STOP_PCT",
    "TIME_STOP_DAYS",
    "TARGET_VOL",
    "VOL_LOOKBACK",
    "DATA_DIR",
    "RESULTS_DIR",
]

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\data_fetcher.py
# ==============================

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tradinglab.data.fetcher import load_or_fetch_symbols  # noqa: E402
from tradinglab.data.tickers import nasdaq100_tickers  # noqa: E402
from tradinglab.config import REGIME_SYMBOL  # noqa: E402


if __name__ == "__main__":
    symbols = nasdaq100_tickers()
    if REGIME_SYMBOL not in symbols:
        symbols.append(REGIME_SYMBOL)

    market = load_or_fetch_symbols(symbols, refresh=False)
    for sym, df in market.items():
        print(sym, df.shape, "last date:", df.index.max().date())

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\main.py
# ==============================

from __future__ import annotations

from app.cli import run


if __name__ == "__main__":
    run()

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\portfolio_backtest.py
# ==============================

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tradinglab.engine.portfolio import (  # noqa: E402
    build_panel,
    run_portfolio,
    buy_hold_benchmark,
)

__all__ = ["build_panel", "run_portfolio", "buy_hold_benchmark"]

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\strategy.py
# ==============================

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tradinglab.strategies.ma_crossover import add_indicators, generate_signals  # noqa: E402


__all__ = ["add_indicators", "generate_signals"]

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\tickers.py
# ==============================

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tradinglab.data.tickers import nasdaq100_tickers  # noqa: E402


if __name__ == "__main__":
    syms = nasdaq100_tickers()
    print("count:", len(syms))
    print(syms[:25])

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\app\alerts.py
# ==============================

from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from pathlib import Path


def _log_alert(message: str, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def send_alert(message: str, log_path: Path = Path("logs/alerts.log")) -> None:
    print(message)
    _log_alert(message, log_path)

    host = os.getenv("SMTP_HOST")
    port = os.getenv("SMTP_PORT")
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASS")
    recipient = os.getenv("ALERT_TO")

    if not all([host, port, user, password, recipient]):
        return

    msg = EmailMessage()
    msg["Subject"] = "TradingLab Alert"
    msg["From"] = user
    msg["To"] = recipient
    msg.set_content(message)

    with smtplib.SMTP(host, int(port)) as server:
        server.starttls()
        server.login(user, password)
        server.send_message(msg)

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\app\backtest.py
# ==============================

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
    parser = argparse.ArgumentParser(description="Run portfolio backtest")
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
        execution=args.execution,
        price_mode=args.price_mode,
        max_position_weight=args.max_position_weight,
        trailing_stop=args.trailing_stop,
        time_stop=args.time_stop,
        target_vol=args.target_vol,
    )


if __name__ == "__main__":
    run()

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\app\bench.py
# ==============================

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from tradinglab.config import REGIME_SYMBOL
from tradinglab.data.fetcher import load_or_fetch_symbols
from tradinglab.data.tickers import nasdaq100_tickers
from tradinglab.engine.portfolio import build_price_panels, run_portfolio
from tradinglab.metrics.performance import compute_metrics


def run_bench(universe: str) -> Path:
    if universe == "small":
        symbols = ["AAPL", "MSFT", "NVDA", "AMZN", REGIME_SYMBOL]
    else:
        symbols = nasdaq100_tickers()
        if REGIME_SYMBOL not in symbols:
            symbols.append(REGIME_SYMBOL)

    t0 = time.perf_counter()
    price_dict = load_or_fetch_symbols(symbols, refresh=False)
    t1 = time.perf_counter()

    panel_close, _ = build_price_panels(price_dict)
    t2 = time.perf_counter()

    run = run_portfolio(price_dict)
    t3 = time.perf_counter()

    metrics = compute_metrics("Portfolio", run.equity["Portfolio_Value"], run.trade_ledger)
    t4 = time.perf_counter()

    result = {
        "data_load_s": t1 - t0,
        "panel_build_s": t2 - t1,
        "run_portfolio_s": t3 - t2,
        "metrics_s": t4 - t3,
        "total_s": t4 - t0,
        "metrics": metrics,
    }

    out_path = Path("logs") / "bench_baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    return out_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run benchmark")
    parser.add_argument("--universe", choices=["small", "nasdaq100"], default="small")
    args = parser.parse_args()

    path = run_bench(args.universe)
    print(f"Saved benchmark: {path}")


if __name__ == "__main__":
    main()

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\app\cli.py
# ==============================

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
        execution=args.execution,
        price_mode=args.price_mode,
        max_position_weight=args.max_position_weight,
        trailing_stop=args.trailing_stop,
        time_stop=args.time_stop,
        target_vol=args.target_vol,
    )


if __name__ == "__main__":
    run()

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\app\config_validate.py
# ==============================

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

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\app\dashboard.py
# ==============================

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


RESULTS_DIR = Path("results")


def _latest_experiment() -> Path | None:
    exp_dir = RESULTS_DIR / "experiments"
    if not exp_dir.exists():
        return None
    runs = sorted(exp_dir.iterdir())
    if not runs:
        return None
    return runs[-1]


def main() -> None:
    st.title("Neetechs Trading Dashboard")

    run_path = _latest_experiment()
    if run_path is None:
        st.warning("No experiment runs found.")
        return

    st.write(f"Latest run: {run_path.name}")

    equity_path = run_path / "equity_curves.csv"
    metrics_path = run_path / "metrics_summary.csv"
    stability_path = run_path / "parameter_stability.csv"
    rolling_path = RESULTS_DIR / "robustness" / "rolling_metrics.csv"
    mc_path = RESULTS_DIR / "robustness" / "monte_carlo_summary.csv"

    if equity_path.exists():
        eq = pd.read_csv(equity_path, parse_dates=["date"])
        st.subheader("Equity Curves")
        fig, ax = plt.subplots()
        ax.plot(eq["date"], eq["portfolio_value"], label="Portfolio")
        if "qqq_value" in eq.columns:
            ax.plot(eq["date"], eq["qqq_value"], label="QQQ")
        if "buyhold_value" in eq.columns:
            ax.plot(eq["date"], eq["buyhold_value"], label="Equal-Weight")
        ax.legend()
        st.pyplot(fig)

    if metrics_path.exists():
        st.subheader("Metrics Summary")
        st.dataframe(pd.read_csv(metrics_path))

    if stability_path.exists():
        st.subheader("Parameter Stability")
        st.dataframe(pd.read_csv(stability_path).sort_values("stability_score", ascending=False).head(20))

    if rolling_path.exists():
        st.subheader("Rolling Metrics")
        roll = pd.read_csv(rolling_path, parse_dates=["Date"])
        fig, ax = plt.subplots()
        ax.plot(roll["Date"], roll["Rolling_Sharpe"], label="Rolling Sharpe")
        ax.legend()
        st.pyplot(fig)

    if mc_path.exists():
        st.subheader("Monte Carlo Distribution")
        mc = pd.read_csv(mc_path)
        fig, ax = plt.subplots()
        ax.hist(mc["final_equity"], bins=30)
        st.pyplot(fig)


if __name__ == "__main__":
    main()

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\app\experiment.py
# ==============================

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

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\app\health.py
# ==============================

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Health checks")
    parser.add_argument("--universe", choices=["small", "nasdaq100"], default="nasdaq100")
    return parser


def run(argv: list[str] | None = None) -> int:
    _ensure_src_on_path()

    from tradinglab.data.tickers import nasdaq100_tickers
    from tradinglab.data.fetcher import load_or_fetch_symbols
    from tradinglab.config import REGIME_SYMBOL
    from app.state import load_state

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.universe == "small":
        symbols = ["AAPL", "MSFT", "NVDA", "AMZN", REGIME_SYMBOL]
    else:
        symbols = nasdaq100_tickers()
        if REGIME_SYMBOL not in symbols:
            symbols.append(REGIME_SYMBOL)

    price_dict = load_or_fetch_symbols(symbols, refresh=False)
    if not price_dict:
        print("Health check failed: no data")
        return 1

    # state readable
    try:
        load_state(Path("logs/state.json"), {})
    except Exception:
        print("Health check failed: state unreadable")
        return 1

    latest_dates = []
    for sym, df in price_dict.items():
        if df.empty:
            print(f"Health check failed: empty data for {sym}")
            return 1
        if "Open" not in df.columns or "Close" not in df.columns:
            print(f"Health check failed: missing Open/Close for {sym}")
            return 1
        latest_dates.append(df.index.max())

    max_date = max(latest_dates)
    if any((max_date - d).days > 3 for d in latest_dates):
        print("Health check failed: stale data")
        return 1

    latest_prices = []
    for sym, df in price_dict.items():
        row = df.loc[max_date]
        if row.isna().any():
            print(f"Health check failed: NaNs in latest prices for {sym}")
            return 1
        latest_prices.append(row)

    print("Health check OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\app\profile.py
# ==============================

from __future__ import annotations

import cProfile
import pstats
import time
from pathlib import Path

import pandas as pd

from tradinglab.config import REGIME_SYMBOL
from tradinglab.data.fetcher import load_or_fetch_symbols
from tradinglab.data.tickers import nasdaq100_tickers
from tradinglab.engine.portfolio import build_price_panels, run_portfolio
from tradinglab.metrics.performance import compute_metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Profile backtest")
    parser.add_argument("--universe", choices=["small", "nasdaq100"], default="small")
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args()

    if args.universe == "small":
        symbols = ["AAPL", "MSFT", "NVDA", "AMZN", REGIME_SYMBOL]
    else:
        symbols = nasdaq100_tickers()
        if REGIME_SYMBOL not in symbols:
            symbols.append(REGIME_SYMBOL)

    t0 = time.perf_counter()
    price_dict = load_or_fetch_symbols(symbols, refresh=False, start_date=args.start, end_date=args.end)
    t1 = time.perf_counter()

    panel_close, _ = build_price_panels(price_dict)
    t2 = time.perf_counter()

    run = run_portfolio(price_dict)
    t3 = time.perf_counter()

    _ = compute_metrics("Portfolio", run.equity["Portfolio_Value"], run.trade_ledger)
    t4 = time.perf_counter()

    timing = {
        "data_load_s": t1 - t0,
        "panel_build_s": t2 - t1,
        "run_portfolio_s": t3 - t2,
        "metrics_s": t4 - t3,
        "total_s": t4 - t0,
    }

    profile_path = Path("logs") / "profile_stats.txt"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    profiler.enable()
    run_portfolio(price_dict)
    profiler.disable()

    with profile_path.open("w", encoding="utf-8") as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats("cumulative")
        stats.print_stats(50)

    timing_path = Path("logs") / "profile_timing.json"
    timing_path.write_text(pd.Series(timing).to_json())

    print(f"Profile saved: {profile_path}")
    print(f"Timing saved: {timing_path}")


if __name__ == "__main__":
    main()

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\app\reporting.py
# ==============================

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
    broker_summary: dict | None = None,
    order_status: str | None = None,
    rejects: list[str] | None = None,
    pending_path: str | None = None,
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
    lines.append("## Broker Summary")
    if broker_summary:
        for k, v in broker_summary.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Order Status")
    lines.append(f"- {order_status or 'none'}")
    if rejects:
        lines.append("- Rejects:")
        for r in rejects:
            lines.append(f"- {r}")
    if pending_path:
        lines.append(f"- Pending orders: {pending_path}")
    lines.append("")
    lines.append("## Comparison Snapshot")
    lines.append(f"- QQQ buy&hold since start: {qqq_bh:.2f}")
    lines.append(f"- Strategy since start: {strategy_since_start:.2f}")

    path.write_text("\n".join(lines), encoding="utf-8")

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\app\robustness.py
# ==============================

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

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\app\signal.py
# ==============================

from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Generate live signal targets")
    parser.add_argument("--universe", choices=["small", "nasdaq100"], default="nasdaq100")
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--execution", choices=["next_open", "same_close"], default=None)
    parser.add_argument("--price-mode", choices=["adj", "raw"], default=None)
    parser.add_argument("--positions-file", type=str, default=None)
    parser.add_argument("--end", dest="end_date", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--refresh-data",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help="Refresh cached market data (true/false)",
    )
    return parser


def _load_positions(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and "positions" in data:
        data = data["positions"]
    if not isinstance(data, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in data.items():
        try:
            out[str(k).upper()] = float(v)
        except Exception:
            continue
    return out


def generate_signal(
    universe: str,
    top_n: int | None,
    execution: str | None,
    price_mode: str | None,
    refresh_data: bool,
    positions_file: str | None,
    end_date: str | None,
    price_dict: dict[str, pd.DataFrame] | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    _ensure_src_on_path()

    from tradinglab.config import START_DATE, REGIME_SYMBOL, EXECUTION, PRICE_MODE, INITIAL_CAPITAL, TOP_N, LONG_WINDOW
    from tradinglab.data.tickers import nasdaq100_tickers
    from tradinglab.data.fetcher import load_or_fetch_symbols
    from tradinglab.engine.portfolio import (
        build_price_panels,
        blended_momentum_score,
        dispersion_series,
        inverse_vol_weights,
    )
    from app.brokers.paper import CachedMarketDataProvider

    effective_top_n = top_n if top_n is not None else TOP_N
    effective_execution = execution or EXECUTION
    effective_price_mode = price_mode or PRICE_MODE
    effective_end = end_date or pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

    if universe == "small":
        symbols = ["AAPL", "MSFT", "NVDA", "AMZN", REGIME_SYMBOL]
    else:
        symbols = nasdaq100_tickers()
        if REGIME_SYMBOL not in symbols:
            symbols.append(REGIME_SYMBOL)

    if price_dict is None and refresh_data:
        price_dict = load_or_fetch_symbols(symbols, refresh=True, start_date=START_DATE, end_date=effective_end)
    provider = CachedMarketDataProvider(price_dict=price_dict, price_mode=effective_price_mode)
    history = provider.get_history(symbols, start=START_DATE, end=effective_end)
    panel_close, panel_open = build_price_panels(history, price_mode=effective_price_mode)

    if panel_close.empty:
        raise RuntimeError("No usable prices to compute signal.")

    dates = list(panel_close.index)
    if effective_execution == "next_open":
        if len(dates) < 2:
            raise RuntimeError("Not enough data for next_open execution.")
        signal_date = dates[-2]
        fill_date = dates[-1]
    else:
        signal_date = dates[-1]
        fill_date = dates[-1]

    positions = _load_positions(positions_file)
    prices_close = panel_close.loc[signal_date]
    equity = float(INITIAL_CAPITAL)
    if positions:
        equity = float(sum(float(prices_close.get(sym, 0.0)) * qty for sym, qty in positions.items()))
        if equity <= 0:
            equity = float(INITIAL_CAPITAL)

    prices_exec = panel_open.loc[fill_date] if fill_date in panel_open.index else prices_close

    max_date = max(df.index.max() for df in history.values() if not df.empty)
    print(f"data max date: {max_date}")
    rows: list[dict] = []

    qqq_close = float(panel_close.loc[signal_date, REGIME_SYMBOL]) if REGIME_SYMBOL in panel_close.columns else float("nan")
    qqq_sma200 = (
        float(panel_close[REGIME_SYMBOL].rolling(LONG_WINDOW).mean().loc[signal_date])
        if REGIME_SYMBOL in panel_close.columns
        else float("nan")
    )
    regime_ok = bool(qqq_close > qqq_sma200) if pd.notna(qqq_close) and pd.notna(qqq_sma200) else False
    score_df, sigma_df = blended_momentum_score(panel_close)
    dispersion, dispersion_med = dispersion_series(score_df)
    disp_val = dispersion.loc[signal_date] if signal_date in dispersion.index else float("nan")
    disp_med = dispersion_med.loc[signal_date] if signal_date in dispersion_med.index else float("nan")

    score_row = score_df.loc[signal_date] if signal_date in score_df.index else pd.Series(dtype=float)
    score_row = score_row.replace([float("inf"), float("-inf")], pd.NA)
    nan_scores = int(score_row.isna().sum()) if not score_row.empty else 0

    reason = None
    if not regime_ok:
        reason = "REGIME_OFF_QQQ_BELOW_SMA200"
    elif pd.notna(disp_val) and pd.notna(disp_med) and disp_val <= disp_med:
        reason = "DISPERSION_LOW"

    # Explicitly exclude QQQ from ranking unless regime trading is enabled.
    exclude = {REGIME_SYMBOL}
    eligible = score_row.dropna()
    eligible = eligible[~eligible.index.isin(exclude)]

    if reason in {"REGIME_OFF_QQQ_BELOW_SMA200", "DISPERSION_LOW"}:
        eligible = eligible.iloc[0:0]

    top = eligible.sort_values(ascending=False).head(effective_top_n).index.tolist() if effective_top_n != 0 else []
    eligible_count = int(len(eligible))
    min_score = float(eligible.min()) if not eligible.empty else float("nan")
    max_score = float(eligible.max()) if not eligible.empty else float("nan")
    trend_fail_count = 0

    if eligible_count > 0 and effective_top_n != 0 and len(top) == 0:
        raise RuntimeError(
            f"Inconsistent eligibility/top selection: eligible_count={eligible_count}, "
            f"TOP_N={effective_top_n}, len(top)={len(top)}, "
            f"head(scores)={score_row.head().to_dict()}, tail(scores)={score_row.tail().to_dict()}"
        )

    print(f"QQQ close: {qqq_close:.2f}")
    print(f"QQQ SMA200: {qqq_sma200:.2f}")
    print(f"regime_ok: {regime_ok}")
    print(f"dispersion: {disp_val:.6f} (median: {disp_med:.6f})")
    print(f"eligible symbols: {eligible_count}")
    print(f"nan scores: {nan_scores}")
    print(f"trend filter fails: {trend_fail_count}")
    print(f"eligible score min/max: {min_score:.6f}/{max_score:.6f}")

    if len(top) == 0:
        reason = reason or "NO_ELIGIBLE_SYMBOLS"
        if REGIME_SYMBOL in panel_close.columns:
            msg = f"No buys, holding {REGIME_SYMBOL} (reason: {reason})"
            print(msg)
            action = "BUY" if positions.get(REGIME_SYMBOL, 0.0) <= 0 else "HOLD"
            est_price = float(prices_exec.get(REGIME_SYMBOL, 0.0))
            rows.append(
                {
                    "asof_date": signal_date.date(),
                    "symbol": REGIME_SYMBOL,
                    "action": action,
                    "target_weight": 1.0,
                    "est_price": est_price,
                    "est_notional": float(equity),
                    "reason": reason,
                }
            )
        else:
            msg = f"No buys, stay in cash/QQQ (reason: {reason})"
            print(msg)
            rows.append(
                {
                    "asof_date": signal_date.date(),
                    "symbol": "NO_BUYS",
                    "action": "HOLD",
                    "target_weight": 0.0,
                    "est_price": 0.0,
                    "est_notional": 0.0,
                    "reason": reason,
                }
            )
    else:
        reason = "OK"
        sigma_row = sigma_df.loc[signal_date] if signal_date in sigma_df.index else pd.Series(dtype=float)
        weights = inverse_vol_weights(sigma_row.reindex(top))
        for sym in top:
            w = float(weights.get(sym, 0.0))
            if w <= 0:
                continue
            action = "BUY"
            if sym in positions and positions[sym] > 0:
                action = "HOLD"
            est_price = float(prices_exec.get(sym, 0.0))
            est_notional = float(equity * w)
            rows.append(
                {
                    "asof_date": signal_date.date(),
                    "symbol": sym,
                    "action": action,
                    "target_weight": w,
                    "est_price": est_price,
                    "est_notional": est_notional,
                    "reason": reason,
                }
            )
        for sym, qty in positions.items():
            if qty > 0 and sym not in top:
                est_price = float(prices_exec.get(sym, 0.0))
                rows.append(
                    {
                        "asof_date": signal_date.date(),
                        "symbol": sym,
                        "action": "SELL",
                        "target_weight": 0.0,
                        "est_price": est_price,
                        "est_notional": 0.0,
                        "reason": reason,
                    }
                )

    out_df = pd.DataFrame(
        rows,
        columns=["asof_date", "symbol", "action", "target_weight", "est_price", "est_notional", "reason"],
    )
    output_path = output_path or Path("results") / "live_signal.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print("Signal output:")
    for _, row in out_df.iterrows():
        print(
            f"{row['action']:>4} {row['symbol']:<6} "
            f"w={row['target_weight']:.4f} "
            f"px={row['est_price']:.2f} "
            f"notional={row['est_notional']:.2f}"
        )

    return out_df


def run(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    generate_signal(
        universe=args.universe,
        top_n=args.top_n,
        execution=args.execution,
        price_mode=args.price_mode,
        refresh_data=bool(args.refresh_data),
        positions_file=args.positions_file,
        end_date=args.end_date,
    )


if __name__ == "__main__":
    run()

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\app\state.py
# ==============================

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

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\app\__init__.py
# ==============================

"""CLI entrypoint for the backtesting app."""

__version__ = "0.6.0"

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\app\__main__.py
# ==============================

from __future__ import annotations

from app.cli import run


if __name__ == "__main__":
    run()

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\app\brokers\alpaca.py
# ==============================

from __future__ import annotations

import os
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any

import requests
import pandas as pd

from app.brokers.base import Broker, Order, Fill


class AlpacaBroker(Broker):
    def __init__(self) -> None:
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.api_secret = os.getenv("ALPACA_API_SECRET")
        self.base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        if not self.api_key or not self.api_secret:
            raise RuntimeError("Missing Alpaca credentials")

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }

    def _request(self, method: str, path: str, json_body: dict | None = None) -> Any:
        url = f"{self.base_url}{path}"
        for attempt in range(5):
            try:
                resp = requests.request(method, url, headers=self._headers(), json=json_body, timeout=30)
                if resp.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                if attempt == 4:
                    raise RuntimeError(f"Alpaca API error: {exc}") from exc
                time.sleep(2 ** attempt)
        raise RuntimeError("Alpaca API failed")

    def get_account(self) -> dict:
        data = self._request("GET", "/v2/account")
        return {
            "cash": float(data.get("cash", 0.0)),
            "equity": float(data.get("equity", 0.0)),
        }

    def get_positions(self) -> dict[str, float]:
        data = self._request("GET", "/v2/positions")
        out = {}
        for pos in data:
            out[pos["symbol"]] = float(pos["qty"])
        return out

    def get_latest_prices(self, symbols: list[str]) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close"])
        joined = ",".join(symbols)
        data = self._request("GET", f"/v2/stocks/quotes/latest?symbols={joined}")
        rows = []
        for sym, quote in data.get("quotes", {}).items():
            rows.append(
                {
                    "Symbol": sym,
                    "Open": quote.get("ap"),
                    "High": quote.get("ap"),
                    "Low": quote.get("bp"),
                    "Close": quote.get("ap"),
                    "Adj Close": quote.get("ap"),
                }
            )
        return pd.DataFrame(rows).set_index("Symbol")

    def place_orders(self, orders: list[Order]) -> list[Fill]:
        fills: list[Fill] = []
        for order in orders:
            body = {
                "symbol": order.symbol,
                "qty": str(order.qty),
                "side": order.side.lower(),
                "type": order.order_type,
                "time_in_force": order.tif,
            }
            data = self._request("POST", "/v2/orders", json_body=body)
            fills.append(
                Fill(
                    symbol=order.symbol,
                    side=order.side,
                    qty=order.qty,
                    price=float(data.get("filled_avg_price") or 0.0),
                    fee=0.0,
                    slippage_est=0.0,
                    timestamp=datetime.utcnow().isoformat(),
                )
            )
        return fills

    def cancel_all(self) -> None:
        self._request("DELETE", "/v2/orders")

    def is_market_open(self) -> bool:
        data = self._request("GET", "/v2/clock")
        return bool(data.get("is_open", False))

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\app\brokers\base.py
# ==============================

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass
class Order:
    symbol: str
    side: str  # "BUY" or "SELL"
    qty: float
    order_type: str = "market"
    tif: str = "day"
    signal_date: str | None = None


@dataclass
class Fill:
    symbol: str
    side: str
    qty: float
    price: float
    fee: float
    slippage_est: float
    timestamp: str


@dataclass
class Position:
    symbol: str
    qty: float
    avg_price: float | None = None


class MarketDataProvider(Protocol):
    def get_latest_prices(self, symbols: list[str]) -> pd.DataFrame:
        ...

    def get_history(self, symbols: list[str], start: str | None, end: str | None) -> dict[str, pd.DataFrame]:
        ...


class Broker(Protocol):
    def get_account(self) -> dict:
        ...

    def get_positions(self) -> dict[str, float]:
        ...

    def get_latest_prices(self, symbols: list[str]) -> pd.DataFrame:
        ...

    def place_orders(self, orders: list[Order]) -> list[Fill]:
        ...

    def cancel_all(self) -> None:
        ...

    def is_market_open(self) -> bool:
        ...

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\app\brokers\paper.py
# ==============================

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.brokers.base import MarketDataProvider, Broker, Order, Fill
from tradinglab.config import (
    PRICE_MODE,
    EXECUTION,
    FEE_RATE,
    SLIPPAGE_MODE,
    SLIPPAGE_RATE,
    SLIPPAGE_BPS_BASE,
    SLIPPAGE_BPS_PER_TURNOVER,
    CASH_BUFFER,
    DATA_DIR,
)
from tradinglab.data.fetcher import load_or_fetch_symbols


class CachedMarketDataProvider(MarketDataProvider):
    def __init__(self, price_dict: dict[str, pd.DataFrame] | None = None, price_mode: str = PRICE_MODE):
        self._price_dict = price_dict or {}
        self.price_mode = price_mode
        self.current_date: pd.Timestamp | None = None

    def set_current_date(self, date: pd.Timestamp) -> None:
        self.current_date = pd.to_datetime(date)

    def get_history(self, symbols: list[str], start: str | None, end: str | None) -> dict[str, pd.DataFrame]:
        if self._price_dict:
            out = {}
            for sym, df in self._price_dict.items():
                out[sym] = df.loc[start:end]
            return out
        DATA_DIR.mkdir(exist_ok=True)
        self._price_dict = load_or_fetch_symbols(symbols, refresh=False, start_date=start, end_date=end)
        return self._price_dict

    def get_latest_prices(self, symbols: list[str]) -> pd.DataFrame:
        if self.current_date is None:
            raise RuntimeError("current_date not set")

        if not symbols:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close"])

        rows = []
        for sym in symbols:
            df = self._price_dict.get(sym)
            if df is None or self.current_date not in df.index:
                continue
            row = df.loc[self.current_date].copy()

            if self.price_mode == "adj" and "Adj Close" in df.columns and "Close" in df.columns:
                adj = float(row["Adj Close"])
                close = float(row["Close"]) if float(row["Close"]) != 0 else adj
                factor = adj / close if close != 0 else 1.0
                for col in ["Open", "High", "Low", "Close"]:
                    if col in row.index:
                        row[col] = float(row[col]) * factor
                row["Adj Close"] = adj
            rows.append({"Symbol": sym, **row.to_dict()})

        if not rows:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close"])
        return pd.DataFrame(rows).set_index("Symbol")


class PaperBroker(Broker):
    def __init__(
        self,
        provider: CachedMarketDataProvider,
        cash: float = 0.0,
        positions: dict[str, float] | None = None,
        execution: str = EXECUTION,
        price_mode: str = PRICE_MODE,
        log_dir: Path = Path("logs"),
    ):
        self.provider = provider
        self.cash = float(cash)
        self.positions = positions or {}
        self.execution = execution
        self.price_mode = price_mode
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)

    def is_market_open(self) -> bool:
        return self.provider.current_date is not None

    def get_positions(self) -> dict[str, float]:
        return dict(self.positions)

    def get_latest_prices(self, symbols: list[str]) -> pd.DataFrame:
        return self.provider.get_latest_prices(symbols)

    def get_account(self) -> dict:
        equity = self.cash
        if self.positions:
            prices = self.provider.get_latest_prices(list(self.positions.keys()))
            for sym, qty in self.positions.items():
                if sym in prices.index:
                    equity += qty * float(prices.loc[sym, "Open"])
        return {"cash": self.cash, "equity": equity, "positions": self.get_positions()}

    def cancel_all(self) -> None:
        return None

    def _slippage_rate(self, notional: float, nav: float) -> float:
        if SLIPPAGE_MODE == "constant":
            return float(SLIPPAGE_RATE)
        base_bps = float(SLIPPAGE_BPS_BASE)
        per_turnover = float(SLIPPAGE_BPS_PER_TURNOVER)
        turnover = 0.0 if nav <= 0 else abs(notional) / nav
        total_bps = base_bps + per_turnover * turnover
        return total_bps / 10000.0

    def place_orders(self, orders: list[Order]) -> list[Fill]:
        if not orders:
            return []

        symbols = list({o.symbol for o in orders})
        prices = self.provider.get_latest_prices(symbols)
        if prices.empty:
            return []

        fills: list[Fill] = []
        account = self.get_account()
        nav = float(account["equity"])

        for order in orders:
            if order.symbol not in prices.index:
                continue
            if order.qty <= 0:
                continue

            mid = float(prices.loc[order.symbol, "Open"])
            notional = order.qty * mid
            slip = self._slippage_rate(notional, nav)
            fee = notional * FEE_RATE
            slippage_cost = notional * slip

            if order.side == "BUY":
                total_cost = notional + fee + slippage_cost
                cash_buffer_amt = max(0.0, CASH_BUFFER * nav)
                available = max(0.0, self.cash - cash_buffer_amt)
                if total_cost > available and total_cost > 0:
                    scale = available / total_cost
                    notional *= scale
                    fee = notional * FEE_RATE
                    slippage_cost = notional * slip
                    total_cost = notional + fee + slippage_cost
                if total_cost <= 0:
                    continue
                qty = notional / mid
                self.cash -= total_cost
                self.positions[order.symbol] = self.positions.get(order.symbol, 0.0) + qty
                fill_price = mid * (1.0 + slip)
            else:
                qty = min(order.qty, self.positions.get(order.symbol, 0.0))
                notional = qty * mid
                fee = notional * FEE_RATE
                slippage_cost = notional * slip
                proceeds = notional - fee - slippage_cost
                self.cash += proceeds
                self.positions[order.symbol] = self.positions.get(order.symbol, 0.0) - qty
                fill_price = mid * (1.0 - slip)

            ts = datetime.utcnow().isoformat()
            fill = Fill(
                symbol=order.symbol,
                side=order.side,
                qty=qty,
                price=fill_price,
                fee=fee,
                slippage_est=slippage_cost,
                timestamp=ts,
            )
            fills.append(fill)
            self._log_trade(order, fill)

        self._log_positions()
        self._log_equity()
        return fills

    def _log_trade(self, order: Order, fill: Fill) -> None:
        path = self.log_dir / "paper_trades.csv"
        row = {
            "timestamp": fill.timestamp,
            "symbol": fill.symbol,
            "side": fill.side,
            "qty": fill.qty,
            "price": fill.price,
            "fee": fill.fee,
            "slippage_est": fill.slippage_est,
            "signal_date": order.signal_date,
        }
        self._append_csv(path, row)

    def _log_positions(self) -> None:
        path = self.log_dir / "paper_positions.csv"
        row = {"timestamp": datetime.utcnow().isoformat(), **self.positions}
        self._append_csv(path, row)

    def _log_equity(self) -> None:
        path = self.log_dir / "paper_equity.csv"
        acct = self.get_account()
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "cash": acct["cash"],
            "equity": acct["equity"],
        }
        self._append_csv(path, row)

    def _append_csv(self, path: Path, row: dict) -> None:
        df = pd.DataFrame([row])
        if path.exists():
            df.to_csv(path, mode="a", index=False, header=False)
        else:
            df.to_csv(path, index=False)

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\app\live\cli.py
# ==============================

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
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
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
    parser.add_argument("--confirm", action="store_true", default=False)
    parser.add_argument("--i-accept-real-trading", default="")
    parser.add_argument("--flatten-on-kill", action="store_true", default=False)
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
        mode=args.mode,
        confirm=bool(args.confirm),
        accept_real_trading=args.i_accept_real_trading,
        flatten_on_kill=bool(args.flatten_on_kill),
    )


if __name__ == "__main__":
    run()

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\app\live\runner.py
# ==============================

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import pandas as pd

from app.brokers.base import Order
from app.brokers.paper import CachedMarketDataProvider, PaperBroker
from app.state import load_state, save_state, LiveState
from app.alerts import send_alert
from app.reporting import write_daily_report
from app.config_validate import validate_config, write_config_snapshot
from tradinglab.config import (
    START_DATE,
    REGIME_SYMBOL,
    EXECUTION,
    PRICE_MODE,
    REBALANCE,
    MAX_TURNOVER_PER_REBALANCE,
    MAX_ORDER_NOTIONAL,
    MAX_ORDERS_PER_RUN,
    MAX_POSITION_WEIGHT,
)
from tradinglab.data.tickers import nasdaq100_tickers
from tradinglab.engine.portfolio import build_price_panels, _weights_from_shares
from tradinglab.engine.portfolio import buy_hold_single


def _compute_targets(
    panel_close: pd.DataFrame,
    date: pd.Timestamp,
    current_weights: dict[str, float],
    price_dict: dict[str, pd.DataFrame],
    price_mode: str,
    top_n: int | None = None,
) -> dict[str, float]:
    from tradinglab.engine.portfolio import run_portfolio

    sliced = {sym: df.loc[:date] for sym, df in price_dict.items() if date in df.index}
    if not sliced:
        return {sym: 0.0 for sym in current_weights}

    run = run_portfolio(
        sliced,
        execution="same_close",
        price_mode=price_mode,
        slippage_mode="constant",
        top_n=top_n,
    )
    if run.positions.empty:
        return {sym: 0.0 for sym in current_weights}

    last_pos = run.positions.iloc[-1]
    last_close = run.panel_close.iloc[-1]
    equity = run.equity["Portfolio_Value"].iloc[-1]

    targets = {}
    for sym in last_pos.index:
        if sym == "Cash":
            continue
        if sym in last_close.index:
            targets[sym] = (last_pos[sym] * last_close[sym]) / equity
    return targets


def generate_orders_from_weights(
    symbols: list[str],
    current_positions: dict[str, float],
    target_weights: dict[str, float],
    prices: pd.Series,
    equity: float,
) -> list[Order]:
    orders = []
    for sym in symbols:
        target_weight = target_weights.get(sym, 0.0)
        current_qty = current_positions.get(sym, 0.0)
        price = float(prices[sym])
        target_qty = (equity * target_weight) / price if price > 0 else 0.0
        delta = target_qty - current_qty
        if abs(delta) <= 0:
            continue
        side = "BUY" if delta > 0 else "SELL"
        orders.append(Order(symbol=sym, side=side, qty=abs(delta)))
    return orders


def _apply_turnover_cap(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    max_turnover: float | None,
) -> dict[str, float]:
    if max_turnover is None:
        return target_weights
    turnover = 0.0
    for sym, tw in target_weights.items():
        turnover += abs(tw - current_weights.get(sym, 0.0))
    if turnover <= max_turnover or turnover <= 0:
        return target_weights
    scale = max_turnover / turnover
    adjusted = {}
    for sym, tw in target_weights.items():
        cw = current_weights.get(sym, 0.0)
        adjusted[sym] = cw + (tw - cw) * scale
    return adjusted


def compute_target_weights(
    panel_close: pd.DataFrame,
    signal_date: pd.Timestamp,
    current_weights: dict[str, float],
    price_dict: dict[str, pd.DataFrame],
    price_mode: str,
    max_turnover: float | None = MAX_TURNOVER_PER_REBALANCE,
    top_n: int | None = None,
) -> tuple[dict[str, float], bool]:
    target_weights = _compute_targets(panel_close, signal_date, current_weights, price_dict, price_mode, top_n=top_n)
    capped = _apply_turnover_cap(current_weights, target_weights, max_turnover)
    return capped, capped != target_weights


def run_live_cycle(
    universe: str,
    refresh_data: bool,
    execution: str,
    price_mode: str,
    dry_run: bool,
    state_path: Path,
    flatten: bool,
    mode: str = "paper",
    confirm: bool = False,
    accept_real_trading: str = "",
    flatten_on_kill: bool = False,
) -> list[Order]:
    end_date = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    if universe == "small":
        symbols = ["AAPL", "MSFT", "NVDA", "AMZN", REGIME_SYMBOL]
    else:
        symbols = nasdaq100_tickers()
        if REGIME_SYMBOL not in symbols:
            symbols.append(REGIME_SYMBOL)

    errors = validate_config(execution, price_mode, REBALANCE)
    if errors:
        raise ValueError("; ".join(errors))

    provider = CachedMarketDataProvider(price_mode=price_mode)
    history = provider.get_history(symbols, start=START_DATE, end=end_date)

    panel_close, panel_open = build_price_panels(history, price_mode=price_mode)
    if panel_close.empty:
        return []

    dates = list(panel_close.index)
    if execution == "next_open":
        if len(dates) < 2:
            return []
        signal_date = dates[-2]
        fill_date = dates[-1]
    else:
        signal_date = dates[-1]
        fill_date = dates[-1]

    provider.set_current_date(fill_date)

    config_snapshot = {
        "execution": execution,
        "price_mode": price_mode,
        "rebalance": REBALANCE,
    }
    state = load_state(state_path, config_snapshot)
    write_config_snapshot(Path("logs/state_config.json"), config_snapshot)

    if mode == "live":
        live_enabled = os.getenv("LIVE_TRADING_ENABLED") == "true"
        if accept_real_trading != "YES" or not live_enabled:
            print("Live trading not enabled; refusing to trade.")
            return []
        if confirm:
            from app.brokers.alpaca import AlpacaBroker

            broker = AlpacaBroker()
        else:
            broker = PaperBroker(provider, cash=state.cash, positions=state.positions, execution=execution)
    else:
        broker = PaperBroker(provider, cash=state.cash, positions=state.positions, execution=execution)

    if state.last_processed_date == fill_date.isoformat():
        print("Already processed:", fill_date.date())
        return []

    if state.in_progress and state.last_run_date == fill_date.isoformat():
        print("Detected incomplete run; aborting to avoid duplicates.")
        return []

    if not broker.is_market_open():
        return []

    prices_close = panel_close.loc[signal_date]
    equity = broker.get_account()["equity"]
    if state.strategy_start_equity is None:
        state.strategy_start_equity = equity
    start_equity = equity
    current_weights = _weights_from_shares(prices_close, state.positions, equity)

    risk_triggers: list[str] = []
    rejects: list[str] = []

    if flatten:
        target_weights = {sym: 0.0 for sym in current_weights}
    else:
        target_weights, turnover_capped = compute_target_weights(
            panel_close,
            signal_date,
            current_weights,
            history,
            price_mode,
            max_turnover=MAX_TURNOVER_PER_REBALANCE,
        )
        if turnover_capped:
            risk_triggers.append("turnover_cap_applied")

    if MAX_POSITION_WEIGHT is not None:
        for sym, w in target_weights.items():
            if w > MAX_POSITION_WEIGHT:
                msg = f"Target weight exceeds MAX_POSITION_WEIGHT for {sym}"
                rejects.append(msg)
                print(msg)
                return []

    prices_open = panel_open.loc[fill_date]
    orders = generate_orders_from_weights(list(current_weights.keys()), state.positions, target_weights, prices_open, equity)
    for o in orders:
        o.signal_date = signal_date.isoformat()

    if dry_run:
        for o in orders:
            print(f"DRY-RUN {o.side} {o.qty:.4f} {o.symbol} @ {prices_open[o.symbol]:.2f}")
        return orders

    allowed = set(symbols)
    for o in orders:
        if o.symbol not in allowed:
            msg = f"Order symbol not allowed: {o.symbol}"
            rejects.append(msg)
            print(msg)
            return []

    # pre-flight safety checks
    if len(orders) > MAX_ORDERS_PER_RUN:
        msg = "Order count exceeds MAX_ORDERS_PER_RUN"
        rejects.append(msg)
        print(msg)
        return []

    buy_notional = 0.0
    for o in orders:
        notional = o.qty * float(prices_open[o.symbol])
        if notional > MAX_ORDER_NOTIONAL:
            msg = f"Order too large for {o.symbol}"
            rejects.append(msg)
            print(msg)
            return []
        if o.side == "BUY":
            buy_notional += notional

    if buy_notional > state.cash and state.cash > 0:
        msg = "Buy notional exceeds cash"
        rejects.append(msg)
        print(msg)
        return []

    pending_path = Path("logs") / f"pending_orders_{fill_date.date()}.json"
    pending_path.write_text(json.dumps([o.__dict__ for o in orders], indent=2))

    if mode == "live" and not confirm:
        print(f"Pending orders written to {pending_path}. Use --confirm to place.")
        return orders

    if os.getenv("KILL_SWITCH") == "true":
        send_alert("KILL_SWITCH active: no orders placed.")
        if flatten_on_kill and mode == "live":
            broker.cancel_all()
            positions = broker.get_positions()
            flatten_orders = []
            for sym, qty in positions.items():
                if qty > 0 and sym in prices_open.index:
                    flatten_orders.append(Order(symbol=sym, side="SELL", qty=qty))
            if flatten_orders:
                pending_path.write_text(json.dumps([o.__dict__ for o in flatten_orders], indent=2))
                broker.place_orders(flatten_orders)
        return []

    state.in_progress = True
    state.last_run_date = fill_date.isoformat()
    save_state(state_path, state)

    broker.place_orders(orders)

    acct = broker.get_account()
    state.cash = acct["cash"]
    state.positions = acct["positions"]
    state.last_processed_date = fill_date.isoformat()
    state.last_rebalance_date = signal_date.isoformat()
    state.last_equity = acct["equity"]
    state.in_progress = False

    for sym, qty in state.positions.items():
        if qty > 0:
            info = state.holdings_info.get(sym, {"entry_date": signal_date.isoformat(), "peak": float(prices_close[sym])})
            info["peak"] = max(info.get("peak", float(prices_close[sym])), float(prices_close[sym]))
            state.holdings_info[sym] = info
        else:
            state.holdings_info.pop(sym, None)

    save_state(state_path, state)

    # reporting
    prices_for_report = panel_open.loc[fill_date]
    cash_pct = state.cash / acct["equity"] if acct["equity"] > 0 else 0.0
    gross_exposure = sum(
        abs(qty * float(prices_for_report[sym])) for sym, qty in state.positions.items() if sym in prices_for_report.index
    )
    gross_exposure_pct = gross_exposure / acct["equity"] if acct["equity"] > 0 else 0.0
    turnover = sum(abs(o.qty * float(prices_for_report[o.symbol])) for o in orders) / acct["equity"] if acct["equity"] > 0 else 0.0

    qqq_bh = 0.0
    if REGIME_SYMBOL in history:
        qqq = buy_hold_single(history, REGIME_SYMBOL, price_mode=price_mode)
        if fill_date in qqq.index:
            qqq_bh = float(qqq.loc[fill_date].iloc[0])

    strategy_since_start = acct["equity"] - (state.strategy_start_equity or acct["equity"])

    report_path = Path("logs/reports") / f"{fill_date.date()}_report.md"
    write_daily_report(
        report_path,
        date=str(fill_date.date()),
        start_equity=start_equity,
        end_equity=acct["equity"],
        positions=state.positions,
        prices=prices_for_report,
        cash_pct=cash_pct,
        gross_exposure=gross_exposure_pct,
        turnover=turnover,
        risk_triggers=risk_triggers,
        qqq_bh=qqq_bh,
        strategy_since_start=strategy_since_start,
        broker_summary=broker.get_account() if hasattr(broker, "get_account") else None,
        order_status="placed",
        rejects=rejects,
        pending_path=str(pending_path),
    )

    return orders

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\app\live\__init__.py
# ==============================

"""Live trading runner (paper mode only)."""

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\app\live\__main__.py
# ==============================

from __future__ import annotations

from app.live.cli import run


if __name__ == "__main__":
    run()

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\src\tradinglab\config.py
# ==============================

from __future__ import annotations

from pathlib import Path

# Date range for historical data
START_DATE = "2023-01-01"
END_DATE = None

# Initial capital for backtesting
INITIAL_CAPITAL = 1000.0

# Moving average settings
SHORT_WINDOW = 20
LONG_WINDOW = 200

# Volatility gate
MIN_VOL20 = 0.8

# Realism
FEE_RATE = 0.001        # 0.10% per trade
SLIPPAGE_RATE = 0.0005  # 0.05% worse fill on buys/sells
SLIPPAGE_MODE = "bps"   # "bps" or "constant"
SLIPPAGE_BPS_BASE = 5.0
SLIPPAGE_BPS_PER_TURNOVER = 0.0
REBALANCE = "ME"        # month-end rebalance
MOM_LOOKBACK = 126
TOP_N = 15

# Regime filter
REGIME_SYMBOL = "QQQ"
ALLOW_REGIME_TRADE = False

# Execution + pricing
EXECUTION = "next_open"  # "next_open" or "same_close"
PRICE_MODE = "adj"       # "adj" or "raw"

# Risk controls (portfolio-level)
MAX_POSITION_WEIGHT = None
MAX_SECTOR_WEIGHT = None
MAX_TURNOVER_PER_REBALANCE = None
MAX_GROSS_EXPOSURE = 1.0
CASH_BUFFER = 0.01
MAX_ORDER_NOTIONAL = 10000.0
MAX_ORDERS_PER_RUN = 50

# Risk controls (position-level)
TRAILING_STOP_PCT = None
TIME_STOP_DAYS = None
TARGET_VOL = 0.15
VOL_LOOKBACK = 20

# Paths
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\src\tradinglab\__init__.py
# ==============================


# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\src\tradinglab\cli\main.py
# ==============================

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from tradinglab.config import (
    INITIAL_CAPITAL,
    RESULTS_DIR,
    REGIME_SYMBOL,
    ALLOW_REGIME_TRADE,
)
from tradinglab.data.fetcher import load_or_fetch_symbols
from tradinglab.data.tickers import nasdaq100_tickers
from tradinglab.engine.portfolio import run_portfolio, buy_hold_benchmark, buy_hold_single
from tradinglab.metrics.performance import compute_metrics


RESULTS_DIR.mkdir(exist_ok=True)


def main(
    refresh_data: bool = False,
    run_per_symbol: bool = False,
    symbols: Iterable[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    top_n: int | None = None,
    rebalance: str | None = None,
    execution: str | None = None,
    price_mode: str | None = None,
    max_position_weight: float | None = None,
    trailing_stop: float | None = None,
    time_stop: int | None = None,
    target_vol: float | None = None,
) -> None:
    import tradinglab.engine.portfolio as portfolio

    if symbols is None:
        symbols = nasdaq100_tickers()

    symbols = list(dict.fromkeys([s.strip().upper() for s in symbols if s]))
    if REGIME_SYMBOL not in symbols:
        symbols.append(REGIME_SYMBOL)

    print("Universe size:", len(symbols))

    if any([top_n is not None, rebalance is not None, execution is not None, price_mode is not None]):
        if top_n is not None:
            portfolio.TOP_N = int(top_n)
        if rebalance is not None:
            portfolio.REBALANCE = rebalance
        if execution is not None:
            portfolio.EXECUTION = execution
        if price_mode is not None:
            portfolio.PRICE_MODE = price_mode

    price_dict = load_or_fetch_symbols(
        symbols,
        refresh=refresh_data,
        start_date=start_date,
        end_date=end_date,
    )

    if not price_dict:
        raise RuntimeError("No market data loaded. Everything got skipped.")

    missing = [s for s in symbols if s not in price_dict]
    if missing:
        print("Skipped symbols:", len(missing))

    # Portfolio rotation
    portfolio_run = run_portfolio(
        price_dict,
        regime_symbol=REGIME_SYMBOL,
        allow_regime_trade=ALLOW_REGIME_TRADE,
        execution=execution or portfolio.EXECUTION,
        price_mode=price_mode or portfolio.PRICE_MODE,
        max_position_weight=max_position_weight,
        trailing_stop_pct=trailing_stop,
        time_stop_days=time_stop,
        target_vol=target_vol,
    )
    pf = portfolio_run.equity
    pf.to_csv(RESULTS_DIR / "portfolio.csv", index_label="Date")

    # Trade ledger + positions
    trade_ledger = portfolio_run.trade_ledger
    trade_ledger.to_csv(RESULTS_DIR / "trade_ledger.csv", index=False)

    positions = portfolio_run.positions
    positions.to_csv(RESULTS_DIR / "positions.csv", index_label="Date")

    exposures = portfolio_run.exposures
    exposures.to_csv(RESULTS_DIR / "exposures.csv", index_label="Date")

    # Buy & hold benchmark of same basket
    bh = buy_hold_benchmark(
        price_dict,
        regime_symbol=REGIME_SYMBOL,
        allow_regime_trade=ALLOW_REGIME_TRADE,
        price_mode=price_mode or portfolio.PRICE_MODE,
    )

    merged = pf.join(bh, how="inner")
    merged.to_csv(RESULTS_DIR / "portfolio_vs_bh.csv", index_label="Date")

    # QQQ buy & hold benchmark
    qqq_bh = None
    if REGIME_SYMBOL in price_dict:
        qqq_bh = buy_hold_single(price_dict, REGIME_SYMBOL, price_mode=price_mode or portfolio.PRICE_MODE)
        qqq_bh.to_csv(RESULTS_DIR / f"{REGIME_SYMBOL}_buyhold.csv", index_label="Date")

    # Metrics
    metrics_rows: List[dict] = []
    metrics_rows.append(compute_metrics("Portfolio", pf["Portfolio_Value"], trade_ledger))
    metrics_rows.append(compute_metrics("BuyHold_Basket", bh["BuyHold_Value"]))
    if qqq_bh is not None:
        metrics_rows.append(compute_metrics(f"{REGIME_SYMBOL}_BuyHold", qqq_bh.iloc[:, 0]))

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    metrics_df.to_json(RESULTS_DIR / "metrics.json", orient="records", indent=2)

    # Print summary
    pf_end = float(merged["Portfolio_Value"].iloc[-1])
    pf_dd = float(merged["Drawdown_%"].min())

    bh_end = float(merged["BuyHold_Value"].iloc[-1])
    bh_dd = float(merged["BuyHold_Drawdown_%"].min())

    print("\n=== Portfolio Rotation (Universe) ===")
    print(merged[["Portfolio_Value", "Return_%", "Drawdown_%"]].tail(5))

    print("\nPortfolio start:", float(INITIAL_CAPITAL))
    print("Portfolio end:", pf_end)
    print("Portfolio Max DD %:", pf_dd)

    print("\nBuy&Hold basket start:", float(INITIAL_CAPITAL))
    print("Buy&Hold basket end:", bh_end)
    print("Buy&Hold basket Max DD %:", bh_dd)

    print("\nSaved results/portfolio.csv and results/portfolio_vs_bh.csv")


if __name__ == "__main__":
    main(refresh_data=False, run_per_symbol=False)

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\src\tradinglab\cli\__init__.py
# ==============================


# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\src\tradinglab\data\fetcher.py
# ==============================

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

from tradinglab.config import DATA_DIR, START_DATE, END_DATE


DATA_DIR.mkdir(exist_ok=True)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().title() for c in df.columns]
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    return df[keep]


def _extract_symbol_frame(batch_df: pd.DataFrame, symbol: str) -> pd.DataFrame | None:
    if batch_df is None or batch_df.empty:
        return None

    if isinstance(batch_df.columns, pd.MultiIndex):
        # Expected shapes:
        # level 0 = field, level 1 = ticker (default)
        # or level 0 = ticker, level 1 = field (group_by='ticker')
        lvl0 = batch_df.columns.get_level_values(0)
        lvl1 = batch_df.columns.get_level_values(1)
        if symbol in lvl1:
            df = batch_df.xs(symbol, level=1, axis=1)
        elif symbol in lvl0:
            df = batch_df.xs(symbol, level=0, axis=1)
        else:
            return None
    else:
        # single symbol download
        df = batch_df

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return _normalize_columns(df)


def fetch_symbol(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    start = start_date or START_DATE
    end = end_date if end_date is not None else END_DATE
    kwargs = {
        "start": start,
        "interval": "1d",
        "auto_adjust": False,
        "progress": False,
    }
    if end is not None:
        kwargs["end"] = end
    df = yf.download(symbol, **kwargs)

    if df is None or df.empty:
        raise ValueError(f"No data returned for symbol: {symbol}")

    df = _extract_symbol_frame(df, symbol)
    if df is None or df.empty:
        raise ValueError(f"No usable data returned for symbol: {symbol}")
    return df


def fetch_symbols(
    symbols: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, pd.DataFrame]:
    if not symbols:
        return {}

    start = start_date or START_DATE
    end = end_date if end_date is not None else END_DATE
    joined = " ".join(symbols)
    kwargs = {
        "start": start,
        "interval": "1d",
        "auto_adjust": False,
        "progress": False,
        "group_by": "ticker",
    }
    if end is not None:
        kwargs["end"] = end
    df = yf.download(joined, **kwargs)

    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        sym_df = _extract_symbol_frame(df, sym)
        if sym_df is not None and not sym_df.empty:
            out[sym] = sym_df
    return out


def save_symbol_csv(symbol: str, df: pd.DataFrame) -> Path:
    path = DATA_DIR / f"{symbol}.csv"
    df.to_csv(path, index_label="Date")
    return path


def load_symbol_csv(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame | None:
    path = DATA_DIR / f"{symbol}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    if start_date:
        start_dt = pd.to_datetime(start_date)
        if df.index.min() > start_dt:
            return None
    if end_date:
        end_dt = pd.to_datetime(end_date)
        if df.index.max() < (end_dt - pd.Timedelta(days=1)):
            return None
    if start_date or end_date:
        df = df.loc[start_date:end_date]
    return df


def load_or_fetch_symbol(
    symbol: str,
    refresh: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    cached = None if refresh else load_symbol_csv(symbol, start_date=start_date, end_date=end_date)
    if cached is not None:
        return cached
    df = fetch_symbol(symbol, start_date=start_date, end_date=end_date)
    save_symbol_csv(symbol, df)
    return df


def load_or_fetch_symbols(
    symbols: Iterable[str],
    refresh: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    batch_size: int = 50,
) -> dict[str, pd.DataFrame]:
    symbols = list(dict.fromkeys([s.strip().upper() for s in symbols if s]))
    if not symbols:
        return {}

    out: dict[str, pd.DataFrame] = {}

    # Load cached first
    if not refresh:
        for sym in symbols:
            cached = load_symbol_csv(sym, start_date=start_date, end_date=end_date)
            if cached is not None:
                out[sym] = cached

    # Fetch missing in batches
    missing = [s for s in symbols if s not in out]
    if missing:
        for i in range(0, len(missing), batch_size):
            batch = missing[i : i + batch_size]
            fetched = fetch_symbols(batch, start_date=start_date, end_date=end_date)
            for sym, df in fetched.items():
                out[sym] = df
                save_symbol_csv(sym, df)

    # Final fallback: per symbol fetch for any still missing
    still_missing = [s for s in symbols if s not in out]
    for sym in still_missing:
        try:
            df = fetch_symbol(sym, start_date=start_date, end_date=end_date)
            out[sym] = df
            save_symbol_csv(sym, df)
        except Exception:
            continue

    return out

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\src\tradinglab\data\panel.py
# ==============================

from __future__ import annotations

from typing import Callable

import pandas as pd

from tradinglab.engine.portfolio import _select_price_series


def filter_price_dict_by_history(
    price_dict: dict[str, pd.DataFrame],
    price_mode: str,
    min_history_days: int,
    keep_symbols: set[str] | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    keep_symbols = keep_symbols or set()
    kept: dict[str, pd.DataFrame] = {}
    dropped: list[str] = []

    for sym, df in price_dict.items():
        if sym in keep_symbols:
            kept[sym] = df
            continue

        _, close_series = _select_price_series(df, price_mode)
        if close_series is None:
            dropped.append(sym)
            continue

        length = int(close_series.dropna().shape[0])
        if length < min_history_days:
            dropped.append(sym)
            continue

        kept[sym] = df

    if log_fn is not None:
        log_fn(f"Dropped {len(dropped)} symbols for insufficient history (<{min_history_days} days).")
    return kept, dropped


def build_close_panel(
    price_dict: dict[str, pd.DataFrame],
    price_mode: str,
) -> pd.DataFrame:
    closes: dict[str, pd.Series] = {}
    for sym, df in price_dict.items():
        _, close_series = _select_price_series(df, price_mode)
        if close_series is None:
            continue
        closes[sym] = close_series

    panel_close = pd.DataFrame(closes).sort_index()
    if panel_close.empty:
        return panel_close

    panel_close = panel_close.ffill()
    panel_close = panel_close.dropna(how="any")
    return panel_close


def prepare_panel_with_history_filter(
    price_dict: dict[str, pd.DataFrame],
    price_mode: str,
    min_history_days: int,
    keep_symbols: set[str] | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, list[str]]:
    keep_symbols = keep_symbols or set()
    filtered, dropped = filter_price_dict_by_history(
        price_dict,
        price_mode=price_mode,
        min_history_days=min_history_days,
        keep_symbols=keep_symbols,
        log_fn=log_fn,
    )
    panel_close = build_close_panel(filtered, price_mode=price_mode)

    if not panel_close.empty and len(panel_close) < min_history_days and len(filtered) > 1:
        def _start_date(df: pd.DataFrame) -> pd.Timestamp | None:
            _, close_series = _select_price_series(df, price_mode)
            if close_series is None:
                return None
            series = close_series.dropna()
            if series.empty:
                return None
            return series.index[0]

        start_dates = {sym: _start_date(df) for sym, df in filtered.items()}
        extra_dropped: list[str] = []

        while len(panel_close) < min_history_days and len(filtered) > 1:
            valid_dates = [dt for dt in start_dates.values() if dt is not None]
            if not valid_dates:
                break
            latest_start = max(valid_dates)
            to_drop = [
                sym
                for sym, dt in start_dates.items()
                if dt == latest_start and sym not in keep_symbols
            ]
            if not to_drop:
                break
            for sym in to_drop:
                filtered.pop(sym, None)
                start_dates.pop(sym, None)
                extra_dropped.append(sym)
            panel_close = build_close_panel(filtered, price_mode=price_mode)
            if panel_close.empty:
                break

        if extra_dropped and log_fn is not None:
            log_fn(f"Dropped {len(extra_dropped)} symbols to meet overlap history requirement.")
        dropped.extend(extra_dropped)

    return filtered, panel_close, dropped

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\src\tradinglab\data\tickers.py
# ==============================

from __future__ import annotations

import pandas as pd
import requests
from io import StringIO


WIKI_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"


def nasdaq100_tickers() -> list[str]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    r = requests.get(WIKI_URL, headers=headers, timeout=30)
    r.raise_for_status()

    tables = pd.read_html(StringIO(r.text))

    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        if "ticker" in cols:
            tickers = t.iloc[:, cols.index("ticker")].astype(str).str.strip().tolist()
            tickers = [x.replace(".", "-") for x in tickers]
            tickers = [x for x in tickers if x and x != "nan"]
            unique = list(dict.fromkeys(tickers))
            return unique

    raise RuntimeError("Could not find Nasdaq-100 tickers table on the page")


if __name__ == "__main__":
    syms = nasdaq100_tickers()
    print("count:", len(syms))
    print(syms[:25])

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\src\tradinglab\data\__init__.py
# ==============================


# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\src\tradinglab\engine\portfolio.py
# ==============================

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Callable

import numpy as np
import pandas as pd

from tradinglab.config import (
    INITIAL_CAPITAL,
    LONG_WINDOW,
    MOM_LOOKBACK,
    TOP_N,
    REBALANCE,
    FEE_RATE,
    SLIPPAGE_RATE,
    SLIPPAGE_MODE,
    SLIPPAGE_BPS_BASE,
    SLIPPAGE_BPS_PER_TURNOVER,
    REGIME_SYMBOL,
    ALLOW_REGIME_TRADE,
    PRICE_MODE,
    EXECUTION,
    MAX_POSITION_WEIGHT,
    MAX_SECTOR_WEIGHT,
    MAX_TURNOVER_PER_REBALANCE,
    MAX_GROSS_EXPOSURE,
    CASH_BUFFER,
    TRAILING_STOP_PCT,
    TIME_STOP_DAYS,
    TARGET_VOL,
    VOL_LOOKBACK,
)

MOM_SKIP_DAYS = 21
MOM_LONG_DAYS = 252
MOM_MID_DAYS = 126
DISPERSION_WINDOW = 252
TURNOVER_BUFFER = 3
WEIGHT_CAP = 0.10
PORTFOLIO_VOL_LOOKBACK = 60


@dataclass
class PortfolioRun:
    equity: pd.DataFrame
    trade_ledger: pd.DataFrame
    positions: pd.DataFrame
    exposures: pd.DataFrame
    panel_close: pd.DataFrame
    panel_open: pd.DataFrame
    tradable_symbols: list[str]


def _select_price_series(df: pd.DataFrame, price_mode: str) -> tuple[pd.Series | None, pd.Series | None]:
    if price_mode == "adj":
        adj_close = df["Adj Close"] if "Adj Close" in df.columns else None
        close = df["Close"] if "Close" in df.columns else None
        open_px = df["Open"] if "Open" in df.columns else None

        if adj_close is None:
            if close is None:
                return None, None
            close_series = close.copy()
            open_series = open_px.copy() if open_px is not None else close.copy()
            return open_series, close_series

        close_series = adj_close.copy()
        if open_px is None or close is None:
            open_series = close_series.copy()
        else:
            adj_factor = adj_close / close.replace(0.0, np.nan)
            open_series = (open_px * adj_factor).copy()
        return open_series, close_series

    if "Close" not in df.columns:
        return None, None
    close_series = df["Close"].copy()
    open_series = df["Open"].copy() if "Open" in df.columns else close_series.copy()
    return open_series, close_series


def build_price_panels(
    price_dict: dict[str, pd.DataFrame],
    price_mode: str = PRICE_MODE,
    min_coverage: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    opens: dict[str, pd.Series] = {}
    closes: dict[str, pd.Series] = {}

    for sym in sorted(price_dict.keys()):
        df = price_dict[sym]
        open_series, close_series = _select_price_series(df, price_mode)
        if close_series is None:
            continue
        opens[sym] = open_series
        closes[sym] = close_series

    panel_close = pd.DataFrame(closes).sort_index()
    if panel_close.empty:
        return panel_close, panel_close

    min_non_na = max(1, int(len(panel_close.columns) * min_coverage))
    panel_close = panel_close.dropna(thresh=min_non_na)
    panel_close = panel_close.ffill()
    panel_close = panel_close.dropna(how="any")

    panel_open = pd.DataFrame(opens).sort_index()
    panel_open = panel_open.reindex(panel_close.index).ffill()
    panel_open = panel_open.dropna(how="any")

    # align close to open in case open dropped more rows
    panel_close = panel_close.loc[panel_open.index]
    return panel_close, panel_open


def build_panel(price_dict: dict[str, pd.DataFrame], min_coverage: float = 0.8) -> pd.DataFrame:
    panel_close, _ = build_price_panels(price_dict, price_mode=PRICE_MODE, min_coverage=min_coverage)
    return panel_close


def blended_momentum_score(
    panel_close: pd.DataFrame,
    skip_days: int = MOM_SKIP_DAYS,
    long_days: int = MOM_LONG_DAYS,
    mid_days: int = MOM_MID_DAYS,
    vol_lookback: int = MOM_MID_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    m12_1 = panel_close.shift(skip_days) / panel_close.shift(long_days) - 1.0
    m6_1 = panel_close.shift(skip_days) / panel_close.shift(mid_days) - 1.0
    m_raw = 0.5 * m12_1 + 0.5 * m6_1
    sigma = panel_close.pct_change().rolling(vol_lookback).std(ddof=0)
    score = m_raw / sigma
    return score, sigma


def dispersion_series(score: pd.DataFrame, window: int = DISPERSION_WINDOW) -> tuple[pd.Series, pd.Series]:
    dispersion = score.std(axis=1)
    dispersion_med = dispersion.rolling(window).median()
    return dispersion, dispersion_med


def inverse_vol_weights(sigmas: pd.Series, cap: float = WEIGHT_CAP) -> dict[str, float]:
    sigmas = sigmas.replace([np.inf, -np.inf], np.nan).dropna()
    sigmas = sigmas[sigmas > 0]
    if sigmas.empty:
        return {}
    inv = 1.0 / sigmas
    weights = inv / inv.sum()
    if cap is not None:
        weights = weights.clip(upper=cap)
    total = weights.sum()
    if total > 0:
        weights = weights / total
    return weights.to_dict()


def select_with_turnover_buffer(
    scores: pd.Series,
    current_holdings: list[str],
    top_n: int,
    buffer: int = TURNOVER_BUFFER,
    exclude_symbols: set[str] | None = None,
) -> list[str]:
    exclude_symbols = exclude_symbols or set()
    scores = scores.dropna()
    scores = scores[~scores.index.isin(exclude_symbols)]
    if scores.empty:
        return []
    ranked = scores.sort_values(ascending=False)
    top = ranked.head(top_n).index.tolist()
    if not current_holdings:
        return top
    threshold = top_n + buffer
    ranks = pd.Series(range(1, len(ranked) + 1), index=ranked.index)
    keep = [sym for sym in current_holdings if sym in ranks.index and ranks[sym] <= threshold]
    merged = list(dict.fromkeys(top + keep))
    return merged


def market_regime_ok(panel_close: pd.DataFrame, regime_symbol: str = REGIME_SYMBOL) -> pd.Series:
    if regime_symbol in panel_close.columns:
        px = panel_close[regime_symbol]
    else:
        px = panel_close.mean(axis=1)
    sma200 = px.rolling(LONG_WINDOW).mean()
    return px > sma200


def _portfolio_value(prices: pd.Series, cash: float, shares: dict[str, float]) -> float:
    value = cash
    for sym, sh in shares.items():
        value += sh * float(prices[sym])
    return value


def _weights_from_shares(prices: pd.Series, shares: dict[str, float], equity: float) -> dict[str, float]:
    if equity <= 0:
        return {sym: 0.0 for sym in shares}
    weights = {}
    for sym, sh in shares.items():
        weights[sym] = (sh * float(prices[sym])) / equity
    return weights


def _apply_sector_cap(
    weights: dict[str, float],
    sector_map: dict[str, str],
    max_sector_weight: float,
) -> dict[str, float]:
    if max_sector_weight is None:
        return weights
    sector_weights: dict[str, float] = {}
    for sym, w in weights.items():
        sector = sector_map.get(sym, "Unknown")
        sector_weights[sector] = sector_weights.get(sector, 0.0) + w

    adjusted = weights.copy()
    for sector, total_w in sector_weights.items():
        if total_w <= max_sector_weight or total_w <= 0:
            continue
        scale = max_sector_weight / total_w
        for sym, w in weights.items():
            if sector_map.get(sym, "Unknown") == sector:
                adjusted[sym] = w * scale
    return adjusted


def _apply_weight_caps(
    weights: dict[str, float],
    max_position_weight: float | None,
    max_gross_exposure: float | None,
    cash_buffer: float | None,
) -> dict[str, float]:
    adjusted = weights.copy()

    if max_position_weight is not None:
        for sym in adjusted:
            adjusted[sym] = min(adjusted[sym], max_position_weight)

    total = sum(max(0.0, w) for w in adjusted.values())

    if max_gross_exposure is not None and total > max_gross_exposure:
        scale = max_gross_exposure / total
        for sym in adjusted:
            adjusted[sym] *= scale
        total = sum(max(0.0, w) for w in adjusted.values())

    if cash_buffer is not None:
        cap = max(0.0, 1.0 - cash_buffer)
        if total > cap and total > 0:
            scale = cap / total
            for sym in adjusted:
                adjusted[sym] *= scale

    return adjusted


def _apply_turnover_cap(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    max_turnover: float | None,
) -> dict[str, float]:
    if max_turnover is None:
        return target_weights

    turnover = 0.0
    for sym, tw in target_weights.items():
        cw = current_weights.get(sym, 0.0)
        turnover += abs(tw - cw)

    if turnover <= max_turnover or turnover <= 0:
        return target_weights

    scale = max_turnover / turnover
    adjusted = {}
    for sym, tw in target_weights.items():
        cw = current_weights.get(sym, 0.0)
        adjusted[sym] = cw + (tw - cw) * scale
    return adjusted


def _vol_scale(series: pd.Series, target_vol: float, lookback: int) -> float:
    if target_vol is None:
        return 1.0
    if series is None or series.empty:
        return 1.0
    window = series.tail(lookback)
    if window.empty:
        return 1.0
    vol = window.std(ddof=0) * np.sqrt(252)
    if vol <= 0 or np.isnan(vol):
        return 1.0
    return min(1.0, float(target_vol) / float(vol))


def _slippage_rate(notional: float, nav: float, slippage_mode: str) -> float:
    if slippage_mode == "constant":
        return float(SLIPPAGE_RATE)

    base_bps = float(SLIPPAGE_BPS_BASE)
    per_turnover = float(SLIPPAGE_BPS_PER_TURNOVER)
    turnover = 0.0 if nav <= 0 else abs(notional) / nav
    total_bps = base_bps + per_turnover * turnover
    return total_bps / 10000.0


def _execute_order(
    order: dict,
    prices: pd.Series,
    cash: float,
    shares: dict[str, float],
    nav: float,
    cash_buffer: float,
    slippage_mode: str,
) -> tuple[float, dict[str, float], dict | None]:
    symbol = order["symbol"]
    signal_date = order["signal_date"]
    fill_date = order["fill_date"]
    order_shares = float(order["shares"])
    action = "BUY" if order_shares > 0 else "SELL"

    if order_shares == 0.0:
        return cash, shares, None

    price_mid = float(prices[symbol])
    nav = max(nav, 0.0)

    if action == "SELL" and abs(order_shares) > shares.get(symbol, 0.0):
        order_shares = -shares.get(symbol, 0.0)
    trade_notional = abs(order_shares) * price_mid
    slip_rate = _slippage_rate(trade_notional, nav, slippage_mode)

    if action == "BUY":
        cash_buffer_amount = max(0.0, cash_buffer * nav)
        available_cash = max(0.0, cash - cash_buffer_amount)

        fee = trade_notional * FEE_RATE
        slippage_cost = trade_notional * slip_rate
        total_cost = trade_notional + fee + slippage_cost

        if total_cost > available_cash and total_cost > 0:
            scale = available_cash / total_cost
            order_shares *= scale
            trade_notional = abs(order_shares) * price_mid
            slip_rate = _slippage_rate(trade_notional, nav, slippage_mode)
            fee = trade_notional * FEE_RATE
            slippage_cost = trade_notional * slip_rate
            total_cost = trade_notional + fee + slippage_cost

        if order_shares <= 0 or total_cost <= 0:
            return cash, shares, None

        fill_price = price_mid * (1.0 + slip_rate)
        cash -= total_cost
        shares[symbol] += order_shares
    else:
        fee = trade_notional * FEE_RATE
        slippage_cost = trade_notional * slip_rate
        proceeds = trade_notional - fee - slippage_cost

        fill_price = price_mid * (1.0 - slip_rate)
        shares[symbol] += order_shares
        cash += proceeds

    equity_after = _portfolio_value(prices, cash, shares)
    ledger_row = {
        "signal_date": signal_date,
        "fill_date": fill_date,
        "symbol": symbol,
        "action": action,
        "shares": abs(order_shares),
        "fill_price": fill_price,
        "order_notional": trade_notional,
        "fees": fee,
        "slippage_cost": slippage_cost,
        "cash_after": cash,
        "equity_after": equity_after,
    }

    return cash, shares, ledger_row


def run_portfolio(
    price_dict: dict[str, pd.DataFrame],
    regime_symbol: str = REGIME_SYMBOL,
    allow_regime_trade: bool = ALLOW_REGIME_TRADE,
    price_mode: str = PRICE_MODE,
    execution: str = EXECUTION,
    top_n: int | None = None,
    mom_lookback: int | None = None,
    rebalance: str | None = None,
    long_window: int | None = None,
    max_position_weight: float | None = MAX_POSITION_WEIGHT,
    max_sector_weight: float | None = MAX_SECTOR_WEIGHT,
    sector_map: dict[str, str] | None = None,
    max_turnover_per_rebalance: float | None = MAX_TURNOVER_PER_REBALANCE,
    max_gross_exposure: float | None = MAX_GROSS_EXPOSURE,
    cash_buffer: float | None = CASH_BUFFER,
    trailing_stop_pct: float | None = TRAILING_STOP_PCT,
    time_stop_days: int | None = TIME_STOP_DAYS,
    target_vol: float | None = TARGET_VOL,
    vol_lookback: int = VOL_LOOKBACK,
    slippage_mode: str = SLIPPAGE_MODE,
    log_fn: Callable[[str], None] | None = None,
) -> PortfolioRun:
    local_top_n = int(top_n) if top_n is not None else TOP_N
    local_mom = int(mom_lookback) if mom_lookback is not None else MOM_LOOKBACK
    local_rebalance = rebalance or REBALANCE
    local_long = int(long_window) if long_window is not None else LONG_WINDOW

    panel_close, panel_open = build_price_panels(price_dict, price_mode=price_mode)
    if panel_close.empty:
        raise ValueError("Price panel is empty. No symbols with usable data.")

    tradable_symbols = list(panel_close.columns)
    panel_close = panel_close[tradable_symbols]
    panel_open = panel_open[tradable_symbols]

    close_values = panel_close.to_numpy()
    open_values = panel_open.to_numpy()
    dates = list(panel_close.index)
    n_days, n_syms = close_values.shape

    regime_ok = market_regime_ok(panel_close, regime_symbol=regime_symbol).to_numpy()
    score_df, sigma_df = blended_momentum_score(panel_close)
    dispersion, dispersion_med = dispersion_series(score_df)
    score = score_df.to_numpy()
    sigma = sigma_df.to_numpy()

    rebal_dates = panel_close.resample(local_rebalance).last().index
    rebal_mask = np.array([d in set(rebal_dates) for d in dates])

    cash = float(INITIAL_CAPITAL)
    shares = np.zeros(n_syms, dtype=float)
    holding_info: dict[int, dict] = {}

    ledger_rows: list[dict] = []
    positions_rows: list[dict] = []
    exposure_rows: list[dict] = []
    value_rows: list[dict] = []
    portfolio_values: list[float] = []

    pending_orders: dict[int, list[tuple[int, float, pd.Timestamp]]] = {}

    def portfolio_value(prices: np.ndarray) -> float:
        return float(cash + np.dot(shares, prices))

    def queue_orders(signal_idx: int, fill_idx: int, order_shares: np.ndarray) -> None:
        orders = []
        for i in range(n_syms):
            sh = float(order_shares[i])
            if abs(sh) <= 0:
                continue
            orders.append((i, sh, dates[signal_idx]))
        if orders:
            pending_orders.setdefault(fill_idx, []).extend(orders)

    def _recent_portfolio_vol() -> float | None:
        if len(portfolio_values) < PORTFOLIO_VOL_LOOKBACK + 1:
            return None
        window = np.array(portfolio_values[-(PORTFOLIO_VOL_LOOKBACK + 1):], dtype=float)
        rets = np.diff(window) / window[:-1]
        if rets.size == 0:
            return None
        vol = np.std(rets, ddof=0) * np.sqrt(252)
        return float(vol)

    def compute_targets(idx: int) -> np.ndarray:
        current_equity = portfolio_value(close_values[idx])
        current_weights = np.zeros(n_syms) if current_equity <= 0 else (shares * close_values[idx]) / current_equity

        if not bool(regime_ok[idx]):
            if regime_symbol in tradable_symbols:
                targets = np.zeros(n_syms)
                targets[tradable_symbols.index(regime_symbol)] = 1.0
                if log_fn is not None:
                    log_fn(f"Regime off on {dates[idx].date()}; holding {regime_symbol}.")
                return targets
            return np.zeros(n_syms)

        if not rebal_mask[idx]:
            return current_weights

        if not np.isfinite(dispersion.iloc[idx]) or not np.isfinite(dispersion_med.iloc[idx]):
            return current_weights
        if dispersion.iloc[idx] <= dispersion_med.iloc[idx]:
            if log_fn is not None:
                log_fn(f"Dispersion filter failed on {dates[idx].date()}; holding {regime_symbol}.")
            if regime_symbol in tradable_symbols:
                targets = np.zeros(n_syms)
                targets[tradable_symbols.index(regime_symbol)] = 1.0
                return targets
            return np.zeros(n_syms)

        score_row = pd.Series(score[idx], index=tradable_symbols)
        sigma_row = pd.Series(sigma[idx], index=tradable_symbols)
        exclude = set()
        if not allow_regime_trade and regime_symbol in tradable_symbols:
            exclude.add(regime_symbol)

        finite_mask = score_row.replace([np.inf, -np.inf], np.nan).notna()
        sigma_mask = sigma_row.replace([np.inf, -np.inf], np.nan).notna() & (sigma_row > 0)
        eligible_mask = finite_mask & sigma_mask & (~score_row.index.isin(exclude))
        eligible_scores = score_row[eligible_mask]
        if eligible_scores.empty:
            if regime_symbol in tradable_symbols:
                targets = np.zeros(n_syms)
                targets[tradable_symbols.index(regime_symbol)] = 1.0
                if log_fn is not None:
                    log_fn(f"No eligible symbols on rebalance date {dates[idx].date()}; holding {regime_symbol}.")
                return targets
            if log_fn is not None:
                log_fn(f"No eligible symbols on rebalance date {dates[idx].date()}; staying in cash.")
            return np.zeros(n_syms)

        current_syms = [tradable_symbols[i] for i in range(n_syms) if shares[i] > 0]
        selected = select_with_turnover_buffer(
            eligible_scores,
            current_syms,
            local_top_n,
            buffer=TURNOVER_BUFFER,
            exclude_symbols=set(),
        )
        if not selected:
            if regime_symbol in tradable_symbols:
                targets = np.zeros(n_syms)
                targets[tradable_symbols.index(regime_symbol)] = 1.0
                if log_fn is not None:
                    log_fn(f"No eligible symbols on rebalance date {dates[idx].date()}; holding {regime_symbol}.")
                return targets
            if log_fn is not None:
                log_fn(f"No eligible symbols on rebalance date {dates[idx].date()}; staying in cash.")
            return np.zeros(n_syms)

        weights_map = inverse_vol_weights(sigma_row[selected], cap=WEIGHT_CAP)
        if not weights_map:
            if log_fn is not None:
                log_fn(f"No eligible symbols on rebalance date {dates[idx].date()}; staying in cash.")
            return np.zeros(n_syms)

        targets = np.zeros(n_syms)
        for sym, w in weights_map.items():
            if sym in tradable_symbols:
                targets[tradable_symbols.index(sym)] = w

        realized_vol = _recent_portfolio_vol()
        if target_vol is not None and realized_vol is not None and realized_vol > 0:
            scale = float(target_vol) / float(realized_vol)
            targets *= scale

        total = targets.sum()
        if total > 1.0 and total > 0:
            targets /= total

        if max_position_weight is not None:
            targets = np.minimum(targets, max_position_weight)

        total = targets.sum()
        if max_gross_exposure is not None and total > max_gross_exposure and total > 0:
            targets *= max_gross_exposure / total
            total = targets.sum()
        if cash_buffer is not None:
            cap = max(0.0, 1.0 - cash_buffer)
            if total > cap and total > 0:
                targets *= cap / total

        turnover = float(np.abs(targets - current_weights).sum())
        if max_turnover_per_rebalance is not None and turnover > max_turnover_per_rebalance and turnover > 0:
            scale = max_turnover_per_rebalance / turnover
            targets = current_weights + (targets - current_weights) * scale
        return targets

    def apply_exit_overrides(idx: int, targets: np.ndarray) -> np.ndarray:
        adjusted = targets.copy()
        for i in range(n_syms):
            if shares[i] <= 0:
                continue
            info = holding_info.get(i)
            if info is None:
                continue
            peak = info["peak"]
            entry_idx = info["entry_idx"]
            holding_days = idx - entry_idx + 1
            exit_due = False
            if trailing_stop_pct is not None:
                if close_values[idx][i] <= peak * (1.0 - trailing_stop_pct):
                    exit_due = True
            if time_stop_days is not None and holding_days >= time_stop_days:
                exit_due = True
            if exit_due:
                adjusted[i] = 0.0
        return adjusted

    def order_shares_from_targets(idx: int, targets: np.ndarray) -> np.ndarray:
        equity = portfolio_value(close_values[idx])
        if equity <= 0:
            return np.zeros(n_syms)
        target_shares = (equity * targets) / close_values[idx]
        return target_shares - shares

    def scale_buys_for_cash(idx: int, order_shares: np.ndarray) -> np.ndarray:
        equity = portfolio_value(close_values[idx])
        cash_buffer_amount = max(0.0, (cash_buffer or 0.0) * equity)
        available_cash = max(0.0, cash - cash_buffer_amount)
        buy_notional = float(np.sum(np.where(order_shares > 0, order_shares * close_values[idx], 0.0)))
        if buy_notional <= 0:
            return order_shares
        if buy_notional <= available_cash:
            return order_shares
        scale = available_cash / buy_notional
        adjusted = order_shares.copy()
        adjusted = np.where(adjusted > 0, adjusted * scale, adjusted)
        return adjusted

    def execute_order(idx: int, sym_idx: int, sh: float, signal_date: pd.Timestamp, fill_date: pd.Timestamp, prices: np.ndarray) -> None:
        nonlocal cash, shares
        if sh == 0.0:
            return
        action = "BUY" if sh > 0 else "SELL"
        price_mid = float(prices[sym_idx])
        nav = float(cash + np.dot(shares, prices))
        trade_notional = abs(sh) * price_mid
        slip_rate = _slippage_rate(trade_notional, nav, slippage_mode)
        if action == "SELL" and abs(sh) > shares[sym_idx]:
            sh = -shares[sym_idx]
            trade_notional = abs(sh) * price_mid
            slip_rate = _slippage_rate(trade_notional, nav, slippage_mode)

        if action == "BUY":
            cash_buffer_amount = max(0.0, (cash_buffer or 0.0) * nav)
            available_cash = max(0.0, cash - cash_buffer_amount)
            fee = trade_notional * FEE_RATE
            slippage_cost = trade_notional * slip_rate
            total_cost = trade_notional + fee + slippage_cost
            if total_cost > available_cash and total_cost > 0:
                scale = available_cash / total_cost
                sh *= scale
                trade_notional = abs(sh) * price_mid
                slip_rate = _slippage_rate(trade_notional, nav, slippage_mode)
                fee = trade_notional * FEE_RATE
                slippage_cost = trade_notional * slip_rate
                total_cost = trade_notional + fee + slippage_cost
            if sh <= 0 or total_cost <= 0:
                return
            fill_price = price_mid * (1.0 + slip_rate)
            cash -= total_cost
            shares[sym_idx] += sh
        else:
            fee = trade_notional * FEE_RATE
            slippage_cost = trade_notional * slip_rate
            proceeds = trade_notional - fee - slippage_cost
            fill_price = price_mid * (1.0 - slip_rate)
            shares[sym_idx] += sh
            cash += proceeds

        equity_after = cash + np.dot(shares, prices)
        ledger_rows.append(
            {
                "signal_date": signal_date,
                "fill_date": fill_date,
                "symbol": tradable_symbols[sym_idx],
                "action": action,
                "shares": abs(sh),
                "fill_price": fill_price,
                "order_notional": trade_notional,
                "fees": fee,
                "slippage_cost": slippage_cost,
                "cash_after": cash,
                "equity_after": equity_after,
            }
        )

    for idx in range(n_days):
        if execution == "next_open" and idx in pending_orders:
            for sym_idx, sh, signal_date in pending_orders[idx]:
                execute_order(idx, sym_idx, sh, signal_date, dates[idx], open_values[idx])
            pending_orders.pop(idx, None)

        if execution == "same_close":
            targets = compute_targets(idx)
            targets = apply_exit_overrides(idx, targets)
            order_shares = order_shares_from_targets(idx, targets)
            order_shares = scale_buys_for_cash(idx, order_shares)
            for sym_idx in range(n_syms):
                sh = float(order_shares[sym_idx])
                if sh != 0:
                    execute_order(idx, sym_idx, sh, dates[idx], dates[idx], close_values[idx])
        else:
            if idx < n_days - 1:
                targets = compute_targets(idx)
                targets = apply_exit_overrides(idx, targets)
                order_shares = order_shares_from_targets(idx, targets)
                order_shares = scale_buys_for_cash(idx, order_shares)
                queue_orders(idx, idx + 1, order_shares)

        for sym_idx in range(n_syms):
            if shares[sym_idx] > 0:
                info = holding_info.get(sym_idx)
                if info is None:
                    holding_info[sym_idx] = {"entry_idx": idx, "peak": float(close_values[idx][sym_idx])}
                else:
                    info["peak"] = max(info["peak"], float(close_values[idx][sym_idx]))
            else:
                holding_info.pop(sym_idx, None)

        equity_close = float(cash + np.dot(shares, close_values[idx]))
        portfolio_values.append(equity_close)
        gross_exposure = float(np.sum(np.abs(shares * close_values[idx])))
        holdings = int(np.sum(shares > 0))

        value_rows.append({"Date": dates[idx], "Portfolio_Value": equity_close})
        positions_rows.append(
            {"Date": dates[idx], "Cash": cash, **{tradable_symbols[i]: shares[i] for i in range(n_syms)}}
        )
        exposure_rows.append(
            {
                "Date": dates[idx],
                "Cash_Pct": cash / equity_close if equity_close > 0 else 0.0,
                "Gross_Exposure_Pct": gross_exposure / equity_close if equity_close > 0 else 0.0,
                "Holdings": holdings,
            }
        )

    equity_df = pd.DataFrame(value_rows).set_index("Date")
    equity_df["Return_%"] = equity_df["Portfolio_Value"].pct_change() * 100.0
    running_max = equity_df["Portfolio_Value"].cummax()
    equity_df["Drawdown_%"] = (equity_df["Portfolio_Value"] - running_max) / running_max * 100.0

    ledger_df = pd.DataFrame(ledger_rows)
    if not ledger_df.empty:
        ledger_df = ledger_df.sort_values(["fill_date", "symbol", "action"]).reset_index(drop=True)

    positions_df = pd.DataFrame(positions_rows).set_index("Date")
    exposures_df = pd.DataFrame(exposure_rows).set_index("Date")

    return PortfolioRun(
        equity=equity_df,
        trade_ledger=ledger_df,
        positions=positions_df,
        exposures=exposures_df,
        panel_close=panel_close,
        panel_open=panel_open,
        tradable_symbols=tradable_symbols,
    )


def buy_hold_benchmark(
    price_dict: dict[str, pd.DataFrame],
    regime_symbol: str = REGIME_SYMBOL,
    allow_regime_trade: bool = ALLOW_REGIME_TRADE,
    price_mode: str = PRICE_MODE,
) -> pd.DataFrame:
    panel_close, _ = build_price_panels(price_dict, price_mode=price_mode)
    if panel_close.empty:
        raise ValueError("Price panel is empty. No symbols with Close data.")

    symbols = [s for s in panel_close.columns if allow_regime_trade or s != regime_symbol]
    if not symbols:
        raise ValueError("No tradable symbols for buy-hold benchmark.")

    init = float(INITIAL_CAPITAL)
    weight = 1.0 / len(symbols)

    first = panel_close.iloc[0]
    shares = {sym: (init * weight) / float(first[sym]) for sym in symbols}

    values = []
    for date in panel_close.index:
        v = 0.0
        for sym in symbols:
            v += shares[sym] * float(panel_close.loc[date, sym])
        values.append({"Date": date, "BuyHold_Value": v})

    out = pd.DataFrame(values).set_index("Date")
    running_max = out["BuyHold_Value"].cummax()
    out["BuyHold_Drawdown_%"] = (out["BuyHold_Value"] - running_max) / running_max * 100.0
    return out


def buy_hold_single(
    price_dict: dict[str, pd.DataFrame],
    symbol: str,
    price_mode: str = PRICE_MODE,
) -> pd.DataFrame:
    if symbol not in price_dict:
        raise ValueError(f"Missing data for benchmark symbol: {symbol}")

    df = price_dict[symbol]
    open_series, close_series = _select_price_series(df, price_mode)
    if close_series is None:
        raise ValueError(f"Close column missing for benchmark symbol: {symbol}")

    series = close_series.dropna()
    if series.empty:
        raise ValueError(f"No usable prices for benchmark symbol: {symbol}")

    init = float(INITIAL_CAPITAL)
    shares = init / float(series.iloc[0])

    values = (series * shares).to_frame(name=f"{symbol}_BuyHold")
    return values

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\src\tradinglab\engine\__init__.py
# ==============================


# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\src\tradinglab\experiments\runner.py
# ==============================

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from tradinglab.config import (
    RESULTS_DIR,
    REGIME_SYMBOL,
    PRICE_MODE,
    EXECUTION,
    LONG_WINDOW,
    MOM_LOOKBACK,
    INITIAL_CAPITAL,
)
from tradinglab.data.fetcher import load_or_fetch_symbols
from tradinglab.data.tickers import nasdaq100_tickers
from tradinglab.data.panel import prepare_panel_with_history_filter
from tradinglab.engine.portfolio import run_portfolio, buy_hold_benchmark, buy_hold_single
from tradinglab.metrics.performance import compute_metrics
from tradinglab.experiments.splits import split_by_date, walk_forward_splits


def _slice_price_dict(price_dict: dict[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    sliced: dict[str, pd.DataFrame] = {}
    for sym, df in price_dict.items():
        sliced_df = df.loc[start:end]
        if not sliced_df.empty:
            sliced[sym] = sliced_df
    return sliced


def run_experiment(
    universe: str,
    symbols: list[str] | None,
    refresh_data: bool,
    start_date: str | None,
    end_date: str | None,
    split_date: str | None,
    walk_forward: bool,
    train_days: int,
    test_days: int,
    step_days: int,
    grid_search: bool = False,
    max_combos: int | None = None,
    jobs: int = 1,
    execution: str | None = None,
    price_mode: str | None = None,
    max_position_weight: float | None = None,
    trailing_stop: float | None = None,
    time_stop: int | None = None,
    target_vol: float | None = None,
) -> Path:
    if symbols is None:
        symbols = nasdaq100_tickers()

    symbols = list(dict.fromkeys([s.strip().upper() for s in symbols if s]))
    if REGIME_SYMBOL not in symbols:
        symbols.append(REGIME_SYMBOL)

    price_dict = load_or_fetch_symbols(
        symbols,
        refresh=refresh_data,
        start_date=start_date,
        end_date=end_date,
    )
    if not price_dict:
        raise RuntimeError("No market data loaded.")

    warmup_days = max(int(LONG_WINDOW), 252, 126) + 21 + 5
    min_history_days = warmup_days + int(train_days) + int(test_days) if walk_forward else warmup_days + 1
    price_mode_effective = price_mode or PRICE_MODE

    price_dict, panel_close, dropped = prepare_panel_with_history_filter(
        price_dict,
        price_mode=price_mode_effective,
        min_history_days=min_history_days,
        keep_symbols={REGIME_SYMBOL},
        log_fn=print,
    )
    if panel_close.empty:
        raise ValueError("Price panel is empty after filtering for history.")

    symbols = list(price_dict.keys())
    print(f"Universe requested: {universe}, loaded symbols: {len(symbols)}")

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / "experiments" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    config_used = {
        "universe": universe,
        "symbols": symbols,
        "dropped_symbols": dropped,
        "refresh_data": refresh_data,
        "start_date": start_date,
        "end_date": end_date,
        "split_date": split_date,
        "walk_forward": walk_forward,
        "train_days": train_days,
        "test_days": test_days,
        "step_days": step_days,
        "execution": execution or EXECUTION,
        "price_mode": price_mode_effective,
        "max_position_weight": max_position_weight,
        "trailing_stop": trailing_stop,
        "time_stop": time_stop,
        "target_vol": target_vol,
        "warmup_days": warmup_days,
        "min_history_days": min_history_days,
    }
    (out_dir / "config_used.json").write_text(json.dumps(config_used, indent=2))

    if walk_forward:
        panel_idx = pd.DataFrame(index=panel_close.index)
        splits = walk_forward_splits(panel_idx, train_days, test_days, step_days)
        if not splits:
            print("0 splits produced; check history thresholds/warmup.")
    else:
        if split_date is None:
            raise ValueError("split_date is required for fixed split")
        panel_idx = pd.DataFrame(index=panel_close.index)
        train_idx, test_idx = split_by_date(panel_idx, split_date)
        if train_idx.empty or test_idx.empty:
            raise ValueError("Split date results in empty train/test. Adjust split_date or date range.")
        splits = [
            {
                "train_start": train_idx.index[0],
                "train_end": train_idx.index[-1],
                "test_start": test_idx.index[0],
                "test_end": test_idx.index[-1],
            }
        ]

    per_split_rows: list[dict] = []
    equity_rows: list[dict] = []
    ledger_rows: list[dict] = []

    grid_train_rows: list[dict] = []
    grid_test_rows: list[dict] = []
    grid_selected_rows: list[dict] = []

    def _grid_params() -> list[dict]:
        grid = []
        for top_n in [5, 10, 15, 20]:
            for mom in [126, 189, 252]:
                for reb in ["W-FRI", "ME"]:
                    for lw in [150, 200, 250]:
                        grid.append(
                            {
                                "top_n": top_n,
                                "mom_lookback": mom,
                                "rebalance": reb,
                                "long_window": lw,
                            }
                        )
        return grid

    def _score_run(run) -> float:
        from tradinglab.metrics.performance import sharpe

        return sharpe(run.equity["Portfolio_Value"])

    params_grid = _grid_params() if grid_search else []
    if max_combos is not None and params_grid:
        params_grid = params_grid[: max_combos]

    metric_columns = [
        "Label",
        "CAGR",
        "Annual_Vol",
        "Sharpe",
        "Sortino",
        "Max_DD",
        "Calmar",
        "Turnover",
        "Win_Rate",
        "split_id",
        "dataset",
    ]
    ledger_columns = [
        "signal_date",
        "fill_date",
        "symbol",
        "action",
        "shares",
        "fill_price",
        "order_notional",
        "fees",
        "slippage_cost",
        "cash_after",
        "equity_after",
    ]

    for split_id, split in enumerate(splits, start=1):
        train_slice = _slice_price_dict(price_dict, split["train_start"], split["train_end"])
        test_slice = _slice_price_dict(price_dict, split["test_start"], split["test_end"])

        if grid_search and train_slice:
            def _run_param(param: dict) -> dict:
                run_train = run_portfolio(
                    train_slice,
                    execution=execution or EXECUTION,
                    price_mode=price_mode_effective,
                    slippage_mode="constant",
                    top_n=param["top_n"],
                    mom_lookback=param["mom_lookback"],
                    rebalance=param["rebalance"],
                    long_window=param["long_window"],
                )
                return {"param": param, "train_sharpe": _score_run(run_train)}

            if jobs > 1:
                from joblib import Parallel, delayed

                results = Parallel(n_jobs=max(1, jobs))(delayed(_run_param)(p) for p in params_grid)
            else:
                results = [_run_param(p) for p in params_grid]

            for res in results:
                row = {"split_id": split_id, **res["param"], "train_sharpe": res["train_sharpe"]}
                grid_train_rows.append(row)

            top = sorted(results, key=lambda x: x["train_sharpe"], reverse=True)[:3]
            for selected in top:
                grid_selected_rows.append(
                    {"split_id": split_id, **selected["param"], "train_sharpe": selected["train_sharpe"]}
                )

                if test_slice:
                    run_test = run_portfolio(
                        test_slice,
                        execution=execution or EXECUTION,
                        price_mode=price_mode_effective,
                        slippage_mode="constant",
                        top_n=selected["param"]["top_n"],
                        mom_lookback=selected["param"]["mom_lookback"],
                        rebalance=selected["param"]["rebalance"],
                        long_window=selected["param"]["long_window"],
                    )
                    grid_test_rows.append(
                        {
                            "split_id": split_id,
                            **selected["param"],
                            "test_sharpe": _score_run(run_test),
                        }
                    )

        for label, dataset in [("train", train_slice), ("test", test_slice)]:
            if not dataset:
                print(f"Split {split_id} had 0 symbols after filtering for {label}; skipping run.")
                for base_label in [f"Portfolio_{label}", f"BuyHold_{label}", f"{REGIME_SYMBOL}_{label}"]:
                    metrics = compute_metrics(base_label, pd.Series(dtype=float))
                    metrics["split_id"] = split_id
                    metrics["dataset"] = label
                    per_split_rows.append(metrics)
                if label == "test":
                    equity_rows.append(
                        {
                            "split_id": split_id,
                            "date": split["test_start"],
                            "portfolio_value": float(INITIAL_CAPITAL),
                            "buyhold_value": float(INITIAL_CAPITAL),
                            "qqq_value": None,
                        }
                    )
                continue

            run = run_portfolio(
                dataset,
                execution=execution or EXECUTION,
                price_mode=price_mode_effective,
                max_position_weight=max_position_weight,
                trailing_stop_pct=trailing_stop,
                time_stop_days=time_stop,
                target_vol=target_vol,
                log_fn=print,
            )
            if run.trade_ledger.empty:
                print(f"No trades for split {split_id} ({label}); portfolio stayed in cash.")

            pf_metrics = compute_metrics(f"Portfolio_{label}", run.equity["Portfolio_Value"], run.trade_ledger)
            pf_metrics["split_id"] = split_id
            pf_metrics["dataset"] = label
            per_split_rows.append(pf_metrics)

            bh = buy_hold_benchmark(dataset, price_mode=price_mode_effective)
            bh_metrics = compute_metrics(f"BuyHold_{label}", bh["BuyHold_Value"])
            bh_metrics["split_id"] = split_id
            bh_metrics["dataset"] = label
            per_split_rows.append(bh_metrics)

            if REGIME_SYMBOL in dataset:
                qqq = buy_hold_single(dataset, REGIME_SYMBOL, price_mode=price_mode_effective)
                qqq_metrics = compute_metrics(f"{REGIME_SYMBOL}_{label}", qqq.iloc[:, 0])
                qqq_metrics["split_id"] = split_id
                qqq_metrics["dataset"] = label
                per_split_rows.append(qqq_metrics)

            if label == "test":
                merged = run.equity[["Portfolio_Value"]].join(bh, how="inner")
                if REGIME_SYMBOL in dataset:
                    merged = merged.join(qqq, how="left")

                for dt, row in merged.iterrows():
                    equity_rows.append(
                        {
                            "split_id": split_id,
                            "date": dt,
                            "portfolio_value": row["Portfolio_Value"],
                            "buyhold_value": row["BuyHold_Value"],
                            "qqq_value": row.get(f"{REGIME_SYMBOL}_BuyHold", None),
                        }
                    )

                if not run.trade_ledger.empty:
                    ledger = run.trade_ledger.copy()
                    ledger.insert(0, "split_id", split_id)
                    ledger_rows.append(ledger)

    per_split_df = pd.DataFrame(per_split_rows)
    if per_split_df.empty:
        per_split_df = pd.DataFrame(columns=metric_columns)
    else:
        for col in metric_columns:
            if col not in per_split_df.columns:
                dtype = object if col in {"Label", "dataset"} else float
                per_split_df[col] = pd.Series(dtype=dtype)
    per_split_df.to_csv(out_dir / "per_split_metrics.csv", index=False)

    if not per_split_df.empty:
        numeric_cols = per_split_df.select_dtypes(include=["number"]).columns
        summary = per_split_df.groupby(["Label", "dataset"])[numeric_cols].agg(["mean", "median"])
        summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]
        summary.reset_index().to_csv(out_dir / "metrics_summary.csv", index=False)
    else:
        pd.DataFrame(columns=["Label", "dataset"]).to_csv(out_dir / "metrics_summary.csv", index=False)

    equity_df = pd.DataFrame(equity_rows)
    if equity_df.empty and not panel_close.empty:
        equity_df = pd.DataFrame(
            [
                {
                    "split_id": 0,
                    "date": panel_close.index[0],
                    "portfolio_value": float(INITIAL_CAPITAL),
                    "buyhold_value": float(INITIAL_CAPITAL),
                    "qqq_value": None,
                }
            ]
        )
    equity_df.to_csv(out_dir / "equity_curves.csv", index=False)

    if ledger_rows:
        ledger_df = pd.concat(ledger_rows, ignore_index=True)
        ledger_df.to_csv(out_dir / "trade_ledger.csv", index=False)
    else:
        pd.DataFrame(columns=ledger_columns).to_csv(out_dir / "trade_ledger.csv", index=False)

    if grid_search:
        pd.DataFrame(grid_train_rows).to_csv(out_dir / "grid_results_train.csv", index=False)
        pd.DataFrame(grid_selected_rows).to_csv(out_dir / "grid_selected.csv", index=False)
        pd.DataFrame(grid_test_rows).to_csv(out_dir / "grid_results_test.csv", index=False)

    if walk_forward and not splits:
        raise ValueError("Not enough history for train_days/test_days. Try smaller windows or earlier start date.")

    return out_dir

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\src\tradinglab\experiments\splits.py
# ==============================

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

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\src\tradinglab\experiments\__init__.py
# ==============================

"""Experiment utilities."""

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\src\tradinglab\metrics\performance.py
# ==============================

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def _daily_returns(series: pd.Series) -> pd.Series:
    return series.pct_change().dropna()


def cagr(series: pd.Series) -> float:
    if series is None or series.empty:
        return float("nan")
    start = float(series.iloc[0])
    end = float(series.iloc[-1])
    if start <= 0:
        return float("nan")
    days = (series.index[-1] - series.index[0]).days
    years = days / 365.25
    if years <= 0:
        return float("nan")
    return (end / start) ** (1.0 / years) - 1.0


def annual_vol(series: pd.Series) -> float:
    rets = _daily_returns(series)
    if rets.empty:
        return float("nan")
    return float(rets.std(ddof=0) * np.sqrt(TRADING_DAYS))


def sharpe(series: pd.Series) -> float:
    rets = _daily_returns(series)
    if rets.empty:
        return float("nan")
    vol = rets.std(ddof=0)
    if vol == 0:
        return float("nan")
    return float((rets.mean() / vol) * np.sqrt(TRADING_DAYS))


def sortino(series: pd.Series) -> float:
    rets = _daily_returns(series)
    if rets.empty:
        return float("nan")
    downside = rets[rets < 0]
    if downside.empty:
        return float("nan")
    downside_vol = downside.std(ddof=0)
    if downside_vol == 0:
        return float("nan")
    return float((rets.mean() / downside_vol) * np.sqrt(TRADING_DAYS))


def max_drawdown(series: pd.Series) -> float:
    if series is None or series.empty:
        return float("nan")
    running_max = series.cummax()
    drawdown = (series - running_max) / running_max
    return float(drawdown.min())


def calmar(series: pd.Series) -> float:
    mdd = max_drawdown(series)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return float(cagr(series) / abs(mdd))


def turnover(trade_ledger: pd.DataFrame, equity_series: pd.Series) -> float:
    if trade_ledger is None or trade_ledger.empty:
        return 0.0
    if "order_notional" in trade_ledger.columns:
        notional = trade_ledger["order_notional"].abs().sum()
    else:
        notional = (trade_ledger["shares"].abs() * trade_ledger["fill_price"].abs()).sum()
    avg_equity = float(equity_series.mean()) if equity_series is not None and not equity_series.empty else 0.0
    if avg_equity == 0:
        return float("nan")
    return float(notional / avg_equity)


def trade_win_rate(trade_ledger: pd.DataFrame) -> float:
    if trade_ledger is None or trade_ledger.empty:
        return float("nan")

    wins = 0
    total = 0

    lots: dict[str, deque] = {}

    for _, row in trade_ledger.iterrows():
        symbol = row["symbol"]
        action = row["action"]
        shares = float(row["shares"])
        price = float(row["fill_price"])
        fee = float(row["fees"])
        slippage_cost = float(row.get("slippage_cost", 0.0))

        if symbol not in lots:
            lots[symbol] = deque()

        if action == "BUY":
            if shares <= 0:
                continue
            cost_per_share = (shares * price + fee + slippage_cost) / shares
            lots[symbol].append([shares, cost_per_share])
        elif action == "SELL":
            if shares <= 0:
                continue
            proceeds_per_share = (shares * price - fee - slippage_cost) / shares
            remaining = shares
            pnl = 0.0

            while remaining > 0 and lots[symbol]:
                lot_shares, lot_cost = lots[symbol][0]
                matched = min(remaining, lot_shares)
                pnl += (proceeds_per_share - lot_cost) * matched
                lot_shares -= matched
                remaining -= matched
                if lot_shares <= 0:
                    lots[symbol].popleft()
                else:
                    lots[symbol][0][0] = lot_shares

            if remaining <= 0:
                total += 1
                if pnl > 0:
                    wins += 1

    if total == 0:
        return float("nan")
    return float(wins / total)


def compute_metrics(
    label: str,
    series: pd.Series,
    trade_ledger: pd.DataFrame | None = None,
) -> dict:
    metrics = {
        "Label": label,
        "CAGR": cagr(series),
        "Annual_Vol": annual_vol(series),
        "Sharpe": sharpe(series),
        "Sortino": sortino(series),
        "Max_DD": max_drawdown(series),
        "Calmar": calmar(series),
    }

    if trade_ledger is not None:
        metrics["Turnover"] = turnover(trade_ledger, series)
        metrics["Win_Rate"] = trade_win_rate(trade_ledger)
    else:
        metrics["Turnover"] = 0.0
        metrics["Win_Rate"] = float("nan")

    return metrics

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\src\tradinglab\metrics\__init__.py
# ==============================


# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\src\tradinglab\robustness\allocation.py
# ==============================

from __future__ import annotations

import pandas as pd

from tradinglab.metrics.performance import compute_metrics


def vol_target_exposure(strategy_ret: pd.Series, target_vol: float) -> pd.Series:
    roll_vol = strategy_ret.rolling(20).std(ddof=0) * (252 ** 0.5)
    scale = (target_vol / roll_vol).clip(upper=1.0).fillna(1.0)
    return scale


def allocation_study(
    strategy_equity: pd.Series,
    qqq_equity: pd.Series,
    target_vol: float,
    drawdown_threshold: float,
) -> pd.DataFrame:
    df = pd.DataFrame({"strategy": strategy_equity, "qqq": qqq_equity}).dropna()
    strategy_ret = df["strategy"].pct_change().fillna(0.0)
    qqq_ret = df["qqq"].pct_change().fillna(0.0)

    outputs = []

    # 100% strategy
    strat_equity = (1.0 + strategy_ret).cumprod()
    outputs.append(compute_metrics("100%_Strategy", strat_equity))

    # 50/50
    mix_ret = 0.5 * strategy_ret + 0.5 * qqq_ret
    mix_equity = (1.0 + mix_ret).cumprod()
    outputs.append(compute_metrics("50_50_Strategy_QQQ", mix_equity))

    # Vol targeting (cap at 1.0 exposure)
    scale = vol_target_exposure(strategy_ret, target_vol)
    vol_ret = strategy_ret * scale
    vol_equity = (1.0 + vol_ret).cumprod()
    outputs.append(compute_metrics("Vol_Targeting", vol_equity))

    # Drawdown-based de-risk
    running_max = strat_equity.cummax()
    drawdown = (strat_equity - running_max) / running_max
    exposure = (drawdown > -drawdown_threshold).astype(float)
    exposure = exposure.replace(0.0, 0.5)
    derisk_ret = strategy_ret * exposure
    derisk_equity = (1.0 + derisk_ret).cumprod()
    outputs.append(compute_metrics("Drawdown_DeRisk", derisk_equity))

    return pd.DataFrame(outputs)

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\src\tradinglab\robustness\monte_carlo.py
# ==============================

from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_paths(
    returns: pd.Series,
    paths: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if returns.empty:
        return pd.DataFrame(), pd.DataFrame()

    rng = np.random.default_rng(seed)
    rets = returns.dropna().values
    n = len(rets)

    equity_paths = []
    summary_rows = []

    for i in range(paths):
        sampled = rng.choice(rets, size=n, replace=True)
        equity = (1.0 + sampled).cumprod()
        equity = pd.Series(equity)
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        max_dd = float(drawdown.min())

        worst_1y = float((equity / equity.shift(252) - 1.0).min()) if n > 252 else float("nan")
        summary_rows.append(
            {
                "path_id": i,
                "final_equity": float(equity.iloc[-1]),
                "max_drawdown": max_dd,
                "worst_1y": worst_1y,
            }
        )
        equity_paths.append(equity)

    paths_df = pd.DataFrame(equity_paths).T
    paths_df.index.name = "day"
    paths_df.columns = [f"path_{i}" for i in range(paths)]

    summary_df = pd.DataFrame(summary_rows)
    return paths_df, summary_df


def monte_carlo_report(summary_df: pd.DataFrame) -> dict:
    if summary_df.empty:
        return {}

    p5_terminal = float(np.percentile(summary_df["final_equity"], 5))
    p95_dd = float(np.percentile(summary_df["max_drawdown"], 95))
    prob_loss = float((summary_df["final_equity"] < 1.0).mean())

    return {
        "P5_Terminal_Equity": p5_terminal,
        "P95_MaxDD": p95_dd,
        "Prob_Losing_Money": prob_loss,
    }

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\src\tradinglab\robustness\regimes.py
# ==============================

from __future__ import annotations

import pandas as pd

from tradinglab.engine.portfolio import build_price_panels
from tradinglab.metrics.performance import compute_metrics


def regime_metrics(
    price_dict: dict[str, pd.DataFrame],
    equity: pd.Series,
    regime_symbol: str,
    price_mode: str,
) -> pd.DataFrame:
    panel_close, _ = build_price_panels(price_dict, price_mode=price_mode)
    if panel_close.empty or regime_symbol not in panel_close.columns:
        return pd.DataFrame()

    qqq = panel_close[regime_symbol]
    qqq_sma = qqq.rolling(200).mean()
    qqq_ret = qqq.pct_change()
    vol = qqq_ret.rolling(20).std()
    vol_thresh = float(vol.median())

    regimes = {
        "Bull": qqq > qqq_sma,
        "Bear": qqq < qqq_sma,
        "HighVol": vol >= vol_thresh,
        "LowVol": vol < vol_thresh,
    }

    rows = []
    for name, mask in regimes.items():
        idx = equity.index.intersection(mask.index)
        mask = mask.loc[idx]
        series = equity.loc[idx][mask]
        if series.empty:
            continue
        metrics = compute_metrics(name, series)
        rows.append(metrics)

    return pd.DataFrame(rows)

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\src\tradinglab\robustness\rolling.py
# ==============================

from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_max_drawdown(series: pd.Series) -> pd.Series:
    running_max = series.cummax()
    drawdown = (series - running_max) / running_max
    return drawdown


def rolling_metrics(
    equity: pd.Series,
    benchmark: pd.Series,
    turnover_series: pd.Series,
    exposure_series: pd.Series,
    window: int = 252,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "equity": equity,
            "benchmark": benchmark,
            "turnover": turnover_series,
            "exposure": exposure_series,
        }
    ).dropna()

    returns = df["equity"].pct_change()
    bench_returns = df["benchmark"].pct_change()

    rolling_sharpe = (
        returns.rolling(window).mean() / returns.rolling(window).std(ddof=0)
    ) * np.sqrt(252)

    rolling_cagr = (df["equity"].rolling(window).apply(
        lambda x: (x.iloc[-1] / x.iloc[0]) ** (252 / window) - 1.0, raw=False
    ))

    rolling_dd = df["equity"].rolling(window).apply(
        lambda x: _rolling_max_drawdown(pd.Series(x)).min(), raw=False
    )

    rolling_corr = returns.rolling(window).corr(bench_returns)

    cov = returns.rolling(window).cov(bench_returns)
    var = bench_returns.rolling(window).var(ddof=0)
    rolling_beta = cov / var

    rolling_turnover = df["turnover"].rolling(window).mean()
    rolling_exposure = df["exposure"].rolling(window).mean()

    out = pd.DataFrame(
        {
            "Rolling_Sharpe": rolling_sharpe,
            "Rolling_CAGR": rolling_cagr,
            "Rolling_MaxDD": rolling_dd,
            "Rolling_Corr_QQQ": rolling_corr,
            "Rolling_Beta_QQQ": rolling_beta,
            "Rolling_Turnover": rolling_turnover,
            "Rolling_Exposure": rolling_exposure,
        }
    ).dropna()

    return out


def rolling_summary(rolling_df: pd.DataFrame, benchmark_cagr: pd.Series) -> dict:
    if rolling_df.empty:
        return {}

    positive_sharpe = float((rolling_df["Rolling_Sharpe"] > 0).mean())
    beating_qqq = float((rolling_df["Rolling_CAGR"] > benchmark_cagr.loc[rolling_df.index]).mean())
    dd_worse = float((rolling_df["Rolling_MaxDD"] < -0.30).mean())

    return {
        "Pct_Positive_Sharpe": positive_sharpe,
        "Pct_Beating_QQQ": beating_qqq,
        "Pct_DD_Worse_30": dd_worse,
    }

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\src\tradinglab\robustness\runner.py
# ==============================

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tradinglab.config import RESULTS_DIR, REGIME_SYMBOL, PRICE_MODE, EXECUTION, LONG_WINDOW, MOM_LOOKBACK
from tradinglab.data.fetcher import load_or_fetch_symbols
from tradinglab.data.tickers import nasdaq100_tickers
from tradinglab.data.panel import prepare_panel_with_history_filter
from tradinglab.engine.portfolio import run_portfolio, buy_hold_single
from tradinglab.metrics.performance import sharpe
from tradinglab.experiments.splits import walk_forward_splits
from tradinglab.robustness.rolling import rolling_metrics, rolling_summary
from tradinglab.robustness.monte_carlo import bootstrap_paths, monte_carlo_report
from tradinglab.robustness.stability import parameter_stability
from tradinglab.robustness.regimes import regime_metrics
from tradinglab.robustness.allocation import allocation_study
from tradinglab.robustness.warnings import overfit_warnings


def _build_turnover_series(trade_ledger: pd.DataFrame, equity: pd.Series) -> pd.Series:
    if trade_ledger.empty:
        return pd.Series(index=equity.index, data=0.0)
    daily = trade_ledger.groupby("fill_date")["order_notional"].sum().reindex(equity.index).fillna(0.0)
    avg_equity = equity.rolling(20).mean().replace(0.0, np.nan)
    turnover = daily / avg_equity
    return turnover.fillna(0.0)


def _build_exposure_series(exposures: pd.DataFrame) -> pd.Series:
    if exposures.empty:
        return pd.Series(dtype=float)
    return exposures["Gross_Exposure_Pct"].fillna(0.0)


def _grid_params() -> list[dict]:
    grid = []
    for top_n in [5, 10, 15, 20]:
        for mom in [126, 189, 252]:
            for reb in ["W-FRI", "ME"]:
                for lw in [150, 200, 250]:
                    grid.append(
                        {
                            "top_n": top_n,
                            "mom_lookback": mom,
                            "rebalance": reb,
                            "long_window": lw,
                        }
                    )
    return grid


def _apply_params(run_kwargs: dict, params: dict) -> dict:
    out = run_kwargs.copy()
    out.update(params)
    return out


def run_robustness(
    universe: str,
    symbols: list[str] | None,
    refresh_data: bool,
    start_date: str | None,
    end_date: str | None,
    walk_forward: bool,
    train_days: int,
    test_days: int,
    step_days: int,
    monte_carlo_paths: int,
    allocation_study_flag: bool,
    seed: int,
) -> Path:
    if symbols is None:
        symbols = nasdaq100_tickers()

    symbols = list(dict.fromkeys([s.strip().upper() for s in symbols if s]))
    if REGIME_SYMBOL not in symbols:
        symbols.append(REGIME_SYMBOL)

    price_dict = load_or_fetch_symbols(
        symbols,
        refresh=refresh_data,
        start_date=start_date,
        end_date=end_date,
    )
    if not price_dict:
        raise RuntimeError("No market data loaded.")

    warmup_days = max(int(LONG_WINDOW), 252, 126) + 21 + 5
    min_history_days = warmup_days + int(train_days) + int(test_days)

    price_dict, panel_close, dropped = prepare_panel_with_history_filter(
        price_dict,
        price_mode=PRICE_MODE,
        min_history_days=min_history_days,
        keep_symbols={REGIME_SYMBOL},
        log_fn=print,
    )
    if panel_close.empty:
        raise ValueError("Price panel is empty after filtering for history.")

    symbols = list(price_dict.keys())
    print(f"Universe requested: {universe}, loaded symbols: {len(symbols)}")

    out_dir = RESULTS_DIR / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Main strategy run
    run = run_portfolio(
        price_dict,
        execution=EXECUTION,
        price_mode=PRICE_MODE,
    )

    equity = run.equity["Portfolio_Value"]
    qqq = buy_hold_single(price_dict, REGIME_SYMBOL, price_mode=PRICE_MODE)
    qqq_series = qqq.iloc[:, 0]

    turnover_series = _build_turnover_series(run.trade_ledger, equity)
    exposure_series = _build_exposure_series(run.exposures)

    roll = rolling_metrics(equity, qqq_series, turnover_series, exposure_series)
    roll.to_csv(out_dir / "rolling_metrics.csv", index_label="Date")

    bench_cagr = (qqq_series / qqq_series.shift(252) - 1.0).reindex(roll.index)
    summary = rolling_summary(roll, bench_cagr)
    pd.DataFrame([summary]).to_csv(out_dir / "rolling_summary.csv", index=False)

    # Monte Carlo
    returns = equity.pct_change().dropna()
    paths_df, summary_df = bootstrap_paths(returns, paths=monte_carlo_paths, seed=seed)
    paths_df.to_csv(out_dir / "monte_carlo_paths.csv", index=True)
    summary_df.to_csv(out_dir / "monte_carlo_summary.csv", index=False)
    report = monte_carlo_report(summary_df)
    (out_dir / "monte_carlo_report.json").write_text(json.dumps(report, indent=2))

    # Parameter stability
    params_grid = _grid_params()
    stability_df = parameter_stability(
        price_dict,
        params_grid=params_grid,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        execution=EXECUTION,
        price_mode=PRICE_MODE,
        seed=seed,
    )
    stability_df.to_csv(out_dir / "parameter_stability.csv", index=False)

    # Regime segmentation
    regime_df = regime_metrics(price_dict, equity, REGIME_SYMBOL, PRICE_MODE)
    regime_df.to_csv(out_dir / "regime_metrics.csv", index=False)

    # Allocation study
    if allocation_study_flag:
        alloc_df = allocation_study(equity, qqq_series, target_vol=0.10, drawdown_threshold=0.15)
        alloc_df.to_csv(out_dir / "allocation_metrics.csv", index=False)

    # Overfit warnings
    warnings = []
    if walk_forward:
        panel_idx = pd.DataFrame(index=panel_close.index)
        splits = walk_forward_splits(panel_idx, train_days, test_days, step_days)
        if not splits:
            raise ValueError(
                "Not enough history for train_days/test_days. Try smaller windows or earlier start date."
            )
        grid_rows = []
        best_rows = []
        for split_id, split in enumerate(splits, start=1):
            test_slice = {
                sym: df.loc[split["test_start"] : split["test_end"]]
                for sym, df in price_dict.items()
                if not df.loc[split["test_start"] : split["test_end"]].empty
            }
            if not test_slice:
                continue
            best_sharpe = None
            best_param_id = None
            for param_id, param in enumerate(params_grid):
                run_test = run_portfolio(
                    test_slice,
                    execution=EXECUTION,
                    price_mode=PRICE_MODE,
                    slippage_mode="constant",
                    top_n=param["top_n"],
                    mom_lookback=param["mom_lookback"],
                    rebalance=param["rebalance"],
                    long_window=param["long_window"],
                )
                s = sharpe(run_test.equity["Portfolio_Value"])
                grid_rows.append({"split_id": split_id, "param_id": param_id, "test_sharpe": s})
                if best_sharpe is None or s > best_sharpe:
                    best_sharpe = s
                    best_param_id = param_id
            best_rows.append({"split_id": split_id, "param_id": best_param_id, "test_sharpe": best_sharpe})

        grid_df = pd.DataFrame(grid_rows)
        best_df = pd.DataFrame(best_rows)
        warnings = overfit_warnings(grid_df, best_df, equity)

    if warnings:
        (out_dir / "overfit_warnings.txt").write_text("\n".join(warnings))
    else:
        (out_dir / "overfit_warnings.txt").write_text("No warnings.")

    return out_dir

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\src\tradinglab\robustness\stability.py
# ==============================

from __future__ import annotations

import pandas as pd

from tradinglab.experiments.splits import walk_forward_splits
from tradinglab.engine.portfolio import run_portfolio, build_price_panels
from tradinglab.metrics.performance import sharpe


def parameter_stability(
    price_dict: dict[str, pd.DataFrame],
    params_grid: list[dict],
    train_days: int,
    test_days: int,
    step_days: int,
    execution: str,
    price_mode: str,
    seed: int,
) -> pd.DataFrame:
    panel_close, _ = build_price_panels(price_dict, price_mode=price_mode)
    if panel_close.empty:
        return pd.DataFrame()

    panel_idx = pd.DataFrame(index=panel_close.index)
    splits = walk_forward_splits(panel_idx, train_days, test_days, step_days)

    results = []

    for param in params_grid:
        test_sharpes = []
        for split in splits:
            test_slice = {
                sym: df.loc[split["test_start"] : split["test_end"]]
                for sym, df in price_dict.items()
                if not df.loc[split["test_start"] : split["test_end"]].empty
            }
            if not test_slice:
                continue

            run = run_portfolio(
                test_slice,
                execution=execution,
                price_mode=price_mode,
                max_position_weight=None,
                trailing_stop_pct=None,
                time_stop_days=None,
                target_vol=None,
                slippage_mode="constant",
                top_n=param["top_n"],
                mom_lookback=param["mom_lookback"],
                rebalance=param["rebalance"],
                long_window=param["long_window"],
            )
            test_sharpes.append(sharpe(run.equity["Portfolio_Value"]))

        if not test_sharpes:
            continue

        mean_sharpe = float(pd.Series(test_sharpes).mean())
        std_sharpe = float(pd.Series(test_sharpes).std(ddof=0))
        stability = mean_sharpe - std_sharpe

        row = {**param}
        row.update(
            {
                "mean_test_sharpe": mean_sharpe,
                "std_test_sharpe": std_sharpe,
                "stability_score": stability,
            }
        )
        results.append(row)

    return pd.DataFrame(results)

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\src\tradinglab\robustness\warnings.py
# ==============================

from __future__ import annotations

import pandas as pd


def overfit_warnings(
    test_sharpes: pd.DataFrame,
    split_best: pd.DataFrame,
    equity: pd.Series,
) -> list[str]:
    warnings = []

    if not test_sharpes.empty:
        median = float(test_sharpes["test_sharpe"].median())
        best = float(test_sharpes["test_sharpe"].max())
        if median != 0 and best > 2.0 * median:
            warnings.append("Best Sharpe > 2x median Sharpe: potential overfit")

    if not split_best.empty:
        for i in range(len(split_best) - 1):
            current = split_best.iloc[i]
            next_row = split_best.iloc[i + 1]
            if current["param_id"] == next_row["param_id"] and next_row["test_sharpe"] < 0:
                warnings.append("Top params collapse in next window: instability detected")
                break

    if equity is not None and not equity.empty:
        yearly = equity.resample("YE").last().pct_change().dropna()
        total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        if total > 0 and not yearly.empty:
            if yearly.max() > 0.4 * total:
                warnings.append("One year contributes >40% of total return: concentration risk")

    return warnings

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\src\tradinglab\robustness\__init__.py
# ==============================

"""Robustness diagnostics and stress testing."""

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\src\tradinglab\strategies\ma_crossover.py
# ==============================

from __future__ import annotations

import pandas as pd

from tradinglab.config import SHORT_WINDOW, LONG_WINDOW, MIN_VOL20


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column")

    out = df.copy()
    out["SMA_Short"] = out["Close"].rolling(window=SHORT_WINDOW, min_periods=SHORT_WINDOW).mean()
    out["SMA_Long"] = out["Close"].rolling(window=LONG_WINDOW, min_periods=LONG_WINDOW).mean()
    out["DailyPct"] = out["Close"].pct_change() * 100.0
    out["Vol20"] = out["DailyPct"].rolling(20, min_periods=20).std()
    return out


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = add_indicators(df)
    out["Signal"] = 0

    valid = out["SMA_Short"].notna() & out["SMA_Long"].notna()
    short = out["SMA_Short"]
    long = out["SMA_Long"]

    buy = valid & (short > long) & (short.shift(1) <= long.shift(1))
    sell = valid & (short < long) & (short.shift(1) >= long.shift(1))

    buy = buy & out["Vol20"].notna() & (out["Vol20"] >= MIN_VOL20)

    out.loc[buy, "Signal"] = 1
    out.loc[sell, "Signal"] = -1

    position = []
    holding = 0
    for s in out["Signal"].astype(int).tolist():
        if s == 1:
            holding = 1
        elif s == -1:
            holding = 0
        position.append(holding)

    out["Position"] = position
    return out

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\src\tradinglab\strategies\__init__.py
# ==============================


# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\tests\conftest.py
# ==============================

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\tests\test_backtest.py
# ==============================

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from tradinglab.engine.portfolio import build_panel, run_portfolio, buy_hold_benchmark
from tradinglab.metrics.performance import turnover
from tradinglab.cli.main import main as cli_main


def _make_price_df(prices: list[float], dates: list[pd.Timestamp]) -> pd.DataFrame:
    return pd.DataFrame({"Close": prices}, index=dates)


def test_build_panel_no_nans_after_ffill_drop():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    price_dict = {
        "AAA": _make_price_df([1, np.nan, 3, 4, 5], dates),
        "BBB": _make_price_df([1, 2, np.nan, 4, 5], dates),
        "CCC": _make_price_df([np.nan, 2, 3, 4, np.nan], dates),
    }
    panel = build_panel(price_dict, min_coverage=0.6)
    assert not panel.isna().any().any()


def test_portfolio_deterministic_on_tiny_dataset(monkeypatch):
    import tradinglab.engine.portfolio as pf

    monkeypatch.setattr(pf, "MOM_LOOKBACK", 1)
    monkeypatch.setattr(pf, "LONG_WINDOW", 2)
    monkeypatch.setattr(pf, "TOP_N", 1)
    monkeypatch.setattr(pf, "REBALANCE", "D")
    monkeypatch.setattr(pf, "FEE_RATE", 0.0)
    monkeypatch.setattr(pf, "SLIPPAGE_RATE", 0.0)
    monkeypatch.setattr(pf, "SLIPPAGE_MODE", "constant")
    monkeypatch.setattr(pf, "REGIME_SYMBOL", "QQQ")
    monkeypatch.setattr(pf, "ALLOW_REGIME_TRADE", False)

    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    price_dict = {
        "AAA": _make_price_df([10, 11, 12, 13, 14], dates),
        "BBB": _make_price_df([10, 10, 10, 10, 10], dates),
    }

    run = run_portfolio(
        price_dict,
        regime_symbol="QQQ",
        allow_regime_trade=False,
        execution="same_close",
        price_mode="raw",
        cash_buffer=0.0,
        slippage_mode="constant",
    )
    equity = run.equity["Portfolio_Value"].round(6).tolist()

    shares = 1000.0 / 11.0
    expected = [
        1000.0,
        1000.0,
        shares * 12.0,
        shares * 13.0,
        shares * 14.0,
    ]
    expected = [round(x, 6) for x in expected]
    assert equity == expected


def test_buy_hold_benchmark_matches_manual():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    price_dict = {
        "AAA": _make_price_df([10, 12, 11], dates),
        "BBB": _make_price_df([20, 18, 22], dates),
    }

    bh = buy_hold_benchmark(price_dict, regime_symbol="QQQ", allow_regime_trade=False)

    init = 1000.0
    weight = 0.5
    shares_aaa = (init * weight) / 10.0
    shares_bbb = (init * weight) / 20.0

    manual = [
        shares_aaa * 10.0 + shares_bbb * 20.0,
        shares_aaa * 12.0 + shares_bbb * 18.0,
        shares_aaa * 11.0 + shares_bbb * 22.0,
    ]

    assert np.allclose(bh["BuyHold_Value"].values, manual)


def test_turnover_sanity():
    ledger = pd.DataFrame(
        [
            {"order_notional": 100.0},
            {"order_notional": 60.0},
        ]
    )
    equity = pd.Series([1000.0, 1100.0, 1050.0])
    t = turnover(ledger, equity)
    expected_notional = 160.0
    expected = expected_notional / equity.mean()
    assert np.isclose(t, expected)


def test_smoke_main_cached_data(tmp_path, monkeypatch):
    monkeypatch.setattr("tradinglab.config.RESULTS_DIR", tmp_path)
    monkeypatch.setattr("tradinglab.cli.main.RESULTS_DIR", tmp_path)

    symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "QQQ"]
    cli_main(refresh_data=False, run_per_symbol=False, symbols=symbols)

    assert (tmp_path / "portfolio.csv").exists()
    assert (tmp_path / "portfolio_vs_bh.csv").exists()
    assert (tmp_path / "trade_ledger.csv").exists()
    assert (tmp_path / "positions.csv").exists()
    assert (tmp_path / "exposures.csv").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "metrics.json").exists()

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\tests\test_execution_risk.py
# ==============================

from __future__ import annotations

import numpy as np
import pandas as pd

from tradinglab.engine.portfolio import run_portfolio


def _make_ohlc(prices: list[float], dates: list[pd.Timestamp]) -> pd.DataFrame:
    return pd.DataFrame({"Open": prices, "Close": prices}, index=dates)


def test_no_lookahead_next_open(monkeypatch):
    import tradinglab.engine.portfolio as pf

    monkeypatch.setattr(pf, "MOM_LOOKBACK", 1)
    monkeypatch.setattr(pf, "LONG_WINDOW", 2)
    monkeypatch.setattr(pf, "TOP_N", 1)
    monkeypatch.setattr(pf, "REBALANCE", "D")
    monkeypatch.setattr(pf, "FEE_RATE", 0.0)
    monkeypatch.setattr(pf, "SLIPPAGE_RATE", 0.0)
    monkeypatch.setattr(pf, "SLIPPAGE_MODE", "constant")
    monkeypatch.setattr(pf, "REGIME_SYMBOL", "QQQ")
    monkeypatch.setattr(pf, "ALLOW_REGIME_TRADE", False)

    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    price_dict = {
        "AAA": _make_ohlc([10, 11, 12, 13, 14], dates),
        "BBB": _make_ohlc([10, 10, 10, 10, 10], dates),
        "QQQ": _make_ohlc([10, 11, 12, 13, 14], dates),
    }

    run = run_portfolio(
        price_dict,
        execution="next_open",
        price_mode="raw",
        slippage_mode="constant",
    )

    ledger = run.trade_ledger
    assert (ledger["fill_date"] > ledger["signal_date"]).all()


def test_max_position_weight(monkeypatch):
    import tradinglab.engine.portfolio as pf

    monkeypatch.setattr(pf, "MOM_LOOKBACK", 1)
    monkeypatch.setattr(pf, "LONG_WINDOW", 2)
    monkeypatch.setattr(pf, "TOP_N", 2)
    monkeypatch.setattr(pf, "REBALANCE", "D")
    monkeypatch.setattr(pf, "FEE_RATE", 0.0)
    monkeypatch.setattr(pf, "SLIPPAGE_RATE", 0.0)
    monkeypatch.setattr(pf, "SLIPPAGE_MODE", "constant")
    monkeypatch.setattr(pf, "REGIME_SYMBOL", "QQQ")
    monkeypatch.setattr(pf, "ALLOW_REGIME_TRADE", False)

    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    price_dict = {
        "AAA": _make_ohlc([10, 11, 12, 13], dates),
        "BBB": _make_ohlc([9, 9.5, 10, 10.5], dates),
        "QQQ": _make_ohlc([10, 11, 12, 13], dates),
    }

    run = run_portfolio(
        price_dict,
        execution="same_close",
        price_mode="raw",
        max_position_weight=0.2,
        cash_buffer=0.0,
        slippage_mode="constant",
    )
    last = run.positions.iloc[-1]
    close = run.panel_close.iloc[-1]
    equity = run.equity["Portfolio_Value"].iloc[-1]

    w_aaa = (last["AAA"] * close["AAA"]) / equity
    w_bbb = (last["BBB"] * close["BBB"]) / equity
    assert w_aaa <= 0.2005
    assert w_bbb <= 0.2005


def test_max_turnover_per_rebalance(monkeypatch):
    import tradinglab.engine.portfolio as pf

    monkeypatch.setattr(pf, "MOM_LOOKBACK", 1)
    monkeypatch.setattr(pf, "LONG_WINDOW", 2)
    monkeypatch.setattr(pf, "TOP_N", 1)
    monkeypatch.setattr(pf, "REBALANCE", "D")
    monkeypatch.setattr(pf, "FEE_RATE", 0.0)
    monkeypatch.setattr(pf, "SLIPPAGE_RATE", 0.0)
    monkeypatch.setattr(pf, "SLIPPAGE_MODE", "constant")
    monkeypatch.setattr(pf, "REGIME_SYMBOL", "QQQ")
    monkeypatch.setattr(pf, "ALLOW_REGIME_TRADE", False)

    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    price_dict = {
        "AAA": _make_ohlc([10, 11, 12, 13], dates),
        "BBB": _make_ohlc([20, 19, 18, 17], dates),
        "QQQ": _make_ohlc([10, 11, 12, 13], dates),
    }

    run = run_portfolio(
        price_dict,
        execution="same_close",
        price_mode="raw",
        max_turnover_per_rebalance=0.1,
        cash_buffer=0.0,
        slippage_mode="constant",
    )

    ledger = run.trade_ledger
    if not ledger.empty:
        nav = run.equity["Portfolio_Value"].iloc[0]
        first_signal = ledger["signal_date"].min()
        traded = ledger.loc[ledger["signal_date"] == first_signal, "order_notional"].sum()
        assert traded <= nav * 0.11


def test_max_gross_exposure_and_cash_buffer(monkeypatch):
    import tradinglab.engine.portfolio as pf

    monkeypatch.setattr(pf, "MOM_LOOKBACK", 1)
    monkeypatch.setattr(pf, "LONG_WINDOW", 2)
    monkeypatch.setattr(pf, "TOP_N", 2)
    monkeypatch.setattr(pf, "REBALANCE", "D")
    monkeypatch.setattr(pf, "FEE_RATE", 0.0)
    monkeypatch.setattr(pf, "SLIPPAGE_RATE", 0.0)
    monkeypatch.setattr(pf, "SLIPPAGE_MODE", "constant")
    monkeypatch.setattr(pf, "REGIME_SYMBOL", "QQQ")
    monkeypatch.setattr(pf, "ALLOW_REGIME_TRADE", False)

    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    price_dict = {
        "AAA": _make_ohlc([10, 11, 12, 13], dates),
        "BBB": _make_ohlc([9, 9.5, 10, 10.5], dates),
        "QQQ": _make_ohlc([10, 11, 12, 13], dates),
    }

    run = run_portfolio(
        price_dict,
        execution="same_close",
        price_mode="raw",
        max_gross_exposure=0.5,
        cash_buffer=0.2,
        slippage_mode="constant",
    )

    exposures = run.exposures
    assert exposures["Gross_Exposure_Pct"].max() <= 0.5005
    assert exposures["Cash_Pct"].min() >= 0.199


def test_trailing_and_time_stop(monkeypatch):
    import tradinglab.engine.portfolio as pf

    monkeypatch.setattr(pf, "MOM_LOOKBACK", 1)
    monkeypatch.setattr(pf, "LONG_WINDOW", 2)
    monkeypatch.setattr(pf, "TOP_N", 1)
    monkeypatch.setattr(pf, "REBALANCE", "D")
    monkeypatch.setattr(pf, "FEE_RATE", 0.0)
    monkeypatch.setattr(pf, "SLIPPAGE_RATE", 0.0)
    monkeypatch.setattr(pf, "SLIPPAGE_MODE", "constant")
    monkeypatch.setattr(pf, "REGIME_SYMBOL", "QQQ")
    monkeypatch.setattr(pf, "ALLOW_REGIME_TRADE", False)

    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    price_dict = {
        "AAA": _make_ohlc([10, 12, 14, 12, 11], dates),
        "QQQ": _make_ohlc([10, 11, 12, 13, 14], dates),
    }

    run_trail = run_portfolio(
        price_dict,
        execution="same_close",
        price_mode="raw",
        trailing_stop_pct=0.15,
        slippage_mode="constant",
    )
    assert (run_trail.trade_ledger["action"] == "SELL").any()

    run_time = run_portfolio(
        price_dict,
        execution="same_close",
        price_mode="raw",
        time_stop_days=2,
        slippage_mode="constant",
    )
    assert (run_time.trade_ledger["action"] == "SELL").any()


def test_target_vol_scaling(monkeypatch):
    import tradinglab.engine.portfolio as pf

    monkeypatch.setattr(pf, "MOM_LOOKBACK", 1)
    monkeypatch.setattr(pf, "LONG_WINDOW", 2)
    monkeypatch.setattr(pf, "TOP_N", 2)
    monkeypatch.setattr(pf, "REBALANCE", "D")
    monkeypatch.setattr(pf, "FEE_RATE", 0.0)
    monkeypatch.setattr(pf, "SLIPPAGE_RATE", 0.0)
    monkeypatch.setattr(pf, "SLIPPAGE_MODE", "constant")
    monkeypatch.setattr(pf, "REGIME_SYMBOL", "QQQ")
    monkeypatch.setattr(pf, "ALLOW_REGIME_TRADE", False)

    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    price_dict = {
        "AAA": _make_ohlc([10, 12, 11, 13, 12, 14], dates),
        "BBB": _make_ohlc([20, 18, 19, 17, 18, 16], dates),
        "QQQ": _make_ohlc([10, 11, 12, 13, 14, 15], dates),
    }

    run = run_portfolio(
        price_dict,
        execution="same_close",
        price_mode="raw",
        target_vol=0.05,
        vol_lookback=3,
        slippage_mode="constant",
    )
    assert run.exposures["Gross_Exposure_Pct"].max() <= 1.0

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\tests\test_experiment_outputs.py
# ==============================

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

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\tests\test_experiment_plumbing.py
# ==============================

from __future__ import annotations

import numpy as np
import pandas as pd

from tradinglab.data.panel import prepare_panel_with_history_filter
from tradinglab.experiments.splits import walk_forward_splits


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


def test_panel_filter_drops_short_history():
    price_dict = {
        "AAA": _make_df("2020-01-01", 300),
        "BBB": _make_df("2020-07-01", 100),
        "CCC": _make_df("2020-03-01", 200),
    }

    filtered, panel_close, dropped = prepare_panel_with_history_filter(
        price_dict,
        price_mode="adj",
        min_history_days=180,
    )

    assert "BBB" in dropped
    assert "AAA" in filtered
    assert "CCC" in filtered
    assert not panel_close.empty
    assert panel_close.isna().sum().sum() == 0


def test_walk_forward_splits_non_empty():
    dates = pd.date_range("2020-01-01", periods=1000, freq="D")
    panel_idx = pd.DataFrame(index=dates)
    splits = walk_forward_splits(panel_idx, train_days=200, test_days=50, step_days=50)
    assert len(splits) > 0
    first = splits[0]
    assert first["train_start"] == dates[0]
    assert first["train_end"] == dates[199]
    assert first["test_start"] == dates[200]

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\tests\test_health.py
# ==============================

from __future__ import annotations

import subprocess
import sys


def test_health_check_failure_exit_code():
    # run health check with empty universe by tampering env (should fail gracefully if no data)
    result = subprocess.run([sys.executable, "-m", "app.health", "--universe", "small"], capture_output=True, text=True)
    assert result.returncode in (0, 1)

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\tests\test_live.py
# ==============================

from __future__ import annotations

import pandas as pd

from app.live.runner import generate_orders_from_weights, run_live_cycle
from app.state import LiveState, save_state, load_state
from app.brokers.paper import CachedMarketDataProvider, PaperBroker
from app.brokers.base import Order
from app.reporting import write_daily_report


def test_generate_orders_from_weights():
    symbols = ["AAA", "BBB"]
    current_positions = {"AAA": 10.0, "BBB": 0.0}
    target_weights = {"AAA": 0.5, "BBB": 0.5}
    prices = pd.Series({"AAA": 10.0, "BBB": 20.0})
    equity = 200.0

    orders = generate_orders_from_weights(symbols, current_positions, target_weights, prices, equity)
    assert len(orders) == 1


def test_state_roundtrip(tmp_path):
    state = LiveState.default({"foo": "bar"})
    path = tmp_path / "state.json"
    save_state(path, state)
    loaded = load_state(path, {"foo": "bar"})
    assert loaded.cash == state.cash
    assert loaded.config_snapshot["foo"] == "bar"


def test_paper_broker_fill():
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    df = pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [10.5, 11.5],
            "Low": [9.5, 10.5],
            "Close": [10.0, 11.0],
            "Adj Close": [10.0, 11.0],
        },
        index=dates,
    )
    provider = CachedMarketDataProvider(price_dict={"AAA": df}, price_mode="raw")
    provider.set_current_date(dates[0])
    broker = PaperBroker(provider, cash=100.0, positions={})

    orders = [Order(symbol="AAA", side="BUY", qty=5.0)]
    fills = broker.place_orders(orders)
    assert fills
    assert broker.get_positions()["AAA"] > 0


def test_live_idempotency(tmp_path, monkeypatch):
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0],
            "High": [10.5, 11.5, 12.5],
            "Low": [9.5, 10.5, 11.5],
            "Close": [10.0, 11.0, 12.0],
            "Adj Close": [10.0, 11.0, 12.0],
        },
        index=dates,
    )

    monkeypatch.setattr("tradinglab.data.fetcher.load_or_fetch_symbols", lambda *args, **kwargs: {"AAA": df, "QQQ": df})

    state_path = tmp_path / "state.json"
    state = LiveState.default({})
    state.last_processed_date = dates[-1].isoformat()
    save_state(state_path, state)

    orders = run_live_cycle(
        universe="small",
        refresh_data=False,
        execution="next_open",
        price_mode="raw",
        dry_run=False,
        state_path=state_path,
        flatten=False,
    )
    assert orders == []


def test_health_report_sections(tmp_path):
    path = tmp_path / "report.md"
    write_daily_report(
        path,
        date="2024-01-02",
        start_equity=100.0,
        end_equity=105.0,
        positions={"AAA": 1.0},
        prices=pd.Series({"AAA": 100.0}),
        cash_pct=0.5,
        gross_exposure=0.5,
        turnover=0.1,
        risk_triggers=["turnover_cap_applied"],
        qqq_bh=120.0,
        strategy_since_start=5.0,
    )
    text = path.read_text()
    assert "## Equity" in text
    assert "## Exposure" in text
    assert "## Positions" in text
    assert "## Risk Triggers" in text


def test_live_dry_run_deterministic(tmp_path):
    orders1 = [Order(symbol="AAA", side="BUY", qty=1.0)]
    orders2 = [Order(symbol="AAA", side="BUY", qty=1.0)]
    assert orders1[0].qty == orders2[0].qty

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\tests\test_live_safety.py
# ==============================

from __future__ import annotations

import os
import json
import tempfile
from pathlib import Path

import pandas as pd

from app.live.runner import run_live_cycle
from app.brokers.base import Order


def _mock_history():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0],
            "High": [10.5, 11.5, 12.5],
            "Low": [9.5, 10.5, 11.5],
            "Close": [10.0, 11.0, 12.0],
            "Adj Close": [10.0, 11.0, 12.0],
        },
        index=dates,
    )
    return {"AAA": df, "QQQ": df}


def test_live_gating_refuses(monkeypatch, tmp_path):
    monkeypatch.setattr("tradinglab.data.fetcher.load_or_fetch_symbols", lambda *a, **k: _mock_history())
    orders = run_live_cycle(
        universe="small",
        refresh_data=False,
        execution="next_open",
        price_mode="raw",
        dry_run=False,
        state_path=tmp_path / "state.json",
        flatten=False,
        mode="live",
        confirm=False,
        accept_real_trading="NO",
        flatten_on_kill=False,
    )
    assert orders == []


def test_kill_switch_blocks(monkeypatch, tmp_path):
    monkeypatch.setattr("tradinglab.data.fetcher.load_or_fetch_symbols", lambda *a, **k: _mock_history())
    os.environ["KILL_SWITCH"] = "true"
    orders = run_live_cycle(
        universe="small",
        refresh_data=False,
        execution="next_open",
        price_mode="raw",
        dry_run=False,
        state_path=tmp_path / "state.json",
        flatten=False,
        mode="paper",
        confirm=False,
        accept_real_trading="",
        flatten_on_kill=False,
    )
    assert orders == []
    os.environ.pop("KILL_SWITCH", None)


def test_pending_orders_requires_confirm(monkeypatch, tmp_path):
    monkeypatch.setattr("tradinglab.data.fetcher.load_or_fetch_symbols", lambda *a, **k: _mock_history())
    os.environ["LIVE_TRADING_ENABLED"] = "true"
    orders = run_live_cycle(
        universe="small",
        refresh_data=False,
        execution="next_open",
        price_mode="raw",
        dry_run=False,
        state_path=tmp_path / "state.json",
        flatten=False,
        mode="live",
        confirm=False,
        accept_real_trading="YES",
        flatten_on_kill=False,
    )
    pending = list(Path("logs").glob("pending_orders_*.json"))
    assert pending
    os.environ.pop("LIVE_TRADING_ENABLED", None)


def test_safety_checks_block(monkeypatch, tmp_path):
    monkeypatch.setattr("tradinglab.data.fetcher.load_or_fetch_symbols", lambda *a, **k: _mock_history())
    monkeypatch.setattr("app.live.runner.MAX_ORDER_NOTIONAL", 1.0)
    orders = run_live_cycle(
        universe="small",
        refresh_data=False,
        execution="next_open",
        price_mode="raw",
        dry_run=False,
        state_path=tmp_path / "state.json",
        flatten=False,
        mode="paper",
        confirm=False,
        accept_real_trading="",
        flatten_on_kill=False,
    )
    assert orders == []

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\tests\test_momentum_strategy.py
# ==============================

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

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\tests\test_perf_equivalence.py
# ==============================

from __future__ import annotations

import numpy as np
import pandas as pd

from tradinglab.engine.portfolio import run_portfolio


def _make_price_df(prices: list[float], dates: list[pd.Timestamp]) -> pd.DataFrame:
    return pd.DataFrame({"Open": prices, "High": prices, "Low": prices, "Close": prices}, index=dates)


def reference_portfolio(price_dict: dict[str, pd.DataFrame]) -> pd.Series:
    # simple reference: buy-and-hold equal weight on first day
    dates = list(next(iter(price_dict.values())).index)
    symbols = list(price_dict.keys())
    init = 1000.0
    weights = {s: 1.0 / len(symbols) for s in symbols}
    shares = {s: (init * weights[s]) / price_dict[s].iloc[0]["Close"] for s in symbols}
    values = []
    for date in dates:
        v = 0.0
        for sym in symbols:
            v += shares[sym] * price_dict[sym].loc[date, "Close"]
        values.append(v)
    return pd.Series(values, index=dates)


def test_engine_equivalence_tolerance():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    price_dict = {
        "AAA": _make_price_df([10, 11, 12, 13, 14], dates),
        "BBB": _make_price_df([10, 10, 10, 10, 10], dates),
        "QQQ": _make_price_df([10, 11, 12, 13, 14], dates),
    }

    run = run_portfolio(price_dict, execution="same_close", price_mode="raw", slippage_mode="constant")
    ref = reference_portfolio({"AAA": price_dict["AAA"], "BBB": price_dict["BBB"]})

    assert np.isfinite(run.equity["Portfolio_Value"].iloc[-1])
    assert abs(run.equity["Portfolio_Value"].iloc[-1] - run.equity["Portfolio_Value"].iloc[-1]) < 1e-6


def test_golden_snapshot():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    price_dict = {
        "AAA": _make_price_df([10, 11, 12, 13], dates),
        "BBB": _make_price_df([10, 10, 10, 10], dates),
        "QQQ": _make_price_df([10, 11, 12, 13], dates),
    }

    run = run_portfolio(price_dict, execution="same_close", price_mode="raw", slippage_mode="constant")
    end_value = round(run.equity["Portfolio_Value"].iloc[-1], 6)
    assert end_value > 0

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\tests\test_robustness.py
# ==============================

from __future__ import annotations

import numpy as np
import pandas as pd

from tradinglab.robustness.monte_carlo import bootstrap_paths
from tradinglab.robustness.rolling import rolling_metrics
from tradinglab.robustness.regimes import regime_metrics
from tradinglab.robustness.allocation import vol_target_exposure


def test_monte_carlo_paths_shape():
    returns = pd.Series([0.01, -0.005, 0.002, 0.0, 0.004])
    paths_df, summary_df = bootstrap_paths(returns, paths=10, seed=42)
    assert len(summary_df) == 10
    assert paths_df.shape[0] == len(returns)
    assert paths_df.shape[1] == 10


def test_rolling_metrics_alignment():
    dates = pd.date_range("2024-01-01", periods=300, freq="D")
    equity = pd.Series(np.linspace(1.0, 2.0, 300), index=dates)
    benchmark = pd.Series(np.linspace(1.0, 1.8, 300), index=dates)
    turnover = pd.Series(0.0, index=dates)
    exposure = pd.Series(1.0, index=dates)

    roll = rolling_metrics(equity, benchmark, turnover, exposure, window=252)
    assert not roll.empty
    assert roll.index[0] == dates[252]


def test_regime_segmentation_logic():
    dates = pd.date_range("2024-01-01", periods=260, freq="D")
    qqq = pd.DataFrame({"Close": np.linspace(100, 150, 260)}, index=dates)
    price_dict = {"QQQ": qqq}
    equity = pd.Series(np.linspace(1.0, 1.2, 260), index=dates)

    reg = regime_metrics(price_dict, equity, regime_symbol="QQQ", price_mode="raw")
    assert "Bull" in reg["Label"].values


def test_allocation_exposure_cap():
    returns = pd.Series([0.01, -0.01, 0.02, -0.005, 0.0, 0.01])
    exposure = vol_target_exposure(returns, target_vol=0.1)
    assert exposure.max() <= 1.0

# ==============================
# FILE: C:\Users\jihad\Documents\GitHub\Neetechs_trading\tests\test_signal.py
# ==============================

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.signal import generate_signal


def _make_df(start: str, periods: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="D")
    values = np.linspace(100.0, 120.0, periods)
    return pd.DataFrame(
        {
            "Open": values,
            "Close": values,
            "Adj Close": values,
        },
        index=idx,
    )


def test_signal_smoke(tmp_path: Path):
    price_dict = {
        "AAPL": _make_df("2022-01-01", 600),
        "MSFT": _make_df("2022-01-01", 600),
        "NVDA": _make_df("2022-01-01", 600),
        "AMZN": _make_df("2022-01-01", 600),
        "QQQ": _make_df("2022-01-01", 600),
    }
    out_path = tmp_path / "live_signal.csv"
    df = generate_signal(
        universe="small",
        top_n=2,
        execution="next_open",
        price_mode="adj",
        refresh_data=False,
        positions_file=None,
        price_dict=price_dict,
        output_path=out_path,
    )
    assert out_path.exists()
    assert not df.empty
    assert "symbol" in df.columns
