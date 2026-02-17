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
