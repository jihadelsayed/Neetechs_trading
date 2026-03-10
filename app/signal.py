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
    parser.add_argument("--momentum-months", type=int, default=6)
    parser.add_argument("--max-pe", type=float, default=None, help="Exclude stocks with P/E greater than this value")
    parser.add_argument("--stop-loss-pct", type=float, default=None, help="Stop loss percent (e.g. 0.1 for 10%%)")
    parser.add_argument(
        "--trailing-stop-pct",
        type=float,
        default=None,
        help="Trailing stop percent (e.g. 0.08 for 8%%)",
    )
    parser.add_argument(
        "--dispersion-threshold-multiplier",
        type=float,
        default=1.0,
        help="Require dispersion >= median_dispersion * multiplier",
    )
    parser.add_argument("--execution", choices=["next_open", "same_close"], default=None)
    parser.add_argument("--price-mode", choices=["adj", "raw"], default=None)
    parser.add_argument(
        "--signal-output-mode",
        choices=["portfolio", "standalone"],
        default="portfolio",
        help="portfolio=weight-delta actions, standalone=price-based BUY/SELL/HOLD signals",
    )
    parser.add_argument(
        "--price-signals-output",
        type=str,
        default="results/price_signals.csv",
        help="Output CSV path for standalone price-based signals",
    )
    parser.add_argument(
        "--position-state-path",
        type=str,
        default="logs/position_state.json",
        help="Persistent position state file path",
    )
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


def _load_positions_and_weights(
    path: str | None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    if not path:
        return {}, {}, {}, {}

    p = Path(path)
    if p.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(p)
        except Exception:
            return {}, {}, {}, {}
        required_cols = {"symbol", "current_weight"}
        if not required_cols.issubset({str(c).strip().lower() for c in df.columns}):
            return {}, {}, {}, {}

        col_map = {str(c).strip().lower(): c for c in df.columns}
        sym_col = col_map["symbol"]
        w_col = col_map["current_weight"]
        entry_col = col_map.get("entry_price")
        high_col = col_map.get("highest_price_since_entry")

        current_weights: dict[str, float] = {}
        entry_prices: dict[str, float] = {}
        highest_prices: dict[str, float] = {}
        for _, row in df.iterrows():
            try:
                sym = str(row[sym_col]).strip().upper()
                w = float(row[w_col])
            except Exception:
                continue
            if not sym:
                continue
            current_weights[sym] = w
            if entry_col is not None:
                try:
                    entry_prices[sym] = float(row[entry_col])
                except Exception:
                    pass
            if high_col is not None:
                try:
                    highest_prices[sym] = float(row[high_col])
                except Exception:
                    pass
        return {}, current_weights, entry_prices, highest_prices

    data = json.loads(p.read_text())
    if not isinstance(data, dict):
        return {}, {}, {}, {}
    positions_data = data["positions"] if isinstance(data.get("positions"), dict) else data
    entry_data = data.get("entry_prices", {})
    high_data = data.get("highest_prices_since_entry", {})
    if not isinstance(positions_data, dict):
        positions_data = {}
    if not isinstance(entry_data, dict):
        entry_data = {}
    if not isinstance(high_data, dict):
        high_data = {}

    positions: dict[str, float] = {}
    for k, v in positions_data.items():
        try:
            positions[str(k).upper()] = float(v)
        except Exception:
            continue
    entry_prices: dict[str, float] = {}
    highest_prices: dict[str, float] = {}
    for k, v in entry_data.items():
        try:
            entry_prices[str(k).upper()] = float(v)
        except Exception:
            continue
    for k, v in high_data.items():
        try:
            highest_prices[str(k).upper()] = float(v)
        except Exception:
            continue
    return positions, {}, entry_prices, highest_prices


def _load_position_state(path: str | None) -> dict[str, dict]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}

    out: dict[str, dict] = {}
    for sym, rec in data.items():
        if not isinstance(rec, dict):
            continue
        key = str(sym).strip().upper()
        if not key:
            continue
        try:
            entry_price = float(rec.get("entry_price")) if rec.get("entry_price") is not None else None
        except Exception:
            entry_price = None
        try:
            highest_price = (
                float(rec.get("highest_price_since_entry"))
                if rec.get("highest_price_since_entry") is not None
                else None
            )
        except Exception:
            highest_price = None
        out[key] = {
            "entry_date": rec.get("entry_date"),
            "entry_price": entry_price,
            "highest_price_since_entry": highest_price,
            "is_open": int(rec.get("is_open", 0)),
        }
    return out


def _save_position_state(path: str | None, state: dict[str, dict]) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    serializable: dict[str, dict] = {}
    for sym, rec in state.items():
        if not isinstance(rec, dict):
            continue
        serializable[str(sym).upper()] = {
            "entry_date": rec.get("entry_date"),
            "entry_price": rec.get("entry_price"),
            "highest_price_since_entry": rec.get("highest_price_since_entry"),
            "is_open": int(rec.get("is_open", 0)),
        }
    p.write_text(json.dumps(serializable, indent=2))


def generate_signal(
    universe: str,
    top_n: int | None,
    execution: str | None,
    price_mode: str | None,
    refresh_data: bool,
    positions_file: str | None,
    end_date: str | None,
    momentum_months: int | None = None,
    max_pe: float | None = None,
    stop_loss_pct: float | None = None,
    trailing_stop_pct: float | None = None,
    dispersion_threshold_multiplier: float = 1.0,
    signal_output_mode: str = "portfolio",
    price_signals_output: str | None = None,
    position_state_path: str | None = None,
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
        dedupe_share_classes,
        dispersion_series,
        filter_scores_by_pe,
        get_pe_map,
        inverse_vol_weights,
    )
    from app.brokers.paper import CachedMarketDataProvider

    effective_top_n = top_n if top_n is not None else TOP_N
    effective_execution = execution or EXECUTION
    effective_price_mode = price_mode or PRICE_MODE
    effective_end = end_date or pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    momentum_lookback_days = None
    if momentum_months is not None:
        momentum_lookback_days = max(int(momentum_months), 1) * 21

    effective_start = START_DATE
    required_history_days = None
    # For larger momentum windows, backfill extra history so momentum, vol, dispersion,
    # and regime filters are all fully populated at the signal date.
    if momentum_months is not None and momentum_months > 6 and momentum_lookback_days is not None:
        skip_days = 21
        vol_lookback_days = 126
        dispersion_window_days = 252
        regime_window_days = LONG_WINDOW
        safety_buffer_days = 252
        required_history_days = (
            max(
                momentum_lookback_days + skip_days,
                vol_lookback_days,
                dispersion_window_days,
                regime_window_days,
            )
            + safety_buffer_days
        )
        end_ts = pd.to_datetime(effective_end).normalize()
        regime_months = max(1, int((LONG_WINDOW + 20) // 21))
        # Use month-based expansion (not business-day subtraction) so fetched history
        # maps to real trading sessions and doesn't under-fetch by holiday gaps.
        required_months = max(int(momentum_months) + 1, 6, 12, regime_months) + 23
        required_start = (end_ts - pd.DateOffset(months=required_months)).normalize()
        configured_start = pd.to_datetime(START_DATE).normalize()
        effective_start = min(configured_start, required_start).strftime("%Y-%m-%d")
        if effective_start != START_DATE:
            print(
                f"momentum_months={momentum_months}: extending history start to {effective_start} "
                f"(~{required_months} months warmup)"
            )

    if universe == "small":
        symbols = ["AAPL", "MSFT", "NVDA", "AMZN", REGIME_SYMBOL]
    else:
        symbols = nasdaq100_tickers()
        if REGIME_SYMBOL not in symbols:
            symbols.append(REGIME_SYMBOL)

    if price_dict is None and refresh_data:
        price_dict = load_or_fetch_symbols(symbols, refresh=True, start_date=effective_start, end_date=effective_end)
    provider = CachedMarketDataProvider(price_dict=price_dict, price_mode=effective_price_mode)
    history = provider.get_history(symbols, start=effective_start, end=effective_end)
    panel_close, panel_open = build_price_panels(
        history,
        price_mode=effective_price_mode,
        required_history_days=required_history_days,
        keep_symbols={REGIME_SYMBOL},
    )

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

    positions, current_weight_overrides, entry_price_overrides, highest_price_overrides = _load_positions_and_weights(
        positions_file
    )
    position_state = _load_position_state(position_state_path)
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
    score_df, sigma_df = blended_momentum_score(panel_close, momentum_lookback_days=momentum_lookback_days)
    dispersion, dispersion_med = dispersion_series(score_df)
    disp_val = dispersion.loc[signal_date] if signal_date in dispersion.index else float("nan")
    disp_med = dispersion_med.loc[signal_date] if signal_date in dispersion_med.index else float("nan")

    score_row = score_df.loc[signal_date] if signal_date in score_df.index else pd.Series(dtype=float)
    score_row = score_row.replace([float("inf"), float("-inf")], pd.NA)
    nan_scores = int(score_row.isna().sum()) if not score_row.empty else 0

    reason = None
    if not regime_ok:
        reason = "REGIME_OFF_QQQ_BELOW_SMA200"
    elif pd.notna(disp_val) and pd.notna(disp_med):
        effective_dispersion_threshold = disp_med * float(dispersion_threshold_multiplier)
        if disp_val < effective_dispersion_threshold:
            reason = "DISPERSION_LOW"

    # Explicitly exclude QQQ from ranking unless regime trading is enabled.
    exclude = {REGIME_SYMBOL}
    eligible = score_row.dropna()
    eligible = eligible[~eligible.index.isin(exclude)]

    if reason in {"REGIME_OFF_QQQ_BELOW_SMA200", "DISPERSION_LOW"}:
        eligible = eligible.iloc[0:0]
    if max_pe is not None:
        pe_map = get_pe_map(eligible.index.tolist())
        eligible, pe_excluded = filter_scores_by_pe(eligible, pe_map, max_pe=max_pe)
        if pe_excluded:
            print(f"P/E filter removed {len(pe_excluded)} symbol(s) with P/E > {max_pe}.")

    top = eligible.sort_values(ascending=False).head(effective_top_n).index.tolist() if effective_top_n != 0 else []
    top = dedupe_share_classes(top, price_dict=history, asof_date=signal_date)
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
    print(
        f"dispersion: {disp_val:.6f} "
        f"(median: {disp_med:.6f}, multiplier: {dispersion_threshold_multiplier:.3f})"
    )
    print(f"eligible symbols: {eligible_count}")
    print(f"nan scores: {nan_scores}")
    print(f"trend filter fails: {trend_fail_count}")
    print(f"eligible score min/max: {min_score:.6f}/{max_score:.6f}")

    current_weights: dict[str, float] = dict(current_weight_overrides)
    if equity > 0:
        for sym, qty in positions.items():
            if qty <= 0:
                continue
            if sym in current_weights:
                continue
            px = float(prices_close.get(sym, float("nan")))
            if pd.notna(px) and px > 0:
                current_weights[sym] = float(qty * px / equity)

    close_row = panel_close.loc[signal_date] if signal_date in panel_close.index else pd.Series(dtype=float)
    trend_row = (
        panel_close.rolling(LONG_WINDOW).mean().loc[signal_date]
        if signal_date in panel_close.index
        else pd.Series(dtype=float)
    )
    ranked_scores = eligible.sort_values(ascending=False)
    rank_map = {sym: int(i + 1) for i, sym in enumerate(ranked_scores.index.tolist())}
    top_set = set(top)

    standalone_rows: list[dict] = []
    signal_symbols = sorted(panel_close.columns.tolist())
    position_threshold = 1e-10
    for sym in signal_symbols:
        close_val = float(close_row.get(sym, float("nan")))
        score_val = float(score_row.get(sym, float("nan"))) if sym in score_row.index else float("nan")
        rank_val = rank_map.get(sym)
        sma200_val = float(trend_row.get(sym, float("nan")))
        above_trend = pd.notna(close_val) and pd.notna(sma200_val) and close_val > sma200_val
        in_top = sym in top_set
        current_weight = float(current_weights.get(sym, 0.0))
        has_position = current_weight > position_threshold
        state_rec = position_state.get(sym, {})
        is_open_state = bool(int(state_rec.get("is_open", 0))) if isinstance(state_rec, dict) else False
        is_open = is_open_state or has_position

        entry_price = float("nan")
        highest_price_since_entry = float("nan")
        stop_loss_price = float("nan")
        trailing_stop_price = float("nan")
        active_sell_price = float("nan")
        stop_triggered = False
        if is_open and pd.notna(close_val):
            entry_from_state = state_rec.get("entry_price") if isinstance(state_rec, dict) else None
            high_from_state = state_rec.get("highest_price_since_entry") if isinstance(state_rec, dict) else None
            entry_price = float(entry_from_state) if entry_from_state is not None else float(entry_price_overrides.get(sym, close_val))
            if not pd.notna(entry_price) or entry_price <= 0:
                entry_price = close_val
            highest_price_since_entry = (
                float(high_from_state) if high_from_state is not None else float(highest_price_overrides.get(sym, close_val))
            )
            if not pd.notna(highest_price_since_entry) or highest_price_since_entry <= 0:
                highest_price_since_entry = close_val

            latest_high = close_val
            sym_df = history.get(sym)
            if sym_df is not None and not sym_df.empty and signal_date in sym_df.index:
                if "High" in sym_df.columns:
                    try:
                        latest_high = float(sym_df.loc[signal_date, "High"])
                        if effective_price_mode == "adj" and "Adj Close" in sym_df.columns and "Close" in sym_df.columns:
                            close_raw = float(sym_df.loc[signal_date, "Close"])
                            adj_close = float(sym_df.loc[signal_date, "Adj Close"])
                            if close_raw != 0:
                                latest_high = latest_high * (adj_close / close_raw)
                    except Exception:
                        latest_high = close_val
            highest_price_since_entry = max(highest_price_since_entry, close_val, latest_high, entry_price)

            if stop_loss_pct is not None and stop_loss_pct > 0:
                stop_loss_price = entry_price * (1.0 - float(stop_loss_pct))
            if trailing_stop_pct is not None and trailing_stop_pct > 0:
                trailing_stop_price = highest_price_since_entry * (1.0 - float(trailing_stop_pct))

            stop_candidates = [x for x in [stop_loss_price, trailing_stop_price] if pd.notna(x)]
            if stop_candidates:
                active_sell_price = max(stop_candidates)
                stop_triggered = close_val <= active_sell_price

        if pd.isna(close_val) or pd.isna(score_val) or pd.isna(sma200_val):
            signal = "HOLD"
            signal_reason = "INSUFFICIENT_DATA"
            signal_rule = "INSUFFICIENT_DATA"
        elif in_top and above_trend:
            signal = "BUY"
            signal_reason = "TOP_N_AND_ABOVE_TREND"
            signal_rule = "TOP_N_AND_ABOVE_TREND"
        elif (not in_top) or (not above_trend):
            signal = "SELL"
            if (not in_top) and (not above_trend):
                signal_reason = "OUT_OF_TOP_N_AND_BELOW_TREND"
            elif not in_top:
                signal_reason = "OUT_OF_TOP_N"
            else:
                signal_reason = "BELOW_TREND"
            signal_rule = signal_reason
        else:
            signal = "HOLD"
            signal_reason = "NO_CHANGE"
            signal_rule = "NO_CHANGE"

        if stop_triggered:
            signal = "SELL"
            signal_reason = "ACTIVE_STOP_PRICE_TRIGGERED"
            signal_rule = "ACTIVE_STOP_PRICE_TRIGGERED"

        exec_price = float(prices_exec.get(sym, close_val if pd.notna(close_val) else 0.0))
        if signal == "BUY":
            if not is_open and pd.notna(exec_price) and exec_price > 0:
                entry_price = exec_price
                if not pd.notna(highest_price_since_entry) or highest_price_since_entry <= 0:
                    highest_price_since_entry = entry_price
                highest_price_since_entry = max(highest_price_since_entry, close_val if pd.notna(close_val) else entry_price)
                if stop_loss_pct is not None and stop_loss_pct > 0:
                    stop_loss_price = entry_price * (1.0 - float(stop_loss_pct))
                if trailing_stop_pct is not None and trailing_stop_pct > 0:
                    trailing_stop_price = highest_price_since_entry * (1.0 - float(trailing_stop_pct))
                stop_candidates = [x for x in [stop_loss_price, trailing_stop_price] if pd.notna(x)]
                active_sell_price = max(stop_candidates) if stop_candidates else float("nan")
            position_state[sym] = {
                "entry_date": fill_date.date().isoformat(),
                "entry_price": float(entry_price) if pd.notna(entry_price) else float(exec_price),
                "highest_price_since_entry": float(highest_price_since_entry)
                if pd.notna(highest_price_since_entry)
                else float(exec_price),
                "is_open": 1,
            }
        elif signal == "SELL":
            if sym in position_state:
                rec = position_state[sym]
                if not isinstance(rec, dict):
                    rec = {}
                rec["is_open"] = 0
                if pd.notna(entry_price):
                    rec["entry_price"] = float(entry_price)
                if pd.notna(highest_price_since_entry):
                    rec["highest_price_since_entry"] = float(highest_price_since_entry)
                position_state[sym] = rec
        else:
            if is_open:
                position_state[sym] = {
                    "entry_date": state_rec.get("entry_date") if isinstance(state_rec, dict) else signal_date.date().isoformat(),
                    "entry_price": float(entry_price) if pd.notna(entry_price) else float(exec_price),
                    "highest_price_since_entry": float(highest_price_since_entry)
                    if pd.notna(highest_price_since_entry)
                    else float(exec_price),
                    "is_open": 1,
                }

        standalone_rows.append(
            {
                "asof_date": signal_date.date(),
                "symbol": sym,
                "close": close_val,
                "score": score_val,
                "rank": rank_val,
                "sma200": sma200_val,
                "signal": signal,
                "reason": signal_reason,
                "entry_price": entry_price,
                "highest_price_since_entry": highest_price_since_entry,
                "stop_loss_price": stop_loss_price,
                "trailing_stop_price": trailing_stop_price,
                "active_sell_price": active_sell_price,
                "signal_rule": signal_rule,
            }
        )

    standalone_df = pd.DataFrame(
        standalone_rows,
        columns=[
            "asof_date",
            "symbol",
            "close",
            "score",
            "rank",
            "sma200",
            "signal",
            "reason",
            "entry_price",
            "highest_price_since_entry",
            "stop_loss_price",
            "trailing_stop_price",
            "active_sell_price",
            "signal_rule",
        ],
    )
    signal_map = dict(zip(standalone_df["symbol"], standalone_df["signal"]))
    buy_symbols = [sym for sym in top if signal_map.get(sym) == "BUY"]
    standalone_path = Path(price_signals_output) if price_signals_output else (Path("results") / "price_signals.csv")
    standalone_path.parent.mkdir(parents=True, exist_ok=True)
    standalone_df.to_csv(standalone_path, index=False)
    _save_position_state(position_state_path, position_state)
    print(f"Standalone signal output: {standalone_path}")

    if signal_output_mode == "standalone":
        for _, row in standalone_df.iterrows():
            print(
                f"{row['signal']:>4} {row['symbol']:<6} "
                f"close={row['close']:.2f} "
                f"score={row['score']:.4f} "
                f"rank={row['rank'] if pd.notna(row['rank']) else 'NA'} "
                f"sma200={row['sma200']:.2f}"
            )
        return standalone_df

    target_weights: dict[str, float] = {}
    if len(buy_symbols) == 0:
        reason = reason or "NO_BUY_SIGNALS"
        if REGIME_SYMBOL in panel_close.columns:
            print(f"No buys, holding {REGIME_SYMBOL} (reason: {reason})")
            target_weights[REGIME_SYMBOL] = 1.0
        else:
            print(f"No buys, stay in cash/QQQ (reason: {reason})")
    else:
        reason = "OK"
        sigma_row = sigma_df.loc[signal_date] if signal_date in sigma_df.index else pd.Series(dtype=float)
        weights = inverse_vol_weights(sigma_row.reindex(buy_symbols))
        for sym in buy_symbols:
            w = float(weights.get(sym, 0.0))
            if w > 0:
                target_weights[sym] = w

    action_threshold = 1e-4
    symbols = sorted(set(current_weights.keys()) | set(target_weights.keys()))
    for sym in symbols:
        current_weight = float(current_weights.get(sym, 0.0))
        target_weight = float(target_weights.get(sym, 0.0))
        delta_weight = float(target_weight - current_weight)

        if current_weight <= action_threshold and target_weight > action_threshold:
            action = "BUY"
        elif current_weight > action_threshold and target_weight <= action_threshold:
            action = "SELL"
        elif (
            current_weight > action_threshold
            and target_weight > action_threshold
            and abs(delta_weight) < action_threshold
        ):
            action = "HOLD"
        elif delta_weight >= action_threshold:
            action = "ADD"
        elif delta_weight <= -action_threshold:
            action = "REDUCE"
        else:
            action = "HOLD"

        est_price = float(prices_exec.get(sym, 0.0))
        rows.append(
            {
                "asof_date": signal_date.date(),
                "symbol": sym,
                "action": action,
                "current_weight": current_weight,
                "target_weight": target_weight,
                "delta_weight": delta_weight,
                "est_price": est_price,
                "est_current_notional": float(equity * current_weight),
                "est_target_notional": float(equity * target_weight),
                "est_delta_notional": float(equity * delta_weight),
                "reason": reason,
            }
        )

    out_df = pd.DataFrame(
        rows,
        columns=[
            "asof_date",
            "symbol",
            "action",
            "current_weight",
            "target_weight",
            "delta_weight",
            "est_price",
            "est_current_notional",
            "est_target_notional",
            "est_delta_notional",
            "reason",
        ],
    )
    output_path = output_path or Path("results") / "live_signal.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print("Signal output:")
    for _, row in out_df.iterrows():
        print(
            f"{row['action']:>4} {row['symbol']:<6} "
            f"cw={row['current_weight']:.4f} "
            f"tw={row['target_weight']:.4f} "
            f"dw={row['delta_weight']:.4f} "
            f"px={row['est_price']:.2f} "
            f"d_notional={row['est_delta_notional']:.2f}"
        )

    return out_df


def run(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cli_args = sys.argv[1:] if argv is None else argv
    momentum_months = args.momentum_months if "--momentum-months" in cli_args else None
    generate_signal(
        universe=args.universe,
        top_n=args.top_n,
        momentum_months=momentum_months,
        max_pe=args.max_pe,
        stop_loss_pct=args.stop_loss_pct,
        trailing_stop_pct=args.trailing_stop_pct,
        dispersion_threshold_multiplier=args.dispersion_threshold_multiplier,
        signal_output_mode=args.signal_output_mode,
        price_signals_output=args.price_signals_output,
        position_state_path=args.position_state_path,
        execution=args.execution,
        price_mode=args.price_mode,
        refresh_data=bool(args.refresh_data),
        positions_file=args.positions_file,
        end_date=args.end_date,
    )


if __name__ == "__main__":
    run()
