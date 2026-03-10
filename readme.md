# Neetechs Trading Lab

Neetechs Trading Lab is a Python trading research project for momentum-based portfolio backtesting, experiments, signal generation, and paper/live execution workflows.

## What It Includes

- Portfolio backtests with configurable execution, pricing, and risk controls
- Experiment runner with walk-forward and optional grid search
- Robustness diagnostics and performance reporting
- Signal generation output (`results/live_signal.csv`) for execution pipelines
- Paper trading loop and optional Alpaca live broker integration
- Streamlit dashboard (optional dependency)

## Requirements

- Python 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```

Install package + CLI scripts:

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e .[dev]
pip install -e .[dashboard]
```

## Quickstart

```bash
neetechs-health --universe small
neetechs-backtest --universe small --start 2024-01-01 --end 2025-01-01
```

Equivalent module commands:

```bash
python -m app.health --universe small
python -m app.backtest --universe small --start 2024-01-01 --end 2025-01-01
```

## Main Commands

- `neetechs-backtest` / `python -m app.backtest`
- `neetechs-experiment` / `python -m app.experiment`
- `neetechs-robustness` / `python -m app.robustness`
- `neetechs-live` / `python -m app.live`
- `neetechs-health` / `python -m app.health`
- `python -m app.signal`

Example runs:

```bash
python -m app.experiment --universe nasdaq100 --walk-forward --train-days 504 --test-days 126 --step-days 63
python -m app.signal --universe nasdaq100 --refresh-data --top-n 10 --execution next_open
python -m app.live --mode paper --universe nasdaq100 --execution next_open
```
python -m app.signal --universe nasdaq100 --refresh-data --top-n 10 --execution next_open --max-pe 35

    -   python -m app.signal --universe nasdaq100 --refresh-data --top-n 10 --execution next_open
    -   results/live_signal.csv
## Outputs

- `data/`: cached market data
- `results/`: backtest, experiment, robustness outputs
- `results/live_signal.csv`: latest generated signal snapshot
- `logs/`: paper/live logs, state, and reports

Daily paper run outputs include:

- `logs/paper_trades.csv`
- `logs/paper_positions.csv`
- `logs/paper_equity.csv`
- `logs/reports/YYYY-MM-DD_report.md`
- `logs/state.json`

## Strategy and Defaults

Current default engine behavior:

- Execution: `next_open`
- Price mode: `adj`
- Rebalance: `ME`
- Slippage mode: `bps` (base 5 bps)
- Cash buffer: `1%`
- Target annual volatility: `0.15`
- Regime symbol: `QQQ`

The live signal and portfolio engine use a momentum ranking framework with dispersion and regime filters. Legacy SMA crossover components also exist for single-symbol backtest flows.

## Live Broker Safety (Optional)

Live broker orders are blocked unless all conditions are met:

- `--mode live`
- `--i-accept-real-trading YES`
- environment variable `LIVE_TRADING_ENABLED=true`

Alpaca environment variables:

- `ALPACA_API_KEY`
- `ALPACA_API_SECRET`
- `ALPACA_BASE_URL` (optional; defaults to paper endpoint)

Global kill switch:

- `KILL_SWITCH=true` prevents order placement and logs an alert.

## Dashboard

```bash
pip install -e .[dashboard]
streamlit run app/dashboard.py
```

## Docker

Build image:

```bash
docker build -t neetechs-trading .
```

Run backtest:

```bash
docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/results:/app/results -v ${PWD}/logs:/app/logs neetechs-trading python -m app.backtest --universe small
```

Run experiment:

```bash
docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/results:/app/results -v ${PWD}/logs:/app/logs neetechs-trading python -m app.experiment --universe small --walk-forward
```

Run paper live cycle:

```bash
docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/results:/app/results -v ${PWD}/logs:/app/logs neetechs-trading python -m app.live --mode paper --universe small --execution next_open
```

## CI and Nightly

- CI workflow: `.github/workflows/ci.yml` (ruff + pytest on Python 3.10/3.11/3.12)
- Nightly experiments: `.github/workflows/nightly.yml`

To enable nightly runs, set repository variable:

- `NIGHTLY_ENABLED=true`

## Notes

- For educational and research use.
- Not financial advice.



python -m app.signal --universe nasdaq100 --refresh-data --top-n 15 --execution next_open --momentum-months 60 --max-pe 30 --dispersion-threshold-multiplier 0.85