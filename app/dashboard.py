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
