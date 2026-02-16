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
