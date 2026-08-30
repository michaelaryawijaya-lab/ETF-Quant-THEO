"""Signal formation, hedge construction, and risk controls."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .research import fair_value_residual, rolling_zscore


def generate_signal(prices: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Create a stateful mean-reversion signal from the trailing carry residual.

    ``etf_weight`` of +1 denotes long SPY / short futures, appropriate when
    futures are rich versus carry fair value. Values are targets, not fills;
    the backtest delays execution by one bar.
    """
    residual = fair_value_residual(prices, config)
    zscore = rolling_zscore(residual, config.lookback_days).clip(-config.max_abs_z, config.max_abs_z)
    target = pd.Series(0.0, index=prices.index, name="etf_weight")
    state = 0.0
    for timestamp, value in zscore.items():
        if not np.isfinite(value):
            target.at[timestamp] = state
            continue
        if state == 0.0 and value >= config.entry_z:
            state = 1.0
        elif state == 0.0 and value <= -config.entry_z:
            state = -1.0
        elif state != 0.0 and abs(value) <= config.exit_z:
            state = 0.0
        target.at[timestamp] = state
    result = pd.DataFrame({"carry_residual": residual, "zscore": zscore, "etf_weight": target})
    result["futures_weight"] = -result["etf_weight"]
    return result


def dollar_neutral_notionals(equity: float, direction: float, gross_exposure: float) -> tuple[float, float]:
    """Return equal-and-opposite ETF/futures notionals under a gross cap."""
    if equity < 0:
        raise ValueError("Equity cannot be negative.")
    if not 0.0 <= gross_exposure <= 2.0:
        raise ValueError("gross_exposure must be between 0 and 2.")
    leg = equity * gross_exposure / 2.0
    return direction * leg, -direction * leg


def validate_signal(signal: pd.DataFrame) -> None:
    """Ensure targets are a bounded, dollar-neutral direction vector."""
    required = {"etf_weight", "futures_weight"}
    if not required.issubset(signal.columns):
        raise ValueError(f"signal must contain {sorted(required)}")
    if not np.allclose(signal["etf_weight"].fillna(0), -signal["futures_weight"].fillna(0)):
        raise ValueError("ETF and futures weights must be equal and opposite.")
    if (signal[["etf_weight", "futures_weight"]].abs() > 1.0).any().any():
        raise ValueError("Signal weights must lie in [-1, 1].")
