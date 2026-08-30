"""Economically interpretable diagnostics for the ETF/futures relationship."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

from .config import StrategyConfig


def theoretical_fair_value(prices: pd.DataFrame, config: StrategyConfig) -> pd.Series:
    """Return the ETF fair-value series implied by the cost-of-carry model."""
    if "etf_close" not in prices:
        raise ValueError("prices must contain etf_close")
    tau = config.days_to_expiry / 365.25
    carry = (config.annual_risk_free_rate - config.annual_dividend_yield) * tau
    return (prices["etf_close"] * np.exp(carry)).rename("fair_value")


def normalized_price_series(prices: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Rebase SPY, normalized futures, and fair value to 100 at the first date."""
    required = {"etf_close", "futures_close"}
    if not required.issubset(prices.columns):
        raise ValueError(f"prices must contain {sorted(required)}")
    levels = pd.DataFrame(
        {
            "SPY": prices["etf_close"],
            "Futures proxy / 10": prices["futures_close"] / config.futures_to_etf_scale,
            "Carry fair value": theoretical_fair_value(prices, config),
        }
    )
    return levels.div(levels.iloc[0]).mul(100.0)


def fair_value_residual(prices: pd.DataFrame, config: StrategyConfig) -> pd.Series:
    """Compute log(F / 10) minus log theoretical SPY fair value under carry.

    Futures prices are S&P 500 index points and SPY trades at roughly one tenth
    of the index. The ratio is therefore normalized before cost-of-carry is
    applied. The result is a dimensionless log residual.
    """
    required = {"etf_close", "futures_close"}
    if not required.issubset(prices.columns):
        raise ValueError(f"prices must contain {sorted(required)}")
    fair_etf = theoretical_fair_value(prices, config)
    normalized_futures = prices["futures_close"] / config.futures_to_etf_scale
    return np.log(normalized_futures / fair_etf).rename("carry_residual")


def rolling_zscore(series: pd.Series, lookback_days: int) -> pd.Series:
    """Trailing z-score using observations available at each timestamp only."""
    mean = series.rolling(lookback_days, min_periods=lookback_days).mean()
    std = series.rolling(lookback_days, min_periods=lookback_days).std(ddof=1)
    return ((series - mean) / std.replace(0.0, np.nan)).rename("zscore")


def stationarity_test(series: pd.Series) -> dict[str, float | int | bool]:
    """Augmented Dickey-Fuller test with a conventional 5% interpretation."""
    clean = series.dropna()
    if len(clean) < 30:
        raise ValueError("At least 30 observations are required for an ADF test.")
    statistic, pvalue, lags, observations, *_ = adfuller(clean, autolag="AIC")
    return {"statistic": float(statistic), "pvalue": float(pvalue), "lags": int(lags), "observations": int(observations), "stationary_at_5pct": bool(pvalue < 0.05)}


def cointegration_test(prices: pd.DataFrame) -> dict[str, float | bool]:
    """Engle-Granger test on log prices; interpret cautiously for rolled futures."""
    clean = prices[["etf_close", "futures_close"]].dropna()
    if len(clean) < 30:
        raise ValueError("At least 30 observations are required for a cointegration test.")
    statistic, pvalue, critical_values = coint(np.log(clean["etf_close"]), np.log(clean["futures_close"]))
    return {"statistic": float(statistic), "pvalue": float(pvalue), "critical_5pct": float(critical_values[1]), "cointegrated_at_5pct": bool(pvalue < 0.05)}


def descriptive_statistics(prices: pd.DataFrame, config: StrategyConfig) -> dict[str, float]:
    """Core price, return, correlation, and residual diagnostics."""
    residual = fair_value_residual(prices, config)
    returns = prices.pct_change().dropna()
    return {
        "observations": float(len(prices)),
        "price_correlation": float(prices["etf_close"].corr(prices["futures_close"])),
        "return_correlation": float(returns["etf_close"].corr(returns["futures_close"])),
        "residual_mean_bps": float(residual.mean() * 10_000),
        "residual_std_bps": float(residual.std(ddof=1) * 10_000),
    }
