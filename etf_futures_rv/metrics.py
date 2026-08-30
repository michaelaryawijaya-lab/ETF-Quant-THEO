"""Performance metrics calculated from a timestamped backtest ledger."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import BacktestResult


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a negative decimal return."""
    return float((equity / equity.cummax() - 1.0).min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Return the complete running peak-to-trough drawdown path."""
    return (equity / equity.cummax() - 1.0).rename("drawdown")


def performance_metrics(result: BacktestResult, trading_days: int = 252) -> dict[str, float]:
    """Calculate standard, explicitly defined metrics from daily returns."""
    daily = result.daily
    returns = daily["daily_return"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 2:
        raise ValueError("At least two daily returns are required.")
    total_return = float(daily["equity"].iloc[-1] / daily["equity"].iloc[0] - 1.0)
    years = len(returns) / trading_days
    annualized_return = float((1 + total_return) ** (1 / years) - 1) if total_return > -1 else -1.0
    volatility = float(returns.std(ddof=1) * np.sqrt(trading_days))
    downside = returns[returns < 0].std(ddof=1)
    sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(trading_days)) if returns.std(ddof=1) else np.nan
    sortino = float(returns.mean() / downside * np.sqrt(trading_days)) if downside and np.isfinite(downside) else np.nan
    drawdown = max_drawdown(daily["equity"])
    gross_profit = float(daily.loc[daily["daily_pnl"] > 0, "daily_pnl"].sum())
    gross_loss = float(-daily.loc[daily["daily_pnl"] < 0, "daily_pnl"].sum())
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "maximum_drawdown": drawdown,
        "calmar_ratio": annualized_return / abs(drawdown) if drawdown < 0 else np.nan,
        "win_rate": float((daily["daily_pnl"] > 0).mean()),
        "profit_factor": gross_profit / gross_loss if gross_loss else np.nan,
        "turnover": float(daily["turnover"].sum()),
        "average_gross_exposure": float((daily["etf_notional"].abs() + daily["futures_notional"].abs()).div(daily["equity"]).mean()),
        "number_of_trades": float(len(result.trades)),
    }


def rolling_metrics(daily_returns: pd.Series, window: int = 63, trading_days: int = 252) -> pd.DataFrame:
    """Trailing annualized volatility and zero-rate Sharpe for plotting."""
    volatility = daily_returns.rolling(window).std(ddof=1) * np.sqrt(trading_days)
    sharpe = daily_returns.rolling(window).mean() / daily_returns.rolling(window).std(ddof=1) * np.sqrt(trading_days)
    return pd.DataFrame({"rolling_volatility": volatility, "rolling_sharpe": sharpe})
