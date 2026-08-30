"""Chronological evaluation helpers; no parameter search is performed here."""

from __future__ import annotations

import pandas as pd

from .backtest import BacktestResult, run_backtest
from .config import BacktestConfig, StrategyConfig
from .strategy import dollar_neutral_notionals, generate_signal


def chronological_split(prices: pd.DataFrame, train_fraction: float = 0.70) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data strictly by timestamp for in-sample research and OOS testing."""
    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be in [0.5, 1.0).")
    split = int(len(prices) * train_fraction)
    if split < 2 or len(prices) - split < 2:
        raise ValueError("Both train and test samples need at least two observations.")
    return prices.iloc[:split].copy(), prices.iloc[split:].copy()


def walk_forward_returns(
    prices: pd.DataFrame,
    strategy_config: StrategyConfig,
    backtest_config: BacktestConfig,
    *,
    initial_train_days: int = 504,
    test_window_days: int = 126,
) -> pd.Series:
    """Run expanding-history, fixed-rule walk-forward tests and join OOS returns.

    The strategy settings are fixed before this function runs. Every test block
    can only use data ending at the beginning of that block for its first trade;
    rolling residual estimates update subsequently with then-observed data.
    """
    if initial_train_days < strategy_config.lookback_days + 2:
        raise ValueError("Initial training period must cover the lookback window.")
    return walk_forward_backtest(
        prices,
        strategy_config,
        backtest_config,
        initial_train_days=initial_train_days,
        test_window_days=test_window_days,
    ).daily["daily_return"].rename("walk_forward_return")


def walk_forward_backtest(
    prices: pd.DataFrame,
    strategy_config: StrategyConfig,
    backtest_config: BacktestConfig,
    *,
    initial_train_days: int = 504,
    test_window_days: int = 126,
) -> BacktestResult:
    """Return a metric-ready rebased result from fixed-rule walk-forward blocks.

    Each expanding-history out-of-sample block is run by the same delayed-fill
    backtester. Block returns are then compounded into one research-only equity
    path; no parameters are selected or optimized in this procedure.
    """
    if initial_train_days < strategy_config.lookback_days + 2:
        raise ValueError("Initial training period must cover the lookback window.")
    blocks: list[pd.DataFrame] = []
    trades: list[pd.DataFrame] = []
    test_start = initial_train_days
    while test_start < len(prices) - 1:
        test_end = min(test_start + test_window_days, len(prices))
        # Include one prior observation to carry an already-known target into OOS.
        window = prices.iloc[:test_end]
        signal = generate_signal(window, strategy_config)
        segment = window.iloc[test_start - 1 : test_end]
        result = run_backtest(segment, signal.loc[segment.index], backtest_config)
        blocks.append(result.daily.iloc[1:].copy())
        if not result.trades.empty:
            trades.append(result.trades.loc[result.trades["timestamp"] >= segment.index[1]].copy())
        test_start = test_end
    if not blocks:
        raise ValueError("Not enough observations for a walk-forward block.")

    daily = pd.concat(blocks).sort_index()
    equity = backtest_config.initial_capital
    rebased_equity: list[float] = []
    rebased_pnl: list[float] = []
    rebased_etf_notional: list[float] = []
    rebased_futures_notional: list[float] = []
    for _, row in daily.iterrows():
        prior_equity = equity
        equity *= 1.0 + float(row["daily_return"])
        etf_notional, futures_notional = dollar_neutral_notionals(
            prior_equity, float(row["position"]), backtest_config.gross_exposure
        )
        rebased_equity.append(equity)
        rebased_pnl.append(equity - prior_equity)
        rebased_etf_notional.append(etf_notional)
        rebased_futures_notional.append(futures_notional)
    daily["equity"] = rebased_equity
    daily["daily_pnl"] = rebased_pnl
    daily["etf_notional"] = rebased_etf_notional
    daily["futures_notional"] = rebased_futures_notional
    trade_ledger = (
        pd.concat(trades, ignore_index=True)
        if trades
        else pd.DataFrame(columns=["timestamp", "previous_direction", "new_direction", "turnover", "cost"])
    )
    return BacktestResult(daily=daily, trades=trade_ledger)


def run_baseline(
    prices: pd.DataFrame, strategy_config: StrategyConfig, backtest_config: BacktestConfig
) -> BacktestResult:
    """Generate fixed baseline targets and evaluate them with delayed execution."""
    return run_backtest(prices, generate_signal(prices, strategy_config), backtest_config)
