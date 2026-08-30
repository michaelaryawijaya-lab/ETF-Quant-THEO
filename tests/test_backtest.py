from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf_futures_rv.backtest import run_backtest
from etf_futures_rv.config import BacktestConfig, StrategyConfig
from etf_futures_rv.strategy import generate_signal


def _zero_cost_config() -> BacktestConfig:
    return BacktestConfig(etf_one_way_cost_bps=0, futures_one_way_cost_bps=0, etf_half_spread_bps=0, futures_half_spread_bps=0, annual_financing_rate=0)


def test_execution_is_delayed_one_full_bar() -> None:
    index = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"etf_close": [100.0, 101.0, 102.0], "futures_close": [1000.0, 1000.0, 1000.0]}, index=index)
    signal = pd.DataFrame({"etf_weight": [1.0, 1.0, 1.0], "futures_weight": [-1.0, -1.0, -1.0]}, index=index)
    result = run_backtest(prices, signal, _zero_cost_config())
    assert result.daily.iloc[1]["daily_pnl"] == 0.0
    expected = 500_000 * (102 / 101 - 1)
    assert result.daily.iloc[2]["daily_pnl"] == pytest.approx(expected)


def test_backtest_costs_reduce_equity() -> None:
    index = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"etf_close": [100.0, 100.0, 100.0], "futures_close": [1000.0, 1000.0, 1000.0]}, index=index)
    signal = pd.DataFrame({"etf_weight": [1.0, 1.0, 1.0], "futures_weight": [-1.0, -1.0, -1.0]}, index=index)
    result = run_backtest(prices, signal, BacktestConfig(annual_financing_rate=0))
    assert result.daily["equity"].iloc[-1] < 1_000_000
    assert len(result.trades) == 1


def test_future_prices_cannot_change_past_backtest_results(prices) -> None:
    config = StrategyConfig(lookback_days=20)
    original_signal = generate_signal(prices, config)
    original = run_backtest(prices, original_signal, _zero_cost_config()).daily
    altered_prices = prices.copy()
    altered_prices.iloc[-1, altered_prices.columns.get_loc("futures_close")] *= 1.5
    altered_signal = generate_signal(altered_prices, config)
    altered = run_backtest(altered_prices, altered_signal, _zero_cost_config()).daily
    pd.testing.assert_frame_equal(original.iloc[:-1], altered.iloc[:-1])
