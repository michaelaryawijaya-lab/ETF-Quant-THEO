from __future__ import annotations

import numpy as np
import pytest

from etf_futures_rv.config import BacktestConfig, StrategyConfig
from etf_futures_rv.metrics import drawdown_series
from etf_futures_rv.research import normalized_price_series, theoretical_fair_value
from etf_futures_rv.validation import walk_forward_backtest, walk_forward_returns


def test_theoretical_fair_value_and_normalized_prices(prices) -> None:
    config = StrategyConfig(annual_risk_free_rate=0.04, annual_dividend_yield=0.01, days_to_expiry=30)
    fair_value = theoretical_fair_value(prices, config)
    normalized = normalized_price_series(prices, config)
    assert (fair_value > prices["etf_close"]).all()
    np.testing.assert_allclose(normalized.iloc[0].to_numpy(), [100.0, 100.0, 100.0])


def test_walk_forward_backtest_is_metric_ready(prices) -> None:
    strategy = StrategyConfig(lookback_days=20)
    backtest = BacktestConfig(annual_financing_rate=0)
    result = walk_forward_backtest(prices, strategy, backtest, initial_train_days=60, test_window_days=20)
    returns = walk_forward_returns(prices, strategy, backtest, initial_train_days=60, test_window_days=20)
    assert result.daily.index.is_monotonic_increasing
    assert len(result.daily) == len(returns)
    assert {"equity", "daily_return", "turnover", "etf_notional", "futures_notional"}.issubset(result.daily.columns)


def test_drawdown_series_starts_at_zero_and_tracks_new_peak() -> None:
    import pandas as pd

    equity = pd.Series([100.0, 110.0, 99.0, 120.0])
    drawdown = drawdown_series(equity)
    assert drawdown.iloc[0] == 0.0
    assert drawdown.iloc[2] == pytest.approx(-0.1)
    assert drawdown.iloc[-1] == 0.0
