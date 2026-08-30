from __future__ import annotations

from etf_futures_rv.backtest import run_backtest
from etf_futures_rv.config import BacktestConfig, StrategyConfig
from etf_futures_rv.metrics import performance_metrics
from etf_futures_rv.strategy import generate_signal


def test_metrics_include_requested_core_fields(prices) -> None:
    signal = generate_signal(prices, StrategyConfig(lookback_days=20))
    result = run_backtest(prices, signal, BacktestConfig(annual_financing_rate=0))
    metrics = performance_metrics(result)
    expected = {"total_return", "annualized_return", "annualized_volatility", "sharpe_ratio", "sortino_ratio", "maximum_drawdown", "calmar_ratio", "win_rate", "profit_factor", "turnover", "average_gross_exposure", "number_of_trades"}
    assert expected.issubset(metrics)
