"""Download data and generate a reproducible fixed-parameter baseline report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from etf_futures_rv.backtest import run_backtest
from etf_futures_rv.config import BacktestConfig, DataConfig, StrategyConfig
from etf_futures_rv.data import download_public_data
from etf_futures_rv.metrics import performance_metrics
from etf_futures_rv.research import cointegration_test, descriptive_statistics, stationarity_test
from etf_futures_rv.strategy import generate_signal
from etf_futures_rv.validation import chronological_split, walk_forward_returns
from etf_futures_rv.visualization import create_report_figures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fallback-index", action="store_true", help="Use ^GSPC if ES=F data cannot be downloaded.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for report artifacts.")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_config, strategy_config, backtest_config = DataConfig(), StrategyConfig(), BacktestConfig()
    prices = download_public_data(data_config, use_fallback=args.fallback_index)
    train, test = chronological_split(prices)
    signal = generate_signal(prices, strategy_config)
    result = run_backtest(test, signal.loc[test.index], backtest_config)
    report = {
        "data_rows": len(prices),
        "train_rows": len(train),
        "test_rows": len(test),
        "research_train": descriptive_statistics(train, strategy_config),
        "stationarity_train": stationarity_test(signal.loc[train.index, "carry_residual"]),
        "cointegration_train": cointegration_test(train),
        "out_of_sample_metrics": performance_metrics(result),
        "walk_forward_mean_daily_return": float(walk_forward_returns(prices, strategy_config, backtest_config).mean()),
    }
    (output_dir / "baseline_report.json").write_text(json.dumps(report, indent=2, default=str))
    create_report_figures(test, signal.loc[test.index], result, output_dir)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
