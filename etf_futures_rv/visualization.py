"""Reproducible diagnostic plots for the research report and notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .backtest import BacktestResult
from .metrics import rolling_metrics


def create_report_figures(prices: pd.DataFrame, signal: pd.DataFrame, result: BacktestResult, output_dir: str | Path) -> list[Path]:
    """Save price/signal and performance diagnostics as portable PNG figures."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    figures: list[Path] = []

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(prices.index, prices["etf_close"], label="SPY adjusted close")
    axes[0].plot(prices.index, prices["futures_close"] / 10, label="ES=F / 10")
    axes[0].set_title("Normalized price series")
    axes[0].legend()
    axes[1].plot(signal.index, signal["carry_residual"] * 10_000, label="Carry residual")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("basis (bps)")
    axes[2].plot(signal.index, signal["zscore"], label="z-score")
    axes[2].scatter(signal.index[signal["etf_weight"] != 0], signal.loc[signal["etf_weight"] != 0, "zscore"], s=8, label="target active")
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].legend()
    fig.tight_layout()
    path = destination / "prices_basis_signal.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(path)

    daily = result.daily
    drawdown = daily["equity"] / daily["equity"].cummax() - 1
    rolling = rolling_metrics(daily["daily_return"])
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(daily.index, daily["equity"], label="Equity")
    axes[0].set_title("Equity curve")
    axes[1].fill_between(drawdown.index, drawdown, 0, alpha=0.35)
    axes[1].set_title("Drawdown")
    axes[2].plot(rolling.index, rolling["rolling_sharpe"], label="63-day Sharpe")
    axes[2].plot(rolling.index, rolling["rolling_volatility"], label="63-day volatility")
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].legend()
    fig.tight_layout()
    path = destination / "equity_risk.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(path)
    return figures
