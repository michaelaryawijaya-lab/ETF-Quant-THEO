"""Immutable, explicit configuration for reproducible experiments."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    """Market-data identifiers and alignment policy."""

    etf_ticker: str = "SPY"
    futures_ticker: str = "ES=F"
    fallback_futures_ticker: str = "^GSPC"
    start: str = "2012-01-01"
    end: str | None = None
    cache_dir: Path = Path("data/raw")
    require_complete_rows: bool = True


@dataclass(frozen=True)
class StrategyConfig:
    """Baseline, economically interpretable strategy assumptions."""

    lookback_days: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    annual_risk_free_rate: float = 0.04
    annual_dividend_yield: float = 0.015
    days_to_expiry: int = 45
    futures_to_etf_scale: float = 10.0
    max_abs_z: float = 6.0


@dataclass(frozen=True)
class BacktestConfig:
    """Conservative accounting and execution assumptions."""

    initial_capital: float = 1_000_000.0
    gross_exposure: float = 1.0
    etf_one_way_cost_bps: float = 1.0
    futures_one_way_cost_bps: float = 1.0
    etf_half_spread_bps: float = 0.5
    futures_half_spread_bps: float = 0.25
    annual_financing_rate: float = 0.04
    trading_days: int = 252
    max_gross_exposure: float = 1.0
