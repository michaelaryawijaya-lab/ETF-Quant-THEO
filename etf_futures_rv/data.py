"""Download, validate, and timestamp-align public daily market data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DataConfig


REQUIRED_COLUMNS = {"etf_close", "futures_close"}


def _as_close_series(frame: pd.DataFrame, name: str) -> pd.Series:
    """Extract a close column from yfinance output, including MultiIndex output."""
    if isinstance(frame.columns, pd.MultiIndex):
        close_columns = [col for col in frame.columns if col[0] in {"Adj Close", "Close"}]
        if not close_columns:
            raise ValueError(f"No Close or Adj Close column available for {name}.")
        column = next((col for col in close_columns if col[0] == "Adj Close"), close_columns[0])
        series = frame[column]
    else:
        column = "Adj Close" if "Adj Close" in frame else "Close"
        if column not in frame:
            raise ValueError(f"No Close or Adj Close column available for {name}.")
        series = frame[column]
    return pd.to_numeric(series, errors="coerce").rename(name)


def normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Return sorted, unique daily observations without fabricating missing prices.

    This deliberately does not forward-fill. Filling a closed-market or a missing
    futures quote could create artificial returns and hide data-quality issues.
    """
    result = prices.copy()
    result.index = pd.to_datetime(result.index).tz_localize(None).normalize()
    result = result[~result.index.duplicated(keep="last")].sort_index()
    result = result.apply(pd.to_numeric, errors="coerce")
    return result


def align_prices(etf: pd.Series, futures: pd.Series, *, require_complete_rows: bool = True) -> pd.DataFrame:
    """Inner-join only contemporaneous prices and validate the resulting panel."""
    aligned = pd.concat([etf.rename("etf_close"), futures.rename("futures_close")], axis=1, join="inner")
    aligned = normalize_prices(aligned)
    if require_complete_rows:
        aligned = aligned.dropna(how="any")
    if not REQUIRED_COLUMNS.issubset(aligned.columns):
        raise ValueError("Aligned data must include ETF and futures closes.")
    if aligned.empty:
        raise ValueError("No valid overlapping observations after timestamp alignment.")
    if (aligned <= 0).any().any():
        raise ValueError("Prices must be strictly positive.")
    return aligned


def download_public_data(config: DataConfig, *, use_fallback: bool = False) -> pd.DataFrame:
    """Download SPY and a public continuous futures proxy, then cache raw closes.

    `ES=F` is Yahoo Finance's continuous front-month E-mini S&P 500 futures
    series. It is a research proxy rather than an executable contract history.
    Setting ``use_fallback`` uses the cash index `^GSPC` if ES data is absent.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError("Install yfinance to download public data: pip install -r requirements.txt") from exc

    futures_ticker = config.fallback_futures_ticker if use_fallback else config.futures_ticker
    tickers = [config.etf_ticker, futures_ticker]
    raw = yf.download(tickers, start=config.start, end=config.end, auto_adjust=False, progress=False)
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no data. Try a different date range or fallback proxy.")

    if isinstance(raw.columns, pd.MultiIndex):
        etf_frame = raw.xs(config.etf_ticker, axis=1, level=-1, drop_level=True)
        futures_frame = raw.xs(futures_ticker, axis=1, level=-1, drop_level=True)
    else:  # defensive branch for a one-ticker provider response
        raise RuntimeError("Expected multi-ticker response from data provider.")
    # Adjusted SPY closes include split and dividend adjustments; futures are not adjusted.
    aligned = align_prices(
        _as_close_series(etf_frame, "etf_close"),
        _as_close_series(futures_frame, "futures_close"),
        require_complete_rows=config.require_complete_rows,
    )
    cache_path = Path(config.cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(cache_path / f"{config.etf_ticker}_{futures_ticker.replace('^', '')}_daily.csv")
    return aligned


def load_prices_csv(path: str | Path) -> pd.DataFrame:
    """Load a cached aligned panel created by :func:`download_public_data`."""
    data = pd.read_csv(path, index_col=0, parse_dates=True)
    if not REQUIRED_COLUMNS.issubset(data.columns):
        raise ValueError(f"{path} must contain {sorted(REQUIRED_COLUMNS)}")
    return align_prices(data["etf_close"], data["futures_close"])
