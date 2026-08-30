from __future__ import annotations

import pandas as pd
import pytest

from etf_futures_rv.data import align_prices, normalize_prices


def test_alignment_uses_only_contemporaneous_complete_prices() -> None:
    etf = pd.Series([100.0, 101.0, 102.0], index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
    futures = pd.Series([None, 1020.0, 1030.0], index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    result = align_prices(etf, futures)
    assert result.index.tolist() == [pd.Timestamp("2024-01-03")]
    assert result.loc[pd.Timestamp("2024-01-03"), "etf_close"] == 102.0


def test_normalization_sorts_and_keeps_latest_duplicate() -> None:
    raw = pd.DataFrame({"price": [102.0, 100.0, 101.0]}, index=pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-02"]))
    result = normalize_prices(raw)
    assert result.index.is_monotonic_increasing
    assert result.loc[pd.Timestamp("2024-01-02"), "price"] == 101.0


def test_alignment_rejects_nonpositive_values() -> None:
    index = pd.date_range("2024-01-01", periods=2)
    with pytest.raises(ValueError, match="strictly positive"):
        align_prices(pd.Series([1.0, 0.0], index=index), pd.Series([10.0, 10.0], index=index))
