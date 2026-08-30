from __future__ import annotations

import numpy as np

from etf_futures_rv.config import StrategyConfig
from etf_futures_rv.strategy import dollar_neutral_notionals, generate_signal, validate_signal


def test_signal_is_dollar_neutral_and_bounded(prices) -> None:
    signal = generate_signal(prices, StrategyConfig(lookback_days=20))
    validate_signal(signal)
    assert (signal["etf_weight"] == -signal["futures_weight"]).all()
    assert signal["etf_weight"].abs().max() <= 1.0


def test_long_etf_notional_is_offset_by_short_futures() -> None:
    etf, futures = dollar_neutral_notionals(1_000_000, 1.0, 1.0)
    assert etf == 500_000
    assert futures == -500_000


def test_future_price_change_cannot_change_prior_signal(prices) -> None:
    config = StrategyConfig(lookback_days=20)
    original = generate_signal(prices, config)
    changed = prices.copy()
    changed.iloc[-1, changed.columns.get_loc("futures_close")] *= 1.5
    altered = generate_signal(changed, config)
    np.testing.assert_allclose(original.iloc[:-1]["zscore"], altered.iloc[:-1]["zscore"], equal_nan=True)
    assert original.iloc[:-1]["etf_weight"].equals(altered.iloc[:-1]["etf_weight"])
