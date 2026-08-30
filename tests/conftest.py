from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def prices() -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=140)
    # Smooth, positive, closely linked synthetic levels with deterministic basis variation.
    etf = 300 + np.arange(len(index)) * 0.15
    futures = etf * 10 * np.exp(0.003 + 0.002 * np.sin(np.arange(len(index)) / 4))
    return pd.DataFrame({"etf_close": etf, "futures_close": futures}, index=index)
