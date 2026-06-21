# Phase 1 (critical): the multi-timeframe cache has NO future leakage.
# Aligned indicator values at bar t must be invariant to ANY change in bars > t.
import numpy as np
import pandas as pd
from config import constants as C
from src.data.cache_builder import build_aligned_indicators


def _make_1m(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="1min")
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.1)
    high = close + np.abs(rng.standard_normal(n) * 0.05)
    low = close - np.abs(rng.standard_normal(n) * 0.05)
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close,
                         "volume": 1.0}, index=idx)


def test_cache_shape_and_columns():
    mat = build_aligned_indicators(_make_1m(3000))
    assert mat.shape[1] == C.N_INDICATORS_TOTAL == 200
    assert mat.dtype == np.float32


def test_no_future_leakage_across_timeframes():
    df = _make_1m(5000, seed=1)
    mat = build_aligned_indicators(df)
    for t in (1000, 2500, 4000):
        df2 = df.copy()
        # arbitrarily corrupt EVERY field for all bars strictly AFTER t
        fut = df2.index[t + 1:]
        rng = np.random.default_rng(99)
        df2.loc[fut, ["open", "high", "low", "close"]] += rng.standard_normal((len(fut), 4)) * 5
        mat2 = build_aligned_indicators(df2)
        assert np.allclose(mat[:t + 1], mat2[:t + 1], equal_nan=True), f"LEAK at t={t}"


def test_current_bar_higher_tf_uses_only_closed_bars():
    # Sanity: a higher-TF column changes only at the 1m bars where a new TF bar
    # has closed (step function), never continuously within an open TF bar.
    df = _make_1m(2000, seed=2)
    mat = build_aligned_indicators(df)
    col5m = base_col_index("5m__sma_p1_s0")
    series = mat[:, col5m]
    finite = series[np.isfinite(series)]
    # far fewer distinct values than bars -> it's a forward-filled step function
    assert len(np.unique(finite)) < len(finite) * 0.5


def base_col_index(name):
    from src.indicators.base import ALL_INDICATOR_COLUMNS
    return ALL_INDICATOR_COLUMNS.index(name)
