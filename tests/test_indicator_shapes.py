# Test 7: indicator outputs align with input bar count (+ 190 column contract).
import numpy as np
from config import constants as C
from src.indicators import base
from src.indicators.sma import sma


def test_indicator_column_counts():
    assert len(base.PER_TF_COLUMNS) == C.N_INDICATORS_PER_TF == 40
    assert len(base.ALL_INDICATOR_COLUMNS) == C.N_INDICATORS_TOTAL == 200


def test_outputs_align_with_bar_count():
    for n in (50, 250, 1000):
        close = np.cumsum(np.random.randn(n)) + 100.0
        m = base.compute_timeframe_indicators(close, close, close)
        assert m.shape == (n, 40)
        assert m.dtype == np.float32


def test_sma_real_value_and_shift():
    x = np.arange(10, dtype=float)
    assert np.allclose(sma(x, 1, 0), x.astype(np.float32))      # period1 == series
    s = sma(x, 1, shift=2)                                       # value from 2 bars ago
    assert np.isnan(s[0]) and np.isnan(s[1]) and s[2] == 0.0 and s[9] == 7.0
    s2 = sma(x, 2, 0)
    assert np.isnan(s2[0]) and s2[1] == 0.5 and s2[2] == 1.5
