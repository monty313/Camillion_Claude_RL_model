# Phase 3: real-CSV loader -> build_aligned_indicators end-to-end (for the single-symbol baseline).
import os, tempfile
import numpy as np, pandas as pd
from src.data.cache_builder import load_ohlcv_csv, build_aligned_indicators

def test_load_ohlcv_csv_roundtrip():
    d = tempfile.mkdtemp(); idx = pd.date_range("2026-01-01", periods=400, freq="1min")
    cl = 100 + np.cumsum(np.random.default_rng(0).standard_normal(len(idx)) * 0.05)
    pd.DataFrame({"datetime": idx, "open": cl, "high": cl + .03, "low": cl - .03,
                  "close": cl, "volume": 1.0}).to_csv(os.path.join(d, "X.csv"), index=False)
    df = load_ohlcv_csv(os.path.join(d, "X.csv"))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 400 and df.index.is_monotonic_increasing
    ind = build_aligned_indicators(df)
    assert ind.shape == (400, 200)
