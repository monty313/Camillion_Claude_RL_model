# Shared synthetic leak-safe cache builder for the audit tests.
import numpy as np, pandas as pd
from src.data.cache_builder import build_aligned_indicators
def cache(n=800, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="1min")
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.05)
    df = pd.DataFrame({"open": close, "high": close + 0.03, "low": close - 0.03,
                       "close": close, "volume": 1.0}, index=idx)
    return (build_aligned_indicators(df), df["close"].values.astype("float32"),
            df.index.values.astype("datetime64[ns]").astype("int64"))
