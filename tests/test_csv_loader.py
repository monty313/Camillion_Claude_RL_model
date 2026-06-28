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
    assert ind.shape == (400, 220)


def test_load_ohlcv_csv_metatrader5_export():
    """MT5 history export: TAB-separated, <...> headers, SPLIT <DATE>/<TIME>, dotted dates, real <VOL>=0.
    Must keep 1-minute resolution (NOT collapse to one bar/day) and use TICK volume."""
    d = tempfile.mkdtemp()
    p = os.path.join(d, "EURUSD_M1_2021_2026.csv")
    lines = ["<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>"]
    base = pd.Timestamp("2021-01-13 11:30:00")
    for i in range(120):                                  # spans >1 hour across the day boundary logic
        t = base + pd.Timedelta(minutes=i)
        lines.append(f"{t.strftime('%Y.%m.%d')}\t{t.strftime('%H:%M:%S')}\t1.21\t1.215\t1.208\t1.212\t{50 + i}\t0\t3")
    open(p, "w").write("\n".join(lines) + "\n")
    df = load_ohlcv_csv(p)
    assert len(df) == 120, f"lost rows -> date/time mis-parse (got {len(df)})"     # not collapsed to 1/day
    assert (df.index[1] - df.index[0]) == pd.Timedelta(minutes=1)                  # minute resolution kept
    assert df.index[0] == pd.Timestamp("2021-01-13 11:30:00")
    assert df["volume"].iloc[0] == 50.0 and float(df["close"].iloc[0]) == 1.212    # tick volume, not <VOL>=0
    assert build_aligned_indicators(df).shape == (120, 220)


def test_load_ohlcv_csv_semicolon_and_no_volume():
    """Semicolon-separated, no volume column -> volume defaults to 1.0, still loads."""
    d = tempfile.mkdtemp()
    p = os.path.join(d, "s.csv")
    idx = pd.date_range("2026-01-01", periods=30, freq="1min")
    pd.DataFrame({"timestamp": idx, "Open": 1.0, "High": 1.1, "Low": 0.9, "Close": 1.05}
                 ).to_csv(p, index=False, sep=";")
    df = load_ohlcv_csv(p)
    assert len(df) == 30 and (df["volume"] == 1.0).all()
