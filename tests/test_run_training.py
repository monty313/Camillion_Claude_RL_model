# The one-command trainer (run_training.py): it must find the data, build the caches, and fail
# FRIENDLY (one clear instruction) when the training engine isn't installed. No SB3 needed here.
import os
import tempfile
import numpy as np
import pandas as pd
import run_training


def _make_csv(path, n=400, base=1.10):
    idx = pd.date_range("2026-03-02 00:00", periods=n, freq="1min")
    cl = base + np.cumsum(np.random.default_rng(0).standard_normal(n) * 0.0003)
    pd.DataFrame({"datetime": idx, "open": cl, "high": cl + 0.01, "low": cl - 0.01,
                  "close": cl, "volume": 1.0}).to_csv(path, index=False)


def test_find_csv_matches_by_symbol_name():
    d = tempfile.mkdtemp()
    open(os.path.join(d, "my_EURUSD_data.csv"), "w").close()
    assert run_training._find_csv(d, "EURUSD").endswith("my_EURUSD_data.csv")
    assert run_training._find_csv(d, "GBPUSD") is None


def test_prepare_caches_finds_and_builds():
    from src.data.cache_builder import load_cache
    d = tempfile.mkdtemp()
    _make_csv(os.path.join(d, "EURUSD_1m.csv"), base=1.10)
    _make_csv(os.path.join(d, "US30_1m.csv"), base=38000.0)
    found = run_training.prepare_caches(d, ["EURUSD", "US30", "XAUUSD"], cache_dir=os.path.join(d, "cache"))
    assert set(found) == {"EURUSD", "US30"}                  # XAUUSD skipped (no file) — graceful
    ind, close, t = load_cache(os.path.join(d, "cache"), "EURUSD")
    assert ind.shape[1] == 220 and len(close) == 400


def test_main_fails_friendly_without_training_engine():
    try:
        import stable_baselines3  # noqa: F401
        print("SKIP test_main_fails_friendly_without_training_engine: SB3 is installed")
        return
    except Exception:
        pass
    d = tempfile.mkdtemp()
    _make_csv(os.path.join(d, "EURUSD_1m.csv"))
    try:
        run_training.main(["--data", d, "--symbols", "EURUSD", "--cache", os.path.join(d, "cache")])
        assert False, "expected a friendly SystemExit"
    except SystemExit as e:
        assert "stable-baselines3" in str(e)                 # ONE clear instruction, not a stack trace
