# Phase 2: walk-forward harness windows correctly and returns a pass-rate.
import numpy as np
import pandas as pd
from config import constants as C
from src.training.walk_forward import make_windows, run
from src.strategies.registry import AlphaRegistry
from src.strategies.examples import register_examples


def _data(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="1min")
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.1)
    ind = rng.standard_normal((n, C.N_INDICATORS_TOTAL)).astype(np.float32)
    return ind, close.astype(np.float32), idx.values.astype("datetime64[ns]").astype(np.int64)


def _hold_policy(obs):
    return np.zeros(C.N_ACTIONS), 0.0   # argmax -> HOLD


def test_make_windows():
    w = make_windows(1000, 400, 100, 200, 150)
    assert len(w) > 0
    assert all(x["test"][1] - x["test"][0] == 200 for x in w)


def test_walk_forward_runs():
    ind, close, t = _data()
    def rf():
        r = AlphaRegistry(); register_examples(r); return r
    out = run(ind, close, t, rf, _hold_policy, train=400, val=100, test=200,
              step=150, target_pct=0.0, warmup=20, max_steps=180)
    assert out["n_windows"] > 0 and 0.0 <= out["pass_rate"] <= 1.0
    assert all("passed" in r and "breached" in r for r in out["results"])
