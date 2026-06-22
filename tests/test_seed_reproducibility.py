# Audit S5.23: same seed + same actions -> byte-identical obs and rewards.
import numpy as np
try:
    from tests._audit_helpers import cache
except ImportError:  # stdlib runner / Colab load test as top-level module
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from _audit_helpers import cache
from src.env.trading_env import TradingEnv
from src.strategies.registry import AlphaRegistry

def _run():
    ind, cl, t = cache()
    e = TradingEnv(ind, cl, t, AlphaRegistry(), warmup=210, position_size=100000.0)
    e.reset(seed=42); rg = np.random.default_rng(99); O, R = [], []
    for _ in range(40):
        o, r, te, tr, _ = e.step(int(rg.integers(0, 4))); O.append(o.copy()); R.append(r)
        if te or tr:
            e.reset(seed=42)
    return np.array(O), np.array(R)

def test_same_seed_identical():
    o1, r1 = _run(); o2, r2 = _run()
    assert np.array_equal(o1, o2) and np.array_equal(r1, r2)
