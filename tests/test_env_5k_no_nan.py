# Audit S1.5: 5000 random steps -> never any NaN/Inf in obs or reward.
import numpy as np
try:
    from tests._audit_helpers import cache
except ImportError:  # stdlib runner / Colab load test as top-level module
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from _audit_helpers import cache
from src.env.trading_env import TradingEnv
from src.strategies.registry import AlphaRegistry

def test_5000_steps_no_nan():
    ind, cl, t = cache(n=2000)
    e = TradingEnv(ind, cl, t, AlphaRegistry(), warmup=210)
    e.reset(); rg = np.random.default_rng(1); bad = 0
    for _ in range(5000):
        o, r, te, tr, _ = e.step(int(rg.integers(0, 4)))
        if not np.all(np.isfinite(o)) or not np.isfinite(r):
            bad += 1
        if te or tr:
            e.reset()
    assert bad == 0
