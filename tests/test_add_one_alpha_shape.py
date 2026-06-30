# Audit S7.31: registering one alpha keeps shape (541,), fills its slot + mask,
# and the summary recomputes -- no retrain, no STATE_DIM change.
import numpy as np
try:
    from tests._audit_helpers import cache
except ImportError:  # stdlib runner / Colab load test as top-level module
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from _audit_helpers import cache
from src.env.trading_env import TradingEnv
from src.strategies.registry import AlphaRegistry
from src.strategies.base import BaseStrategy
from src.observation import observation_contract as OC


class _Buy(BaseStrategy):
    def compute_signal(self, ctx):
        return 1


def test_add_one_alpha_keeps_shape():
    ind, cl, t = cache()
    e0 = TradingEnv(ind, cl, t, AlphaRegistry(), warmup=210)
    o0, _ = e0.reset()
    assert o0.shape == (541,) and o0[OC.BLOCK_SLICES["alpha_mask"]].sum() == 0
    reg = AlphaRegistry(); slot = reg.register(_Buy("dummy"))
    e1 = TradingEnv(ind, cl, t, reg, warmup=210)
    o1, _ = e1.reset()
    assert o1.shape == (541,)                              # shape unchanged
    assert o1[OC.BLOCK_SLICES["alpha_mask"]][slot] == 1.0  # slot mask filled
    assert o1[OC.BLOCK_SLICES["alpha_values"]][slot] == 1.0
    s = o1[OC.BLOCK_SLICES["alpha_summary"]]
    assert abs(s[0] - 1.0) < 1e-6 and abs(s[2] - 1.0) < 1e-6   # buy%=1, active%=1
