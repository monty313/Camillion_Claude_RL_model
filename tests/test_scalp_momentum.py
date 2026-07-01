# v1.12.0 1m SCALP-MOMENTUM block (4 leak-free scores): the 1m entry-timing layer (bb_interactions starts 5m).
import numpy as np, pandas as pd
from config import constants as C
from src.observation import scalp_momentum as SM
from src.observation import observation_contract as OC
from src.data.cache_builder import build_aligned_indicators


def _real(n=6000, seed=2):
    idx = pd.date_range("2026-01-01", periods=n, freq="1min")
    cl = 100 + np.cumsum(np.r_[np.full(n // 2, 0.05), np.full(n - n // 2, -0.04)]) + \
        np.random.default_rng(seed).standard_normal(n) * 0.05
    df = pd.DataFrame({"open": cl, "high": cl + .05, "low": cl - .05, "close": cl, "volume": 1.}, index=idx)
    return build_aligned_indicators(df), cl.astype(np.float64)


def test_shape_names_bounds():
    ind, cl = _real()
    m = SM.compute_scalp_momentum(ind, cl)
    assert m.shape == (len(cl), C.OBS_BLOCK_SCALP_MOMENTUM) == (len(cl), 4)
    assert len(SM.SCALP_MOMENTUM_NAMES) == 4 and m.dtype == np.float32 and np.all(np.isfinite(m))
    assert m.min() >= -1.0 - 1e-6 and m.max() <= 1.0 + 1e-6   # tanh-bounded


def test_carries_signal():
    ind, cl = _real()
    m = SM.compute_scalp_momentum(ind, cl)
    nm = SM.SCALP_MOMENTUM_NAMES
    for s in ("scalp_fast_dist_1m", "scalp_fast_roc_1m"):
        assert np.ptp(m[300:, nm.index(s)]) > 0.05, s


def test_block_in_contract():
    sl = OC.BLOCK_SLICES["scalp_momentum"]                     # position shifts as later blocks append (v1.13.0+)
    assert sl.stop - sl.start == C.OBS_BLOCK_SCALP_MOMENTUM == 4
    assert OC.BLOCK_NAMES["scalp_momentum"] == list(SM.SCALP_MOMENTUM_NAMES)


def test_leak_free_prefix_invariance():
    ind, cl = _real(n=5000, seed=4)
    full = SM.compute_scalp_momentum(ind, cl)
    cut = 3800
    part = SM.compute_scalp_momentum(ind[:cut], cl[:cut])
    np.testing.assert_allclose(full[200:cut], part[200:cut], atol=1e-6)
