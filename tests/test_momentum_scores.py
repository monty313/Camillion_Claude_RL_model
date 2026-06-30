# v1.9.0 MOMENTUM-PERCEPTION block: 9 leak-free per-bar scores (one per the operator's momentum tree).
# These are STATIC features (market-only) the policy LEARNS to act on — not hard-coded rules.
import numpy as np, pandas as pd
from config import constants as C
from src.data.cache_builder import build_aligned_indicators
from src.observation import momentum_scores as M
from src.observation import observation_contract as OC


def _ind(n=3000, seed=3, drift=0.005):
    idx = pd.date_range("2026-01-01", periods=n, freq="1min")
    cl = 100 + np.cumsum(np.random.default_rng(seed).standard_normal(n) * 0.05 + drift)
    df = pd.DataFrame({"open": cl, "high": cl + .05, "low": cl - .05, "close": cl, "volume": 1.}, index=idx)
    return build_aligned_indicators(df), cl.astype(np.float64)


def test_shape_names_and_bounds():
    ind, cl = _ind()
    m = M.compute_momentum_scores(ind, cl)
    assert m.shape == (len(cl), C.OBS_BLOCK_MOMENTUM) == (len(cl), 9)
    assert len(M.MOMENTUM_NAMES) == 9 and m.dtype == np.float32
    assert np.all(np.isfinite(m))                                  # never NaN/inf (warmup -> 0, not poison)
    names = M.MOMENTUM_NAMES
    # [0,1] scores
    for nm in ("mom_tradeability", "mom_strength", "mom_exhaustion", "mom_structure", "mom_persistence", "mom_decay"):
        col = m[:, names.index(nm)]
        assert col.min() >= -1e-6 and col.max() <= 1.0 + 1e-6, nm
    # [-1,1] scores
    for nm in ("mom_bias", "mom_alignment", "mom_location"):
        col = m[:, names.index(nm)]
        assert col.min() >= -1.0 - 1e-6 and col.max() <= 1.0 + 1e-6, nm


def test_block_is_in_the_517plus_contract():
    assert OC.BLOCK_SLICES["momentum"] == slice(C.OBS_TOTAL_SIZE - 9, C.OBS_TOTAL_SIZE)
    assert OC.BLOCK_NAMES["momentum"] == list(M.MOMENTUM_NAMES)


def test_leak_free_prefix_invariance():
    # A per-bar leak-free feature must not change at bar t when FUTURE bars are appended.
    ind, cl = _ind(n=2000, seed=7)
    full = M.compute_momentum_scores(ind, cl)
    cut = 1500
    part = M.compute_momentum_scores(ind[:cut], cl[:cut])
    # compare a window safely inside both (after warmup, before the cut) — must be identical
    np.testing.assert_allclose(full[800:cut], part[800:cut], atol=1e-6)
