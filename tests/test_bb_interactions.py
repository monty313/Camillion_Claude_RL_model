# v1.11.0 DUAL-BB INTERACTION block (12 leak-free scores): BB squeeze/expansion + cross-TF momentum cascade +
# BB-extreme mean-reversion flags. STATIC market-only features; ONLY logic not already in momentum/hug/trade_risk.
import numpy as np, pandas as pd
from config import constants as C
from src.observation import bb_interactions as BBI
from src.observation import observation_contract as OC
from src.indicators.base import ALL_INDICATOR_COLUMNS
from src.data.cache_builder import build_aligned_indicators


def _real(n=8000, seed=2):
    idx = pd.date_range("2026-01-01", periods=n, freq="1min")
    base = np.cumsum(np.r_[np.full(n // 2, 0.05), np.full(n - n // 2, -0.04)])
    cl = 100 + base + np.random.default_rng(seed).standard_normal(n) * 0.05
    df = pd.DataFrame({"open": cl, "high": cl + .05, "low": cl - .05, "close": cl, "volume": 1.}, index=idx)
    return build_aligned_indicators(df), cl.astype(np.float64)


def test_shape_names_bounds():
    ind, cl = _real()
    m = BBI.compute_bb_interactions(ind, cl)
    assert m.shape == (len(cl), C.OBS_BLOCK_BB_INTERACTIONS) == (len(cl), 12)
    assert len(BBI.BB_INTERACTION_NAMES) == 12 and m.dtype == np.float32 and np.all(np.isfinite(m))
    # every feature is bounded to ~[-1, 1] (tanh / fractions / flags)
    assert m.min() >= -1.0 - 1e-6 and m.max() <= 1.0 + 1e-6


def test_cascade_and_expansion_vary():
    ind, cl = _real()
    m = BBI.compute_bb_interactions(ind, cl)
    nm = BBI.BB_INTERACTION_NAMES
    for s in ("bbw_expansion_5m", "bb_cascade_5m_30m"):     # these must carry real (non-constant) signal
        assert np.ptp(m[300:, nm.index(s)]) > 0.05, s


def test_block_in_contract():
    sl = OC.BLOCK_SLICES["bb_interactions"]               # appended at v1.11.0 (scalp_momentum appended after)
    assert sl == slice(541, 553) and sl.stop - sl.start == 12
    assert OC.BLOCK_NAMES["bb_interactions"] == list(BBI.BB_INTERACTION_NAMES)


def test_leak_free_prefix_invariance():
    ind, cl = _real(n=6000, seed=4)
    full = BBI.compute_bb_interactions(ind, cl)
    cut = 4500
    part = BBI.compute_bb_interactions(ind[:cut], cl[:cut])
    np.testing.assert_allclose(full[300:cut], part[300:cut], atol=1e-6)


def test_mean_reversion_flag_fires_on_a_forced_setup():
    # Force the EXACT setup: price pinned at the 30m BB200 UPPER edge (slow_dist>1.5) while the 5m fast band
    # distance steps from extreme (>2) back inside (<1.5) -> bb_mr_short_30m must fire. (Real 30m/4h BB200 needs
    # long warmup, so synthetic price rarely reaches it; this verifies the LOGIC directly.)
    T = 400
    ind = np.zeros((T, len(ALL_INDICATOR_COLUMNS)), np.float64)
    cl = np.full(T, 111.0)
    def col(name): return ALL_INDICATOR_COLUMNS.index(name)
    # 30m BB200: middle=100, +/-2sigma band [90,110] -> slow_std=5 -> slow_dist=(111-100)/5=2.2 (> 1.5 edge)
    ind[:, col("30m__bb200_dev2.0_middle")] = 100.0
    ind[:, col("30m__bb200_dev2.0_upper")] = 110.0
    ind[:, col("30m__bb200_dev2.0_lower")] = 90.0
    # 5m BB20: fast_std=1 (upper=middle+2). middle steps 108 -> 110 at T//2 so fast_dist 3 -> 1 (reverting down)
    mid5 = np.where(np.arange(T) < T // 2, 108.0, 110.0)
    ind[:, col("5m__bb20_dev2.0_middle")] = mid5
    ind[:, col("5m__bb20_dev2.0_upper")] = mid5 + 2.0
    ind[:, col("5m__bb20_dev2.0_lower")] = mid5 - 2.0
    m = BBI.compute_bb_interactions(ind, cl)
    mr_short = m[:, BBI.BB_INTERACTION_NAMES.index("bb_mr_short_30m")]
    assert mr_short[T // 2 + 2:T // 2 + 20].max() == 1.0, "fade-the-extreme flag must fire just after the reversal"
    assert m[:, BBI.BB_INTERACTION_NAMES.index("bb_mr_long_30m")].max() == 0.0   # opposite side stays off
