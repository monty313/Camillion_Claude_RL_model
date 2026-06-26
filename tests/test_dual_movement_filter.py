# Tests for the 2 non-directional movement alphas (STRAT-006 -> dual_movement_filter).
# Key guarantees: (1) 1/0 only (never -1); (2) 1 iff ADX & ATR rising on BOTH TFs;
# (3) ADX is ALPHA-PRIVATE -> the observation stays 479 / v1.5.0 (NO contract change);
# (4) absent alpha-private matrix is safe (alphas read NaN -> 0); (5) streak is automatic.
import numpy as np
import pandas as pd
from config import constants as C
from src.indicators import base
from src.data.cache_builder import build_aligned_indicators, build_aligned_alpha_private
from src.strategies.context import MarketContext
from src.strategies.dual_movement_filter_5m_30m_alpha import DualMovementFilter5m30mAlpha
from src.strategies.dual_movement_filter_30m_4h_alpha import DualMovementFilter30m4hAlpha
from src.strategies.registry import AlphaRegistry
from src.strategies.alpha_pack import register_all
from src.env.trading_env import TradingEnv
from src.observation import observation_contract as OC


def _ctx(vals):
    return MarketContext(close=1.0, indicators=vals, bar_index=10, symbol="EURUSD", minute_of_day=600)


def _moving_pair(low_tf, high_tf):
    """Indicator dict where ADX & ATR are RISING (raw > 5-bars-ago) on both TFs."""
    d = {}
    for tf in (low_tf, high_tf):
        d[f"{tf}__adx14_raw"] = 30.0; d[f"{tf}__adx14_sma1sh5"] = 20.0
        d[f"{tf}__atr14_raw"] = 2.0;  d[f"{tf}__atr14_sma1sh5"] = 1.0
    return d


def test_logic_is_binary_and_requires_both_tfs():
    for AlphaCls, lo, hi in [(DualMovementFilter5m30mAlpha, "5m", "30m"),
                             (DualMovementFilter30m4hAlpha, "30m", "4h")]:
        a = AlphaCls()
        moving = _moving_pair(lo, hi)
        assert a.signal(_ctx(moving)) == 1                       # both rising -> 1
        # ADX falling on the low TF -> 0
        d = dict(moving); d[f"{lo}__adx14_raw"] = 10.0
        assert a.signal(_ctx(d)) == 0
        # ATR not rising (equal) on the high TF -> 0 (strict >)
        d = dict(moving); d[f"{hi}__atr14_raw"] = 1.0
        assert a.signal(_ctx(d)) == 0
        # only one TF moving -> 0 (BOTH required)
        d = dict(moving); d[f"{hi}__adx14_raw"] = 5.0
        assert a.signal(_ctx(d)) == 0
        # missing data -> 0, and NEVER -1
        assert a.signal(_ctx({})) == 0
        assert a.signal(_ctx(moving)) in (0, 1)


def test_alpha_private_columns_do_not_enlarge_the_obs():
    # 4 private cols/TF (adx raw/sh5, atr raw/sh5) -> a SELF-CONTAINED bundle the alphas read.
    assert len(base.PER_TF_ALPHA_PRIVATE_COLUMNS) == 4
    assert len(base.ALL_ALPHA_PRIVATE_COLUMNS) == 4 * C.N_TIMEFRAMES
    # THE REAL GUARANTEE: the private columns are NOT appended to the obs -> obs is unchanged.
    # (atr14_raw intentionally also exists in the obs; identical values, so the merge is consistent.)
    assert C.N_INDICATORS_TOTAL == 220 and len(base.ALL_INDICATOR_COLUMNS) == 220
    assert C.OBS_TOTAL_SIZE == 479 and C.OBSERVATION_CONTRACT_VERSION == "v1.5.0"


def _synth_df(n, seed=0):
    idx = pd.date_range("2026-03-02 00:00", periods=n, freq="1min")
    cl = 100 + np.cumsum(np.random.default_rng(seed).standard_normal(n) * 0.02)
    df = pd.DataFrame({"open": cl, "high": cl + 0.05, "low": cl - 0.05, "close": cl, "volume": 1.0}, index=idx)
    return idx, df


def test_env_feeds_alphas_and_obs_stays_479():
    n = 700
    idx, df = _synth_df(n)
    ind = build_aligned_indicators(df)
    assert ind.shape[1] == C.N_INDICATORS_TOTAL == 220

    # controlled alpha-private matrix: make 5m, 30m AND 4h all "moving" -> slots 16 & 17 fire 1
    K = len(base.ALL_ALPHA_PRIVATE_COLUMNS)
    ap = np.zeros((n, K), dtype=np.float32)
    def setcol(name, val):
        ap[:, base.ALL_ALPHA_PRIVATE_COLUMNS.index(name)] = val
    for tf in ("5m", "30m", "4h"):
        setcol(f"{tf}__adx14_raw", 30.0); setcol(f"{tf}__adx14_sma1sh5", 20.0)
        setcol(f"{tf}__atr14_raw", 2.0);  setcol(f"{tf}__atr14_sma1sh5", 1.0)

    reg = AlphaRegistry(); register_all(reg); assert reg.assigned_count == 18
    env = TradingEnv(ind, df["close"].values.astype("float32"),
                     idx.values.astype("datetime64[ns]").astype("int64"), reg,
                     warmup=300, symbol="EURUSD", alpha_indicators=ap)
    o, _ = env.reset()
    # THE POINT: adding ADX + 2 alphas did NOT change the observation
    assert o.shape == (479,) and C.OBSERVATION_CONTRACT_VERSION == "v1.5.0"
    assert np.all(np.isfinite(o))
    # both movement slots fire 1, and are strictly binary (never -1)
    assert (env.alpha_matrix[:, 16] == 1.0).all()
    assert (env.alpha_matrix[:, 17] == 1.0).all()
    assert set(np.unique(env.alpha_matrix[:, 16:18])).issubset({0.0, 1.0})
    # streak block (already in the obs) increases while the 1-signal persists -- no new code needed
    assert env.streak_matrix[306, 16] > env.streak_matrix[301, 16]


def test_absent_alpha_private_is_safe():
    n = 600
    idx, df = _synth_df(n, seed=1)
    ind = build_aligned_indicators(df)
    reg = AlphaRegistry(); register_all(reg)
    env = TradingEnv(ind, df["close"].values.astype("float32"),
                     idx.values.astype("datetime64[ns]").astype("int64"), reg, warmup=300)
    o, _ = env.reset()
    assert o.shape == (479,)
    # without ADX data the movement alphas read NaN -> always 0 (no crash, slots still occupied)
    assert (env.alpha_matrix[:, 16:18] == 0.0).all()


def test_cache_builder_alpha_private_aligns():
    n = 3000
    _, df = _synth_df(n, seed=2)
    ap = build_aligned_alpha_private(df)
    assert ap.shape == (n, len(base.ALL_ALPHA_PRIVATE_COLUMNS))
    # the 5m ADX is well-defined with this many bars -> finite in the tail (alignment works).
    j = base.ALL_ALPHA_PRIVATE_COLUMNS.index("5m__adx14_raw")
    assert np.all(np.isfinite(ap[1000:, j]))
    # higher TFs (4h/1d) may stay NaN with little data -- that's expected; alphas treat NaN as 0.
