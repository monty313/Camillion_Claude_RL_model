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


def test_directional_consensus_excludes_gates():
    # THE PURPOSE FIX: a non-directional gate's 1 must NOT be counted as a buy.
    from src.signals.signal_summary import summarize, net_balance
    av = np.array([1, -1, 1, 0], dtype=np.float32)     # buy, sell, GATE(1), empty
    occ = np.array([1, 1, 1, 0], dtype=np.float32)
    dirm = np.array([1, 1, 0, 0], dtype=np.float32)    # slot 2 is a non-directional gate
    # legacy (no mask): the gate's 1 is miscounted as a buy -> polluted
    s_all = summarize(av, occ)
    assert abs(s_all[0] - 2 / 3) < 1e-6 and abs(s_all[3] - 1 / 3) < 1e-6
    assert abs(net_balance(av) - 1 / 3) < 1e-6
    # with the directional mask: gate excluded -> clean 50/50, net 0
    s_dir = summarize(av, occ, dirm)
    assert s_dir[0] == 0.5 and s_dir[1] == 0.5 and s_dir[3] == 0.0
    assert net_balance(av, dirm) == 0.0


def test_registry_marks_movement_alphas_non_directional():
    reg = AlphaRegistry(); register_all(reg)
    occ = reg.occupancy_mask(); dm = reg.directional_mask()
    assert occ[16] == 1.0 and occ[17] == 1.0           # movement alphas ARE assigned (visible)
    assert dm[16] == 0.0 and dm[17] == 0.0             # but NOT directional (excluded from consensus)
    assert dm[:16].sum() == 16.0 and dm.sum() == 16.0  # exactly the 16 directional alphas vote


def test_env_gate_fires_but_does_not_move_the_consensus():
    # register ONLY the 5m/30m movement gate, force it "moving": it fires 1 in its slot,
    # yet the directional consensus (net_signal + alpha_summary) stays empty.
    from src.strategies.register_dual_movement_filter_5m_30m_alpha import register as reg_gate
    n = 700
    idx, df = _synth_df(n, seed=3)
    ind = build_aligned_indicators(df)
    K = len(base.ALL_ALPHA_PRIVATE_COLUMNS)
    ap = np.zeros((n, K), dtype=np.float32)
    def setcol(name, val):
        ap[:, base.ALL_ALPHA_PRIVATE_COLUMNS.index(name)] = val
    for tf in ("5m", "30m"):
        setcol(f"{tf}__adx14_raw", 30.0); setcol(f"{tf}__adx14_sma1sh5", 20.0)
        setcol(f"{tf}__atr14_raw", 2.0);  setcol(f"{tf}__atr14_sma1sh5", 1.0)
    reg = AlphaRegistry(); reg_gate(reg)               # ONLY the gate -> slot 0
    env = TradingEnv(ind, df["close"].values.astype("float32"),
                     idx.values.astype("datetime64[ns]").astype("int64"), reg,
                     warmup=300, symbol="EURUSD", alpha_indicators=ap)
    o, _ = env.reset()
    assert (env.alpha_matrix[:, 0] == 1.0).all()       # the gate fires 1 in its own slot
    assert np.allclose(env.net_signal, 0.0)            # ...but contributes NOTHING to net_signal
    sl = OC.BLOCK_SLICES["alpha_summary"]
    assert np.allclose(o[sl], 0.0)                     # buy%/sell%/active%/net% all 0 (no buy vote)


def test_cache_builder_alpha_private_aligns():
    n = 3000
    _, df = _synth_df(n, seed=2)
    ap = build_aligned_alpha_private(df)
    assert ap.shape == (n, len(base.ALL_ALPHA_PRIVATE_COLUMNS))
    # the 5m ADX is well-defined with this many bars -> finite in the tail (alignment works).
    j = base.ALL_ALPHA_PRIVATE_COLUMNS.index("5m__adx14_raw")
    assert np.all(np.isfinite(ap[1000:, j]))
    # higher TFs (4h/1d) may stay NaN with little data -- that's expected; alphas treat NaN as 0.
