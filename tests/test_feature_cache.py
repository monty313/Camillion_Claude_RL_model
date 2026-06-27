# Feature cache (2026-06-27): save the expensive per-symbol precompute to disk (Drive on Colab) and
# reload it ONLY when the fingerprint matches exactly -- so re-runs are fast with NO risk of stale
# features. These tests lock the round-trip, the no-mismatch guard, and the keys staying in sync.
import os
import tempfile
import numpy as np
import pandas as pd
from config import constants as C
from src.env.trading_env import TradingEnv
from src.data import feature_cache as FC
from src.strategies.registry import AlphaRegistry
from src.strategies.alpha_pack import register_all


def _reg():
    r = AlphaRegistry(); register_all(r); return r


def _data(n=1500, seed=0, base=100.0):
    close = (base + np.cumsum(np.random.default_rng(seed).standard_normal(n) * 0.05)).astype(np.float32)
    ind = np.zeros((n, C.N_INDICATORS_TOTAL), np.float32)
    tns = pd.date_range("2026-03-02", periods=n, freq="1min").values.astype("datetime64[ns]").astype(np.int64)
    return ind, close, tns


def test_cache_keys_stay_in_sync_with_env():
    # The env's exported arrays must match what the cache expects (a drift here = silent corruption).
    assert list(TradingEnv._PRECOMPUTED_ATTRS) == list(FC.PRECOMPUTED_ARRAY_KEYS)


def test_save_then_load_gives_identical_features_and_obs():
    ind, close, tns = _data()
    base = tempfile.mkdtemp()
    reg = _reg()
    env = TradingEnv(ind, close, tns, reg, symbol="EURUSD")
    FC.save(base, "EURUSD", ind, close, tns, reg, env)
    loaded = FC.load(base, "EURUSD", ind, close, tns, _reg())
    assert loaded is not None, "exact same inputs must be a cache HIT"
    env2 = TradingEnv(ind, close, tns, _reg(), symbol="EURUSD", precomputed=loaded)
    for a in TradingEnv._PRECOMPUTED_ATTRS:
        assert np.array_equal(np.asarray(getattr(env, a)), np.asarray(getattr(env2, a))), f"array {a} differs"
    env.reset(); env2.reset()
    for _ in range(5):                                   # the observation must be byte-identical to a rebuild
        assert np.array_equal(env._obs(), env2._obs())
        env.step(C.ACTION_HOLD); env2.step(C.ACTION_HOLD)


def test_stale_or_mismatched_inputs_are_rejected():
    ind, close, tns = _data(seed=0)
    base = tempfile.mkdtemp()
    reg = _reg()
    FC.save(base, "EURUSD", ind, close, tns, reg, TradingEnv(ind, close, tns, reg, symbol="EURUSD"))
    # same shape + same dates but DIFFERENT content -> must MISS (never load stale)
    ind2, close2, tns2 = _data(seed=99)
    assert FC.load(base, "EURUSD", ind2, close2, tns2, _reg()) is None
    # different symbol -> miss
    assert FC.load(base, "GBPUSD", ind, close, tns, _reg()) is None


def test_fingerprint_includes_code_and_contract():
    # The fingerprint must fold in the contract version + the feature CODE hashes (the gap the old
    # env_fingerprint missed). A names-only key would not change on a threshold edit.
    ind, close, tns = _data()
    _, man = FC.fingerprint("EURUSD", ind, close, tns, _reg())
    assert man["contract_version"] == C.OBSERVATION_CONTRACT_VERSION
    assert man["max_strategies"] == C.MAX_STRATEGIES
    assert "strategy_code" in man["code_hashes"] and "precompute_code" in man["code_hashes"]
    assert man["alpha_roster"] and isinstance(man["alpha_roster"][0], list)  # slot-ORDERED [slot, name]
    assert man["data"]["data_hash"] and man["key"]


def test_build_portfolio_subs_round_trips_via_cache():
    from src.env.portfolio_env import build_portfolio_subs
    base = tempfile.mkdtemp()
    sd = {}
    for i, (s, b) in enumerate([("EURUSD", 1.10), ("US30", 38000.0)]):
        sd[s] = _data(n=1500, seed=i, base=b)
    subs1 = build_portfolio_subs(sd, _reg, progress=False, feature_cache_dir=base)   # builds + SAVES
    subs2 = build_portfolio_subs(sd, _reg, progress=False, feature_cache_dir=base)   # LOADS from cache
    for s in sd:
        assert np.array_equal(subs1[s].alpha_matrix, subs2[s].alpha_matrix)
        assert np.array_equal(subs1[s].cross_asset_matrix, subs2[s].cross_asset_matrix)
        assert np.array_equal(subs1[s].streak_matrix, subs2[s].streak_matrix)
    # a manifest.json exists and says what it's for
    sub_dirs = [os.path.join(base, d) for d in os.listdir(base)]
    assert any(os.path.isfile(os.path.join(d, "manifest.json")) for d in sub_dirs)
