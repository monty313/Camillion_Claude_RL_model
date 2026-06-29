# ADX-DI alignment alphas (slots 16/17) + the v1.7.0 OHLC observation block + the DI side-channel.
# Locks: the -DI vs +DI agreement logic, slot wiring, that DI feeds the alphas via aux (not the obs),
# that the raw OHLC block is present + leak-free, and that the obs shape is 513.
import numpy as np
import pandas as pd
from config import constants as C
from src.observation import observation_contract as OC
from src.data import aux_features as AX
from src.data.cache_builder import build_aligned_indicators, build_aligned_aux
from src.strategies.context import MarketContext
from src.strategies.registry import AlphaRegistry
from src.strategies.alpha_pack import register_all
from src.strategies.adx_di_align_5m_30m_alpha import AdxDiAlign5m30mAlpha
from src.strategies.adx_di_align_30m_4h_alpha import AdxDiAlign30m4hAlpha
from src.env.trading_env import TradingEnv


def _ctx_for(alpha, plus, minus):
    """Build a MarketContext where EVERY (tf, period) DI pair of `alpha` is set to (plus, minus)."""
    d = {}
    for tf in alpha.TIMEFRAMES:
        for p in alpha.PERIODS:
            d[f"{tf}__plus_di{p}"] = plus
            d[f"{tf}__minus_di{p}"] = minus
    return MarketContext(indicators=d)


def test_adx_di_logic_both_alphas():
    for alpha in (AdxDiAlign5m30mAlpha(), AdxDiAlign30m4hAlpha()):
        assert alpha.compute_signal(_ctx_for(alpha, 10.0, 30.0)) == -1   # -DI above +DI everywhere -> SELL
        assert alpha.compute_signal(_ctx_for(alpha, 30.0, 10.0)) == 1    # -DI below +DI everywhere -> BUY
        assert alpha.compute_signal(_ctx_for(alpha, 20.0, 20.0)) == 0    # equal -> inactive (strict)
        assert alpha.compute_signal(MarketContext(indicators={})) == 0   # missing/warmup -> inactive


def test_adx_di_one_disagreeing_pair_blocks_signal():
    a = AdxDiAlign5m30mAlpha()
    d = {f"{tf}__plus_di{p}": 10.0 for tf in a.TIMEFRAMES for p in a.PERIODS}   # bearish base
    d.update({f"{tf}__minus_di{p}": 30.0 for tf in a.TIMEFRAMES for p in a.PERIODS})
    assert a.compute_signal(MarketContext(indicators=d)) == -1
    d["30m__minus_di45"] = 5.0    # flip ONE pair to bullish -> no longer unanimous
    assert a.compute_signal(MarketContext(indicators=d)) == 0


def _df(n=6000, seed=1, drift=0.003):
    idx = pd.date_range("2026-01-01", periods=n, freq="1min")
    close = 100 + np.cumsum(np.random.default_rng(seed).standard_normal(n) * 0.05 + drift)
    return pd.DataFrame({"open": close, "high": close + 0.05, "low": close - 0.05,
                         "close": close, "volume": 1.0}, index=idx)


def _env(df, aux=True, warmup=300):
    ind = build_aligned_indicators(df)
    a = build_aligned_aux(df) if aux else None
    reg = AlphaRegistry(); register_all(reg)
    t = df.index.values.astype("datetime64[ns]").astype("int64")
    return TradingEnv(ind, df["close"].values.astype("float32"), t, reg, warmup=warmup, aux=a), reg


def test_slots_16_17_wired_and_obs_513():
    reg = AlphaRegistry(); register_all(reg)
    names = [s.name if s else None for s in reg._slots]
    assert names[16] == "adx_di_align_5m_30m" and names[17] == "adx_di_align_30m_4h"
    env, _ = _env(_df())
    o, _ = env.reset()
    assert o.shape == (513,) and np.all(np.isfinite(o))
    assert o[OC.BLOCK_SLICES["alpha_mask"]][16] == 1.0 and o[OC.BLOCK_SLICES["alpha_mask"]][17] == 1.0


def test_di_feeds_alphas_only_with_aux():
    # WITHOUT aux the DI columns are absent -> the two alphas can never fire (always 0).
    env0, _ = _env(_df(), aux=False)
    assert (env0.alpha_matrix[:, 16] == 0).all() and (env0.alpha_matrix[:, 17] == 0).all()
    # WITH aux on a trending series, slot 16 (5m/30m, shorter warmup) DOES fire.
    env1, _ = _env(_df(drift=0.02))
    assert (env1.alpha_matrix[:, 16] != 0).any()


def test_ohlc_block_present_and_matches_aux():
    df = _df()
    env, _ = _env(df)
    i = env.warmup + 50
    env.reset(); env.ptr = i
    obs = env._obs()
    block = obs[OC.BLOCK_SLICES["ohlc"]]
    assert block.shape[0] == C.OBS_BLOCK_OHLC == 20
    # the obs OHLC block must equal the env's aligned OHLC matrix row (the first 20 aux columns),
    # AFTER the builder's nan_to_num (a not-yet-closed higher-TF bar is NaN -> 0 in the obs).
    assert np.allclose(block, np.nan_to_num(env.ohlc_matrix[i].astype(np.float32)))
    # 1m close field == the current close (1m bar's own close; OHLC_COLUMNS = tf-major, fields O,H,L,C)
    c_idx = AX.OHLC_COLUMNS.index("1m__close")
    assert abs(float(block[c_idx]) - float(df["close"].values[i])) < 1e-3


def test_obs_without_aux_has_zero_ohlc_but_valid_shape():
    env, _ = _env(_df(), aux=False)
    o, _ = env.reset()
    assert o.shape == (513,) and np.all(np.isfinite(o))
    assert np.all(o[OC.BLOCK_SLICES["ohlc"]] == 0.0)   # no aux -> OHLC block is zeros (still valid)


def test_aux_is_leak_free():
    # Appending a FUTURE bar with a spiked high must NOT change aux at earlier bars (last-closed-bar rule).
    df = _df(n=3000)
    aux_a = build_aligned_aux(df)
    df2 = df.copy()
    df2.iloc[2500, df2.columns.get_loc("high")] += 50.0   # spike a FUTURE bar's high
    df2.iloc[2500, df2.columns.get_loc("close")] += 50.0
    aux_b = build_aligned_aux(df2)
    assert np.allclose(aux_a[:2400], aux_b[:2400], equal_nan=True)   # earlier rows unchanged -> no leak
