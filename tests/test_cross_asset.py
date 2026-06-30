# v1.7.0 CROSS-ASSET perception block: asset-class one-hot (covers the FULL FTMO broker via a
# name classifier) + ATR-normalized movement (scale-free -> comparable across pairs/indices/
# metals/energies/crypto) + sessions (Asian, London-NY overlap). Lets ONE policy generalize.
import numpy as np
import pandas as pd
from config import constants as C
from config import asset_specs as A
from src.env.trading_env import TradingEnv
from src.strategies.registry import AlphaRegistry
from src.observation import observation_contract as OC


def test_block_in_contract_v1_4_0():
    assert C.OBSERVATION_CONTRACT_VERSION == "v1.12.0" and C.OBS_TOTAL_SIZE == 557
    assert C.OBS_BLOCK_CROSS_ASSET == len(C.ASSET_CLASSES) + 5
    sl = OC.BLOCK_SLICES["cross_asset"]
    assert sl.stop - sl.start == C.OBS_BLOCK_CROSS_ASSET
    assert OC.FEATURE_NAMES[sl.start] == "class_pair"
    assert OC.FEATURE_NAMES[sl.stop - 1] == "session_london_ny_overlap"


def test_classifier_covers_the_full_broker():
    cls = A.asset_class
    assert cls("EURUSD") == "pair" and cls("GBPJPY") == "pair"
    assert cls("US30") == "index" and cls("NAS100") == "index" and cls("GER40") == "index"
    assert cls("XAUUSD") == "metal" and cls("XAGUSD") == "metal"
    assert cls("USOIL") == "energy" and cls("NGAS") == "energy" and cls("UKOIL") == "energy"
    assert cls("BTCUSD") == "crypto" and cls("ETHUSD") == "crypto" and cls("SOLUSD") == "crypto"
    assert cls("ZZ") is None                         # unknown -> safe all-zeros one-hot


def _env(symbol, price=1.10, dr=0.0003, n=400):
    idx = pd.date_range("2026-03-02 09:00", periods=n, freq="1min")
    close = (price + np.cumsum(np.random.default_rng(0).standard_normal(n)) * dr).astype(np.float32)
    ind = np.zeros((n, C.N_INDICATORS_TOTAL), np.float32)
    return TradingEnv(ind, close, idx.values.astype("datetime64[ns]").astype(np.int64),
                      AlphaRegistry(), warmup=260, symbol=symbol, position_size=1.0)


def test_class_one_hot_in_obs_per_symbol():
    sl = OC.BLOCK_SLICES["cross_asset"]
    k = len(C.ASSET_CLASSES)
    for sym, klass in [("EURUSD", "pair"), ("US30", "index"), ("XAUUSD", "metal")]:
        obs, _ = _env(sym).reset()
        block = obs[sl]
        assert block[C.ASSET_CLASSES.index(klass)] == 1.0
        assert block[:k].sum() == 1.0                # exactly one class set
        assert obs.shape == (557,) and np.all(np.isfinite(obs))


def test_atr_normalized_features_present_and_bounded():
    obs, _ = _env("EURUSD").reset()
    k = len(C.ASSET_CLASSES)
    block = obs[OC.BLOCK_SLICES["cross_asset"]]
    move_in_atr, atr_pct, atr_regime = block[k], block[k + 1], block[k + 2]
    assert -1.0 <= move_in_atr <= 1.0
    assert 0.0 <= atr_pct <= 1.0
    assert 0.0 <= atr_regime <= 1.0
