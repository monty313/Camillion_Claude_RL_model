# v1.10.0 SHIFTED-SMA HUGGING-PRESSURE block (15 leak-free scores, 5m/15m/1h). STATIC market-only features.
import numpy as np, pandas as pd
from config import constants as C
from src.observation import hug_pressure as H
from src.observation import observation_contract as OC
from src.data.aux_features import OHLC_COLUMNS


def _ohlc(n=6000, seed=1, up=2500):
    ts = pd.date_range("2024-03-04 00:00", periods=n, freq="1min").values.astype("datetime64[ns]").astype("int64")
    rng = np.random.default_rng(seed)
    seg = np.r_[np.cumsum(np.abs(rng.normal(0.5, 0.1, up))), np.cumsum(-np.abs(rng.normal(0.4, 0.1, n - up)))]
    close = 52000 + seg
    ohlc = np.zeros((n, 20), np.float32)
    ohlc[:, OHLC_COLUMNS.index("1m__high")] = close + 2
    ohlc[:, OHLC_COLUMNS.index("1m__low")] = close - 2
    return ohlc, ts, up


def test_shape_names_bounds():
    ohlc, ts, _ = _ohlc()
    m = H.compute_hug_pressure(ohlc, ts)
    assert m.shape == (len(ts), C.OBS_BLOCK_HUG_PRESSURE) == (len(ts), 15)
    assert len(H.HUG_PRESSURE_NAMES) == 15 and m.dtype == np.float32 and np.all(np.isfinite(m))
    nm = H.HUG_PRESSURE_NAMES
    for s in ("hug_5m_side", "hug_15m_side", "hug_1h_side", "hug_net_pressure", "hug_dominant_side"):
        c = m[:, nm.index(s)]; assert c.min() >= -1 - 1e-6 and c.max() <= 1 + 1e-6, s
    for s in ("hug_5m_count", "hug_agree_bull", "hug_agree_bear", "hug_strength", "hug_continuation_2plus"):
        c = m[:, nm.index(s)]; assert c.min() >= -1e-6 and c.max() <= 1 + 1e-6, s


def test_direction_tracks_the_trend():
    ohlc, ts, up = _ohlc()
    m = H.compute_hug_pressure(ohlc, ts)
    nm = H.HUG_PRESSURE_NAMES
    # mid up-leg -> bullish dominant; mid down-leg -> bearish dominant
    assert m[up // 2 + 400, nm.index("hug_dominant_side")] > 0
    assert m[up + (len(ts) - up) // 2, nm.index("hug_dominant_side")] < 0
    assert m[up // 2 + 400, nm.index("hug_agree_bull")] >= 2 / 3   # multi-TF agreement on the trend


def test_block_in_contract():
    assert OC.BLOCK_SLICES["hug_pressure"] == slice(C.OBS_TOTAL_SIZE - 15, C.OBS_TOTAL_SIZE)
    assert OC.BLOCK_NAMES["hug_pressure"] == list(H.HUG_PRESSURE_NAMES)


def test_leak_free_prefix_invariance():
    ohlc, ts, _ = _ohlc(n=4000, seed=4, up=1800)
    full = H.compute_hug_pressure(ohlc, ts)
    cut = 3000
    part = H.compute_hug_pressure(ohlc[:cut], ts[:cut])
    np.testing.assert_allclose(full[200:cut], part[200:cut], atol=1e-6)   # past bars unaffected by future


def test_no_ohlc_is_neutral_zero():
    ts = pd.date_range("2024-03-04", periods=500, freq="1min").values.astype("datetime64[ns]").astype("int64")
    m = H.compute_hug_pressure(np.zeros((500, 20), np.float32), ts)   # env built without aux
    assert m.shape == (500, 15) and not np.any(m)
