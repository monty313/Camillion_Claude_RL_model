# v1.12.0 Stage 2: env TP/SL BRACKET-order model + risk-clamped lot (CPU). Fed-values unit test (isolated --
# no policy/PPO): feed tp01/sl01/lot01 into PortfolioEnv.step(), verify TP/SL exits at the LOCKED level price,
# the 1%-equity lot clamp, and the open/close logging. Brackets default-OFF; existing behavior untouched.
import numpy as np, pandas as pd
from config import constants as C
from src.env.portfolio_env import PortfolioEnv, build_portfolio_subs
from src.strategies.registry import AlphaRegistry
from src.strategies.alpha_pack import register_all
from src.data.aux_features import OHLC_COLUMNS

_H1, _L1, _C1 = OHLC_COLUMNS.index("1m__high"), OHLC_COLUMNS.index("1m__low"), OHLC_COLUMNS.index("1m__close")


def _reg():
    r = AlphaRegistry(); register_all(r); return r


def _env(high, low, close, *, bracket=True, n=40):
    tns = pd.date_range("2024-03-04 13:00", periods=n, freq="1min").values.astype("datetime64[ns]").astype(np.int64)
    ind = np.zeros((n, 220), np.float32)
    aux = np.zeros((n, 32), np.float32)
    aux[:, _H1] = high; aux[:, _L1] = low; aux[:, _C1] = close
    sd = {"US30": (ind, close.astype(np.float32), tns, aux)}
    subs = build_portfolio_subs(sd, _reg, warmup=5, progress=False)
    return PortfolioEnv(subs=subs, warmup=5, bracket_enabled=bracket)


def _flat(n=40, px=100.0):
    return np.full(n, px), np.full(n, px), np.full(n, px)   # high, low, close


def _buy_then_hold(env, tp01, sl01, lot01, until_bar):
    """HOLD up to bar (until_bar-1), BUY at until_bar, then HOLD — returns when flat again or 12 steps."""
    while env.t < until_bar:
        env.step(C.ACTION_HOLD)
    env.step(C.ACTION_BUY, tp01=tp01, sl01=sl01, lot01=lot01)   # opens the bracket at this bar
    for _ in range(12):
        if env.position["US30"] == 0:
            break
        env.step(C.ACTION_HOLD)


def test_tp_hit_closes_at_locked_tp_level():
    high, low, close = _flat()
    high[10] = 200.0                                   # a spike that crosses TP at bar 10
    env = _env(high, low, close); env.reset()
    _buy_then_hold(env, tp01=0.5, sl01=0.5, lot01=0.2, until_bar=8)
    log = env._bracket_log
    opens = [e for e in log if e["event"] == "open"]; closes = [e for e in log if e["event"] == "close"]
    assert len(opens) == 1 and len(closes) == 1
    o, c = opens[0], closes[0]
    assert c["hit"] == "TP"
    tp_price = 100.0 * (1.0 + o["tp_pct"])             # long TP locked at entry
    assert abs(c["level"] - tp_price) < 1e-6 and c["pnl"] > 0.0
    assert env.position["US30"] == 0
    # the open log carries the required fields
    for k in ("tp_pct", "sl_pct", "rr", "lot_raw", "lot_used", "clamped", "session_active", "alignment_score_at_entry"):
        assert k in o
    assert abs(o["rr"] - o["tp_pct"] / o["sl_pct"]) < 1e-9


def test_sl_hit_closes_at_locked_sl_level():
    high, low, close = _flat()
    low[10] = 50.0                                     # a drop that crosses SL at bar 10
    env = _env(high, low, close); env.reset()
    _buy_then_hold(env, tp01=0.9, sl01=0.5, lot01=0.2, until_bar=8)
    closes = [e for e in env._bracket_log if e["event"] == "close"]
    assert len(closes) == 1 and closes[0]["hit"] == "SL"
    sl_price = 100.0 * (1.0 - [e for e in env._bracket_log if e["event"] == "open"][0]["sl_pct"])
    assert abs(closes[0]["level"] - sl_price) < 1e-6 and closes[0]["pnl"] < 0.0


def test_lot_clamp_caps_risk_at_1pct_equity():
    high, low, close = _flat()
    env = _env(high, low, close); env.reset()
    while env.t < 8:
        env.step(C.ACTION_HOLD)
    base = float(env.subs["US30"].position_size); equity = float(env.acc.equity)
    env.step(C.ACTION_BUY, tp01=0.9, sl01=1.0, lot01=1.0)        # max SL distance + max lot -> should clamp
    o = [e for e in env._bracket_log if e["event"] == "open"][0]
    sl_pct = C.SL_MIN_PCT + 1.0 * (C.SL_MAX_PCT - C.SL_MIN_PCT)
    lot_raw_expected = C.LOT_MAX_MULT * base
    risk_cap = C.MAX_TRADE_RISK_PCT / 100.0 * equity
    # INVARIANT: post-clamp open risk never exceeds 1% of equity
    assert o["lot_used"] * sl_pct * 100.0 <= risk_cap + 1e-6
    assert o["lot_used"] <= o["lot_raw"] + 1e-9 and abs(o["lot_raw"] - lot_raw_expected) < 1e-6
    # clamp fires iff the raw lot's risk would have exceeded the cap (deterministic given base + equity)
    assert o["clamped"] == bool(lot_raw_expected * sl_pct * 100.0 > risk_cap + 1e-9)


def test_default_off_ignores_bracket_args():
    high, low, close = _flat()
    high[10] = 200.0
    env = _env(high, low, close, bracket=False); env.reset()
    _buy_then_hold(env, tp01=0.5, sl01=0.5, lot01=1.0, until_bar=8)
    assert env._bracket_log == []                                # OFF -> no brackets, args ignored
    assert env._tp_price["US30"] == 0.0
