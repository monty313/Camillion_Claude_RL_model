# =====================================================================
# WHEN 2026-06-28 | WHO Claude for Monty
# WHY  OPTIONAL on-device indicators in pure jnp (blueprint Rule 1 / "co-location"
#      ideal: generate features on the TPU with operator-fusion). These are NOT on the
#      critical path — the trainer indexes the precomputed static tensor for exact
#      parity + speed. They exist so a future fully-on-device pipeline can compute
#      RSI/ATR/CCI/SMA/Bollinger in the XLA graph, and they are parity-tested against
#      the CPU pandas implementations (src/indicators/*).
# WHERE jax_tpu/jax_indicators.py
# HOW   Wilder EWM via lax.scan (alpha=1/period, adjust=False, min_periods=period);
#       rolling mean/std via cumsum with NaN propagation; CCI mean-abs-dev via a
#       vmapped sliding window. Matches the pandas warmup/NaN-prefix exactly.
# DEPENDS_ON: jax
# USED_BY: jax_tpu/tests/test_jax_parity.py (parity vs src/indicators/*); future on-device env
# CHANGE_NOTES(IRAC): I: the full on-device ideal needs indicators in jnp. R: blueprint
#   Rule 1. A: port each CPU indicator to jnp with identical smoothing + NaN warmup,
#   parity-tested. C: a verified on-device indicator path for when we want to fuse it in.
# =====================================================================
"""OPTIONAL on-device indicators in jnp (RSI/ATR/CCI/SMA/Bollinger) — parity-tested vs src/indicators/*."""
from __future__ import annotations
from functools import partial
import jax
import jax.numpy as jnp

_NAN = jnp.nan


def _ewm_wilder(x, period):
    """pandas Series.ewm(alpha=1/period, adjust=False, min_periods=period).mean() with leading NaNs.
    Seeds y at the first valid x (adjust=False), carries over interior NaNs, masks until min_periods."""
    alpha = 1.0 / period
    valid = jnp.isfinite(x)
    x0 = jnp.where(valid, x, 0.0)

    def body(carry, inp):
        y_prev, started, count = carry
        xv, v = inp
        seed = (1.0 - started) * v                       # first valid -> seed
        cont = started * v                               # subsequent valid -> recurse
        y = seed * xv + cont * ((1.0 - alpha) * y_prev + alpha * xv) + (1.0 - v) * y_prev
        started = jnp.maximum(started, v)
        count = count + v
        return (y, started, count), (y, count)

    (_, _, _), (ys, counts) = jax.lax.scan(
        body, (jnp.asarray(0.0, x0.dtype), jnp.asarray(0.0, x0.dtype), jnp.asarray(0.0, x0.dtype)),
        (x0, valid.astype(x0.dtype)))
    return jnp.where(counts >= period, ys, _NAN)


def _rolling_mean(x, period):
    """rolling(period, min_periods=period).mean() with NaN propagation (any NaN in window -> NaN)."""
    valid = jnp.isfinite(x).astype(x.dtype)
    xz = jnp.where(jnp.isfinite(x), x, 0.0)
    cs = jnp.concatenate([jnp.zeros(1, xz.dtype), jnp.cumsum(xz)])
    cv = jnp.concatenate([jnp.zeros(1, xz.dtype), jnp.cumsum(valid)])
    idx = jnp.arange(x.shape[0])
    lo = idx - period + 1
    wsum = cs[idx + 1] - jnp.where(lo > 0, cs[jnp.maximum(lo, 0)], 0.0)
    wcnt = cv[idx + 1] - jnp.where(lo > 0, cv[jnp.maximum(lo, 0)], 0.0)
    full = (idx >= period - 1) & (wcnt >= period)
    return jnp.where(full, wsum / period, _NAN)


def _rolling_std_pop(x, period):
    """rolling(period).std(ddof=0) (population std) with NaN propagation."""
    m = _rolling_mean(x, period)
    m2 = _rolling_mean(x * x, period)
    var = jnp.maximum(m2 - m * m, 0.0)
    return jnp.sqrt(var)


def _shift_fwd(x, shift):
    """Shift forward by `shift` bars (out[t]=x[t-shift]); leading -> NaN. Matches pandas .shift."""
    if shift <= 0:
        return x
    pad = jnp.full((shift,), _NAN, x.dtype)
    return jnp.concatenate([pad, x[:-shift]])


def sma(values, period, shift=0):
    period = max(1, int(period))
    return _shift_fwd(_rolling_mean(jnp.asarray(values), period), int(shift))


def rsi_raw(close, period):
    close = jnp.asarray(close)
    delta = jnp.concatenate([jnp.full((1,), _NAN, close.dtype), jnp.diff(close)])
    gain = jnp.clip(delta, 0.0, None)
    loss = jnp.clip(-delta, 0.0, None)
    avg_gain = _ewm_wilder(gain, period)
    avg_loss = _ewm_wilder(loss, period)
    rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = jnp.where(avg_loss == 0.0, 100.0, rsi)   # all-gains window
    rsi = jnp.where(avg_gain == 0.0, 0.0, rsi)     # all-losses window
    return rsi


def rsi_post(close, period, post_sma=2, post_shift=2):
    return sma(rsi_raw(close, period), post_sma, shift=post_shift)


def atr_raw(high, low, close, period=14):
    high, low, close = jnp.asarray(high), jnp.asarray(low), jnp.asarray(close)
    prev_close = jnp.concatenate([jnp.full((1,), _NAN, close.dtype), close[:-1]])
    tr = jnp.maximum(high - low, jnp.maximum(jnp.abs(high - prev_close), jnp.abs(low - prev_close)))
    tr = jnp.where(jnp.isfinite(tr), tr, high - low)   # bar 0: prev_close NaN -> max skips it (== h-l)
    return _ewm_wilder(tr, period)


def atr_post(high, low, close, period=14, post_sma=2, post_shift=4):
    return sma(atr_raw(high, low, close, period), post_sma, shift=post_shift)


def cci_raw(high, low, close, period):
    high, low, close = jnp.asarray(high), jnp.asarray(low), jnp.asarray(close)
    tp = (high + low + close) / 3.0
    ma = _rolling_mean(tp, period)

    def mad(i):
        w = jax.lax.dynamic_slice(tp, (i,), (period,))
        return jnp.mean(jnp.abs(w - jnp.mean(w)))

    idx = jnp.arange(tp.shape[0] - period + 1)
    md_valid = jax.vmap(mad)(idx)
    md = jnp.concatenate([jnp.full((period - 1,), _NAN, tp.dtype), md_valid])
    return (tp - ma) / (0.015 * md)


def cci_post(high, low, close, period, post_sma=2, post_shift=4):
    return sma(cci_raw(high, low, close, period), post_sma, shift=post_shift)


def bollinger(values, period, dev):
    v = jnp.asarray(values)
    m = _rolling_mean(v, period)
    sd = _rolling_std_pop(v, period)
    return m + dev * sd, m, m - dev * sd
