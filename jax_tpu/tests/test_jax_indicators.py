# =====================================================================
# WHEN 2026-06-28 | WHO Claude for Monty
# WHY  Parity for the OPTIONAL on-device indicators (jax_tpu/jax_indicators.py) vs the
#      CPU pandas reference (src/indicators/*). Confirms the jnp RSI/ATR/CCI/SMA/Bollinger
#      match value-for-value AND share the same NaN warmup prefix.
# WHERE jax_tpu/tests/test_jax_indicators.py
# =====================================================================
"""On-device indicator parity vs src/indicators/* (pandas reference, no TA-Lib needed)."""
from __future__ import annotations
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from src.indicators import sma as CSMA, rsi as CRSI, atr as CATR, cci as CCCI, bollinger as CBOL
from jax_tpu import jax_indicators as JI

ATOL = 1e-3   # float EWM accumulation tolerance


def _series(n=600, seed=5):
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    high = close + rng.uniform(0.0, 0.5, n)
    low = close - rng.uniform(0.0, 0.5, n)
    return high, low, close


def _cmp(name, cpu, jx):
    cpu = np.asarray(cpu, np.float64)
    jx = np.asarray(jx, np.float64)
    assert cpu.shape == jx.shape, f"{name}: shape {cpu.shape} vs {jx.shape}"
    cpu_nan, jx_nan = np.isnan(cpu), np.isnan(jx)
    # NaN warmup prefix must match (allow jnp to be valid where cpu valid)
    np.testing.assert_array_equal(cpu_nan, jx_nan, err_msg=f"{name}: NaN mask mismatch")
    ok = ~cpu_nan
    np.testing.assert_allclose(jx[ok], cpu[ok], atol=ATOL, rtol=1e-4, err_msg=f"{name}: value mismatch")


def test_sma_parity():
    _, _, close = _series()
    for period, shift in [(1, 0), (2, 1), (50, 0), (200, 0), (3, 2)]:
        _cmp(f"sma{period}.{shift}", CSMA.sma(close, period, shift), JI.sma(close, period, shift))


def test_rsi_parity():
    _, _, close = _series()
    for p in (4, 14):
        _cmp(f"rsi_raw{p}", CRSI.rsi_raw(close, p), JI.rsi_raw(close, p))
        _cmp(f"rsi_post{p}", CRSI.rsi_post(close, p), JI.rsi_post(close, p))


def test_atr_parity():
    high, low, close = _series()
    _cmp("atr_raw14", CATR.atr_raw(high, low, close, 14), JI.atr_raw(high, low, close, 14))
    _cmp("atr_post14", CATR.atr_post(high, low, close, 14), JI.atr_post(high, low, close, 14))


def test_cci_parity():
    high, low, close = _series()
    for p in (30, 100):
        _cmp(f"cci_raw{p}", CCCI.cci_raw(high, low, close, p), JI.cci_raw(high, low, close, p))
        _cmp(f"cci_post{p}", CCCI.cci_post(high, low, close, p), JI.cci_post(high, low, close, p))


def test_bollinger_parity():
    _, _, close = _series()
    for period in (20, 200):
        for dev in (0.5, 1.0, 2.0, 4.0):
            cu, cm, cl = CBOL.bollinger(close, period, dev)
            ju, jm, jl = JI.bollinger(close, period, dev)
            _cmp(f"bb{period}.{dev}.u", cu, ju)
            _cmp(f"bb{period}.{dev}.m", cm, jm)
            _cmp(f"bb{period}.{dev}.l", cl, jl)
