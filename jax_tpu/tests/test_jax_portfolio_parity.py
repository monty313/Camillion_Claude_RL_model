# =====================================================================
# WHEN 2026-06-28 | WHO Claude for Monty
# WHY  THE PORTFOLIO GATE. The shared-pot JAX env (jax_portfolio_env.py) is only valid if
#      it matches the CPU PortfolioEnv (src/env/portfolio_env.py) bar-for-bar: same 499 obs,
#      same reward (incl. alpha-shaping + midnight day-scoring + two-phase banking) across
#      the symbol cycle. This steps both on identical actions and asserts they match.
# WHERE jax_tpu/tests/test_jax_portfolio_parity.py
# =====================================================================
"""CPU<->JAX parity for the SHARED-POT PortfolioEnv (obs + reward across the symbol cycle)."""
from __future__ import annotations
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pandas as pd

from config.ftmo_config import load_ftmo_config
from src.env.portfolio_env import PortfolioEnv, build_portfolio_subs
from src.strategies.registry import AlphaRegistry
from src.strategies.alpha_pack import register_all
from jax_tpu import jax_static_features as JSF
from jax_tpu import jax_portfolio_env as JPE

ATOL_OBS = 1e-4
ATOL_REW = 1e-5


def _reg():
    r = AlphaRegistry(); register_all(r); return r


def _symbol_data(symbols, n_bars=4000, seed=21, drift=0.0):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2024-03-04 00:00:00").value
    time_ns = (t0 + np.arange(n_bars, dtype=np.int64) * 60_000_000_000).astype(np.int64)
    out = {}
    for k, s in enumerate(symbols):
        steps = rng.normal(drift, 1e-4, n_bars)
        close = (1.10 + 0.2 * k) + np.cumsum(steps).astype(np.float64)
        ind = rng.normal(0, 1.0, (n_bars, 220)).astype(np.float32)
        out[s] = (ind, close, time_ns)
    return out


def _run(symbols, continue_after_pass, n_steps=1600, seed=21, sym_data=None, actions=None, behaviors=None):
    cfg = load_ftmo_config()
    # v1.7.0 trade-risk behaviours (default OFF). behaviors = {bb_stop, risk_pct, band_bonus, reentry_bonus}.
    b = behaviors or {}
    bb_stop = bool(b.get("bb_stop", False))
    risk_pct = b.get("risk_pct", None)
    open_gate = bool(b.get("open_gate", False))
    if "band_bonus" in b or "reentry_bonus" in b or "conviction_bonus" in b:
        import dataclasses
        cfg = dataclasses.replace(cfg, band_stack_bonus=float(b.get("band_bonus", 0.0)),
                                  reentry_bonus=float(b.get("reentry_bonus", 0.0)),
                                  conviction_bonus=float(b.get("conviction_bonus", 0.0)))
    if sym_data is None:
        sym_data = _symbol_data(symbols, seed=seed)
    subs = build_portfolio_subs(sym_data, _reg, cfg=cfg, warmup=50, progress=False)
    env = PortfolioEnv(subs=subs, cfg=cfg, warmup=50, continue_after_pass=continue_after_pass,
                       bb_stop_enabled=bb_stop, risk_per_trade_pct=risk_pct, open_gate=open_gate)

    psd = JSF.build_portfolio_static(subs)
    static = JPE.make_portfolio_device_static(psd)
    params = JPE.portfolio_params(
        psd, daily_target_frac=cfg.daily_target_pct / 100.0,
        trailing_dd_frac=cfg.trailing_drawdown_pct / 100.0, daily_dd_frac=cfg.daily_drawdown_pct / 100.0,
        total_dd_frac=cfg.max_total_drawdown_pct / 100.0, profit_target_frac=cfg.profit_target_total_pct / 100.0,
        trailing_enabled=1.0 if cfg.trailing_enabled else 0.0,
        two_phase_enabled=1.0 if cfg.two_phase_enabled else 0.0,
        phase2_continue=1.0 if cfg.phase2_continue else 0.0, phase2_trailing_frac=cfg.phase2_trailing_pct / 100.0,
        continue_after_pass=1.0 if continue_after_pass else 0.0,
        alpha_on=1.0 if cfg.alpha_reward_enabled else 0.0, alpha_agree=cfg.alpha_agree_bonus,
        alpha_against=cfg.alpha_against_penalty, alpha_beat=cfg.alpha_beat_bonus,
        day_pass_reward=cfg.day_pass_reward, day_fail_penalty=cfg.day_fail_penalty,
        streak_bonus=cfg.streak_bonus, streak_bonus_cap=cfg.streak_bonus_cap,
        target_seek_weight=cfg.target_seek_weight, idle_day_penalty=cfg.idle_day_penalty,
        dd_proximity_coef=cfg.dd_proximity_coef, breach_penalty=cfg.breach_penalty, pass_bonus=cfg.pass_bonus,
        bb_stop_enabled=1.0 if bb_stop else 0.0, risk_based=1.0 if risk_pct is not None else 0.0,
        risk_frac=(float(risk_pct) / 100.0) if risk_pct is not None else 0.0,
        band_stack_bonus=cfg.band_stack_bonus, reentry_bonus=cfg.reentry_bonus,
        conviction_bonus=cfg.conviction_bonus, open_gate=1.0 if open_gate else 0.0)

    cpu_obs, _ = env.reset()
    state = JPE.init_state(static, params, start=env.warmup, end=env.T - 1,
                           daily_target_frac=cfg.daily_target_pct / 100.0,
                           trailing_dd_frac=cfg.trailing_drawdown_pct / 100.0)
    jx_obs = np.asarray(JPE.reset_obs(state, static, params))
    np.testing.assert_allclose(jx_obs, cpu_obs, atol=ATOL_OBS, err_msg="RESET obs mismatch")

    step = jax.jit(JPE.step_portfolio, static_argnums=(3,))
    if actions is None:
        actions = np.random.default_rng(7).integers(0, 4, size=n_steps)
    mo = mr = 0.0
    events = {"banked": False, "won_day": False, "breach": False}  # alpha-shaping is proven by the reward atol
    prev_streak = 0
    for k, a in enumerate(actions):
        cpu_obs, cpu_r, cpu_term, cpu_trunc, info = env.step(int(a))
        state, jx_obs, jx_r, jx_term, jx_trunc = step(state, int(a), static, params)
        jx_obs = np.asarray(jx_obs); jx_r = float(jx_r)
        mo = max(mo, float(np.max(np.abs(jx_obs - cpu_obs)))); mr = max(mr, abs(jx_r - cpu_r))
        np.testing.assert_allclose(jx_obs, cpu_obs, atol=ATOL_OBS,
                                   err_msg=f"step {k} obs mismatch (action={a}, j just decided)")
        assert abs(jx_r - cpu_r) < ATOL_REW, f"step {k} reward: cpu={cpu_r:.10f} jax={jx_r:.10f} (action={a})"
        assert bool(jx_term > 0.5) == bool(cpu_term), f"step {k} terminated mismatch"
        assert bool(jx_trunc > 0.5) == bool(cpu_trunc), f"step {k} truncated mismatch"
        events["banked"] |= bool(info.get("day_locked") or info.get("phase2_active"))
        if info.get("daily_pass_streak", 0) > prev_streak:
            events["won_day"] = True
        prev_streak = info.get("daily_pass_streak", 0)
        events["breach"] |= bool(cpu_term and not env.acc.episode_passed)
        if cpu_term or cpu_trunc:
            break
    return mo, mr, events


def test_portfolio_parity_two_symbols_continue():
    mo, mr, ev = _run(["EURUSD", "GBPUSD"], continue_after_pass=True, n_steps=2000)
    print(f"\n[portfolio EURUSD,GBPUSD continue] max|obs|={mo:.2e} max|reward|={mr:.2e} events={ev}")


def test_portfolio_parity_three_symbols_terminate():
    mo, mr, ev = _run(["EURUSD", "GBPUSD", "XAUUSD"], continue_after_pass=False, n_steps=2000, seed=33)
    print(f"\n[portfolio 3sym terminate] max|obs|={mo:.2e} max|reward|={mr:.2e} events={ev}")


def test_portfolio_parity_banking_and_won_days():
    """Strong uptrend + BUY-heavy actions -> forces two-phase banking + WON days (+2.5%) + the
    4-in-a-row bonus path. Asserts those rare branches actually fire AND still match the CPU."""
    syms = ["EURUSD", "GBPUSD"]
    sym_data = _symbol_data(syms, seed=5, drift=6e-5)          # upward drift
    acts = np.where(np.arange(6000) % 7 < 5, 1, 0)            # mostly BUY, some HOLD
    mo, mr, ev = _run(syms, continue_after_pass=True, sym_data=sym_data, actions=acts)
    print(f"\n[portfolio uptrend bank/won] max|obs|={mo:.2e} max|reward|={mr:.2e} events={ev}")
    assert ev["banked"], "expected two-phase banking to fire on the uptrend (test is too weak otherwise)"
    assert ev["won_day"], "expected at least one WON day (+2.5%) on the uptrend"


def test_portfolio_parity_trade_risk_behaviors_on():
    """v1.7.0: with the BB(10,1) HARD STOP + RISK-BASED sizing + band-stack & re-entry CLOSE bonuses ALL ON,
    the JAX env must STILL match the CPU PortfolioEnv bar-for-bar (517 obs + reward). An uptrend + BUY-heavy
    actions exercise risk-sized entries, hard-stop closes, and the band-stack bonus."""
    syms = ["EURUSD", "GBPUSD"]
    sym_data = _symbol_data(syms, seed=5, drift=6e-5)
    acts = np.where(np.arange(6000) % 7 < 5, 1, 0)
    behaviors = {"bb_stop": True, "risk_pct": 0.1, "band_bonus": 0.005, "reentry_bonus": 0.003,
                 "conviction_bonus": 0.1, "open_gate": True}
    mo, mr, ev = _run(syms, continue_after_pass=True, sym_data=sym_data, actions=acts, behaviors=behaviors)
    print(f"\n[portfolio trade-risk behaviors ON] max|obs|={mo:.2e} max|reward|={mr:.2e} events={ev}")
    assert mo < ATOL_OBS and mr < ATOL_REW, f"behaviors-on parity broke: max|obs|={mo:.2e} max|reward|={mr:.2e}"


def test_portfolio_parity_breach():
    """Strong downtrend + BUY-and-hold -> forces a drawdown BREACH (episode terminates). Parity must
    hold through the breach + penalty."""
    syms = ["EURUSD", "GBPUSD"]
    sym_data = _symbol_data(syms, seed=9, drift=-1.5e-4)       # downward drift
    acts = np.ones(6000, dtype=int)                           # BUY and hold into the drawdown
    mo, mr, ev = _run(syms, continue_after_pass=False, sym_data=sym_data, actions=acts)
    print(f"\n[portfolio downtrend breach] max|obs|={mo:.2e} max|reward|={mr:.2e} events={ev}")
    assert ev["breach"], "expected a drawdown breach on the downtrend"
