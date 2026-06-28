# =====================================================================
# WHEN 2026-06-28 | WHO Claude for Monty
# WHY  THE GATE. The JAX env is only a valid second implementation if it produces
#      the SAME observation + reward as the CPU env on identical bars/actions. This
#      file proves it at three levels: (1) FTMO breach math, (2) the 40 dynamic obs
#      floats, (3) full STEP parity (obs+reward) of jax_env vs CPU TradingEnv.
# WHERE jax_tpu/tests/test_jax_parity.py
# HOW   Build a small synthetic symbol (random-walk close + fabricated indicator
#       array) with the REAL alpha registry, instantiate the CPU TradingEnv, derive
#       the JAX static tensor from it, then step both on a scripted action sequence
#       and assert obs match (atol 1e-4) and reward matches (atol 1e-5).
# DEPENDS_ON: jax, numpy, src/* (CPU reference), jax_tpu/*
# USED_BY: pytest jax_tpu/tests/test_jax_parity.py   (or python jax_tpu/tests/run_parity.py)
# =====================================================================
"""CPU<->JAX parity: FTMO rules, dynamic obs blocks, and full step parity (THE GATE)."""
from __future__ import annotations
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)   # float64 money math -> near-exact step parity
import jax.numpy as jnp

from config.ftmo_config import load_ftmo_config
from config import constants as C
from src.account.account_state import AccountState
from src.account import win_loss_features as WL
from src.risk.ftmo_rules import FTMORules

from jax_tpu import jax_ftmo, jax_obs_blocks

ATOL_OBS = 1e-4
ATOL_REW = 1e-5


# --------------------------------------------------------------------------
# 1) FTMO breach parity
# --------------------------------------------------------------------------
def test_ftmo_breach_parity():
    cfg = load_ftmo_config()
    rules = FTMORules(cfg)
    start = 100_000.0
    # (equity, day_start, peak) grid spanning breach / no-breach / banking
    cases = [
        (100_000, 100_000, 100_000),
        (96_500, 100_000, 100_000),    # 4% trailing breach
        (97_000, 100_000, 100_000),    # safe
        (95_500, 99_000, 100_000),     # daily + trailing
        (89_000, 100_000, 100_000),    # total + trailing
        (102_600, 100_000, 102_600),   # +2.6% -> daily target hit -> auto-flat
        (102_400, 100_000, 102_400),   # +2.4% -> not yet
    ]
    dt = cfg.daily_target_pct / 100.0
    dd = cfg.daily_drawdown_pct / 100.0
    tot = cfg.max_total_drawdown_pct / 100.0
    tr = cfg.trailing_drawdown_pct / 100.0
    ten = 1.0 if cfg.trailing_enabled else 0.0
    two = 1.0 if cfg.two_phase_enabled else 0.0
    for eq, day0, peak in cases:
        acc = AccountState(starting_balance=start)
        acc.balance = day0           # day starts here
        acc.day_start_balance = day0
        acc.episode_peak_equity = peak
        acc.equity = float(eq)
        cpu_breach = bool(rules.reasons(acc))
        cpu_flat = bool(rules.should_auto_flat(acc))
        b = jax_ftmo.breach(float(eq), float(day0), start, float(peak), dd, tot, tr, ten)
        jax_breach = bool(np.asarray(b["any_breach"]) > 0.5)
        jax_flat = bool(np.asarray(jax_ftmo.should_auto_flat(float(eq), float(day0), start, dt, two)) > 0.5)
        assert jax_breach == cpu_breach, f"breach mismatch eq={eq}: cpu={cpu_breach} jax={jax_breach}"
        assert jax_flat == cpu_flat, f"auto_flat mismatch eq={eq}: cpu={cpu_flat} jax={jax_flat}"


# --------------------------------------------------------------------------
# 2) Dynamic obs-block parity (account_daily / episode / portfolio / sizing / recent_context)
# --------------------------------------------------------------------------
def _make_acc(rng, start=100_000.0):
    acc = AccountState(starting_balance=start)
    # randomize a plausible mid-episode state
    acc.balance = start + rng.uniform(-3000, 4000)
    acc.equity = acc.balance + rng.uniform(-1500, 1500)
    acc.day_start_balance = start + rng.uniform(-2000, 3000)
    acc.episode_start_balance = start
    acc.episode_peak_equity = max(acc.equity, start + rng.uniform(0, 5000))
    acc.day_peak_equity = max(acc.equity, acc.day_start_balance)
    acc.daily_realized_pnl = rng.uniform(-2000, 3000)
    acc.episode_realized_pnl = rng.uniform(-4000, 6000)
    acc.daily_wins = int(rng.integers(0, 8)); acc.daily_losses = int(rng.integers(0, 8))
    acc.daily_trades = acc.daily_wins + acc.daily_losses
    acc.episode_wins = int(rng.integers(0, 30)); acc.episode_losses = int(rng.integers(0, 30))
    acc.episode_trades = acc.episode_wins + acc.episode_losses
    acc.daily_consecutive_losses = int(rng.integers(0, 6))
    acc.episode_consecutive_losses = int(rng.integers(0, 9))
    acc.episode_breached = bool(rng.integers(0, 2))
    return acc


def test_dynamic_obs_block_parity():
    cfg = load_ftmo_config()
    rng = np.random.default_rng(0)
    dt = cfg.daily_target_pct / 100.0
    dd = cfg.daily_drawdown_pct / 100.0
    tot = cfg.max_total_drawdown_pct / 100.0
    tr = cfg.trailing_drawdown_pct / 100.0
    pt = cfg.profit_target_total_pct / 100.0
    ten = 1.0 if cfg.trailing_enabled else 0.0
    for trial in range(200):
        acc = _make_acc(rng)
        position = float(rng.choice([-1.0, 0.0, 1.0]))
        # set the single-symbol portfolio aggregates exactly like TradingEnv._portfolio_block
        acc.open_positions = 1 if position != 0 else 0
        acc.net_exposure = position
        acc.gross_exposure = abs(position)
        acc.unrealized_pnl = acc.equity - acc.balance
        acc.largest_position_dir = int(np.sign(position))
        value_per_point = float(rng.uniform(1.0, 100_000.0))
        position_size = float(rng.uniform(1.0, 100_000.0))
        ref_move = float(rng.uniform(0.0, 50.0))
        week_avg = float(rng.uniform(1e-6, 50.0))
        prev_day = float(rng.uniform(0.0, 50.0)); prev2 = float(rng.uniform(0.0, 50.0))
        today_sofar = float(rng.uniform(0.0, 50.0))
        typical_range = float(rng.choice([0.0, rng.uniform(1.0, 60.0)]))
        days_elapsed = float(rng.integers(0, 25))
        typ_arg = typical_range if typical_range > 0 else None

        # --- CPU reference blocks ---
        cpu_daily = WL.daily_features(acc, cfg)
        cpu_epi = WL.episode_features(acc, cfg)
        cpu_port = WL.portfolio_features(acc)
        cpu_size = WL.sizing_features(acc, cfg, value_per_point=value_per_point,
                                      ref_move=ref_move, position_size=position_size)
        cpu_ctx = WL.recent_context_features(acc, cfg, week_avg=week_avg, prev_day=prev_day,
                                             prev2=prev2, today_sofar=today_sofar,
                                             typical_range=typ_arg, days_elapsed=days_elapsed)

        # --- JAX blocks ---
        j_daily = jax_obs_blocks.daily_features(
            acc.equity, acc.day_start_balance, acc.starting_balance, acc.daily_realized_pnl,
            float(acc.daily_wins), float(acc.daily_trades), float(acc.daily_consecutive_losses), dt, dd)
        j_epi = jax_obs_blocks.episode_features(
            acc.equity, acc.episode_start_balance, acc.starting_balance, acc.episode_realized_pnl,
            float(acc.episode_wins), float(acc.episode_trades), float(acc.episode_consecutive_losses),
            float(acc.episode_breached), acc.episode_peak_equity, tr, tot, ten, pt)
        j_port = jax_obs_blocks.portfolio_features(acc.equity, acc.balance, acc.starting_balance, position)
        j_size = jax_obs_blocks.sizing_features(
            acc.equity, acc.day_start_balance, acc.starting_balance, acc.episode_peak_equity,
            value_per_point, ref_move, position_size, dt, tr, tot, ten)
        j_ctx = jax_obs_blocks.recent_context_features(
            acc.equity, acc.starting_balance, week_avg, prev_day, prev2, today_sofar,
            typical_range, days_elapsed, dt, pt)

        for name, cpu, jx in [("daily", cpu_daily, j_daily), ("episode", cpu_epi, j_epi),
                              ("portfolio", cpu_port, j_port), ("sizing", cpu_size, j_size),
                              ("recent_context", cpu_ctx, j_ctx)]:
            np.testing.assert_allclose(
                np.asarray(jx, dtype=np.float32), np.asarray(cpu, dtype=np.float32),
                atol=ATOL_OBS, err_msg=f"[{name}] block mismatch on trial {trial}")


# --------------------------------------------------------------------------
# 3) FULL STEP PARITY — the gate. CPU TradingEnv vs JAX env, obs + reward.
# --------------------------------------------------------------------------
def _synthetic_symbol(n_bars=3000, seed=7):
    """Random-walk close + noise indicators + 1-min timestamps spanning a few days."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    close = 1.10 + np.cumsum(rng.normal(0, 1e-4, size=n_bars)).astype(np.float64)
    ind = rng.normal(0, 1.0, size=(n_bars, C.N_INDICATORS_TOTAL)).astype(np.float32)
    t0 = pd.Timestamp("2024-03-04 00:00:00").value  # a Monday 00:00 UTC
    time_ns = (t0 + np.arange(n_bars, dtype=np.int64) * 60_000_000_000).astype(np.int64)
    return ind, close, time_ns


def _build_envs(symbol, position_size, n_bars=3000, warmup=5, seed=7):
    from src.env.trading_env import TradingEnv
    from src.strategies.registry import AlphaRegistry
    from src.strategies.alpha_pack import register_all
    from jax_tpu import jax_static_features as JSF
    from jax_tpu import jax_env as JE

    ind, close, time_ns = _synthetic_symbol(n_bars, seed)
    reg = AlphaRegistry(); register_all(reg)
    cfg = load_ftmo_config()
    env = TradingEnv(ind, close, time_ns, reg, cfg=cfg, position_size=position_size,
                     warmup=warmup, symbol=symbol)
    sd = JSF.build_static_data(env)
    static = JE.make_device_static(sd)
    params = JE.params_from_static(
        sd,
        daily_target_frac=cfg.daily_target_pct / 100.0,
        trailing_dd_frac=cfg.trailing_drawdown_pct / 100.0,
        daily_dd_frac=cfg.daily_drawdown_pct / 100.0,
        total_dd_frac=cfg.max_total_drawdown_pct / 100.0,
        profit_target_frac=cfg.profit_target_total_pct / 100.0,
        trailing_enabled=1.0 if cfg.trailing_enabled else 0.0,
        two_phase_enabled=1.0 if cfg.two_phase_enabled else 0.0,
        phase2_continue=1.0 if cfg.phase2_continue else 0.0,
        phase2_trailing_frac=cfg.phase2_trailing_pct / 100.0,
    )
    return env, sd, static, params


def _run_parity(symbol, position_size, n_steps=900, seed=7):
    from jax_tpu import jax_env as JE
    env, sd, static, params = _build_envs(symbol, position_size, seed=seed)

    cpu_obs, _ = env.reset()
    cfg = load_ftmo_config()
    state = JE.init_state(static, params, start=sd.warmup, end=sd.T - 1,
                          daily_target_frac=cfg.daily_target_pct / 100.0,
                          trailing_dd_frac=cfg.trailing_drawdown_pct / 100.0)
    jax_obs = np.asarray(JE.reset_obs(state, static, params))
    np.testing.assert_allclose(jax_obs, cpu_obs, atol=ATOL_OBS,
                               err_msg=f"[{symbol}] RESET obs mismatch")

    step_jit = jax.jit(JE.step_env, static_argnums=(3,))
    rng = np.random.default_rng(123)
    actions = rng.integers(0, 4, size=n_steps)
    max_abs_obs = 0.0
    max_abs_rew = 0.0
    for k, a in enumerate(actions):
        cpu_obs, cpu_r, cpu_term, cpu_trunc, _ = env.step(int(a))
        state, jx_obs, jx_r, jx_term, jx_trunc = step_jit(state, int(a), static, params)
        jx_obs = np.asarray(jx_obs); jx_r = float(jx_r)
        max_abs_obs = max(max_abs_obs, float(np.max(np.abs(jx_obs - cpu_obs))))
        max_abs_rew = max(max_abs_rew, abs(jx_r - cpu_r))
        np.testing.assert_allclose(jx_obs, cpu_obs, atol=ATOL_OBS,
                                   err_msg=f"[{symbol}] step {k} obs mismatch (action={a})")
        assert abs(jx_r - cpu_r) < ATOL_REW, \
            f"[{symbol}] step {k} reward mismatch: cpu={cpu_r:.10f} jax={jx_r:.10f} (action={a})"
        assert bool(jx_term > 0.5) == bool(cpu_term), f"[{symbol}] step {k} terminated mismatch"
        assert bool(jx_trunc > 0.5) == bool(cpu_trunc), f"[{symbol}] step {k} truncated mismatch"
        if cpu_term or cpu_trunc:
            break
    return max_abs_obs, max_abs_rew


def test_step_parity_pair():
    """THE GATE — non-index symbol (NY bonus off). Full obs+reward parity over many steps."""
    mo, mr = _run_parity("EURUSD", position_size=100_000.0, n_steps=1200, seed=7)
    print(f"\n[EURUSD] max|obs|={mo:.2e}  max|reward|={mr:.2e}")


def test_step_parity_index_ny_bonus():
    """Index symbol exercises the NY-session bonus path; smaller size to run longer."""
    mo, mr = _run_parity("US30", position_size=50.0, n_steps=1200, seed=11)
    print(f"\n[US30] max|obs|={mo:.2e}  max|reward|={mr:.2e}")


def test_step_parity_open_gate():
    """open_gate=True: new directional opens are blocked when the 5m CCI is neutral. Parity must hold
    on the gated path too (regression for the adversarial-review finding that JAX lacked open_gate)."""
    import pandas as pd
    from src.env.trading_env import TradingEnv
    from src.strategies.registry import AlphaRegistry
    from src.strategies.alpha_pack import register_all
    from jax_tpu import jax_static_features as JSF, jax_env as JE
    rng = np.random.default_rng(13); n = 3000
    close = 1.10 + np.cumsum(rng.normal(0, 1e-4, n)).astype(np.float64)
    ind = (rng.normal(0, 1, (n, C.N_INDICATORS_TOTAL)) * 100.0).astype(np.float32)  # big -> gate toggles
    t0 = pd.Timestamp("2024-03-04").value
    time_ns = (t0 + np.arange(n) * 60_000_000_000).astype(np.int64)
    reg = AlphaRegistry(); register_all(reg)
    cfg = load_ftmo_config()
    env = TradingEnv(ind, close, time_ns, reg, cfg=cfg, position_size=100.0, warmup=5,
                     symbol="EURUSD", open_gate=True)
    assert env.open_gate_blocked.any() and not env.open_gate_blocked.all(), "gate not toggling — test too weak"
    sd = JSF.build_static_data(env); static = JE.make_device_static(sd)
    params = JE.params_from_static(
        sd, daily_target_frac=cfg.daily_target_pct / 100, trailing_dd_frac=cfg.trailing_drawdown_pct / 100,
        daily_dd_frac=cfg.daily_drawdown_pct / 100, total_dd_frac=cfg.max_total_drawdown_pct / 100,
        profit_target_frac=cfg.profit_target_total_pct / 100, two_phase_enabled=1.0,
        phase2_continue=1.0 if cfg.phase2_continue else 0.0, phase2_trailing_frac=cfg.phase2_trailing_pct / 100,
        open_gate=1.0)
    cpu_obs, _ = env.reset()
    state = JE.init_state(static, params, sd.warmup, sd.T - 1, cfg.daily_target_pct / 100, cfg.trailing_drawdown_pct / 100)
    np.testing.assert_allclose(np.asarray(JE.reset_obs(state, static, params)), cpu_obs, atol=ATOL_OBS)
    step = jax.jit(JE.step_env, static_argnums=(3,))
    for k, a in enumerate(np.random.default_rng(5).integers(0, 4, 900)):
        cpu_obs, cr, ct, ctr, _ = env.step(int(a))
        state, jo, jr, jt, jtr = step(state, int(a), static, params)
        np.testing.assert_allclose(np.asarray(jo), cpu_obs, atol=ATOL_OBS, err_msg=f"open_gate step {k} (a={a})")
        assert abs(float(jr) - cr) < ATOL_REW, f"open_gate step {k} reward"
        if ct or ctr:
            break
