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


def _symbol_data(symbols, n_bars=4000, seed=21, drift=0.0, with_aux=False):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2024-03-04 00:00:00").value
    time_ns = (t0 + np.arange(n_bars, dtype=np.int64) * 60_000_000_000).astype(np.int64)
    out = {}
    for k, s in enumerate(symbols):
        steps = rng.normal(drift, 1e-4, n_bars)
        close = (1.10 + 0.2 * k) + np.cumsum(steps).astype(np.float64)
        ind = rng.normal(0, 1.0, (n_bars, 220)).astype(np.float32)
        if with_aux:                                   # real 1m High/Low -> the hug block is NON-zero
            aux = np.zeros((n_bars, 32), np.float32)
            aux[:, 1] = close + 1e-3                    # 1m__high
            aux[:, 2] = close - 1e-3                    # 1m__low
            out[s] = (ind, close, time_ns, aux)
        else:
            out[s] = (ind, close, time_ns)
    return out


def _run(symbols, continue_after_pass, n_steps=1600, seed=21, sym_data=None, actions=None, behaviors=None):
    cfg = load_ftmo_config()
    # v1.7.0 trade-risk behaviours (default OFF). behaviors = {bb_stop, risk_pct, band_bonus, reentry_bonus}.
    b = behaviors or {}
    bb_stop = bool(b.get("bb_stop", False))
    risk_pct = b.get("risk_pct", None)
    open_gate = bool(b.get("open_gate", False))
    trade_wheels = bool(b.get("trade_wheels", False))          # training wheels: hard directional open-gate
    exit_band_pen = float(b.get("exit_band_penalty", 0.0))     # v1.13.0: exit-band penalty magnitude (0 = off)
    bracket_on = bool(b.get("bracket", False))                 # v1.12.0: TP/SL/lot bracket model ON
    tp01, sl01, lot01 = float(b.get("tp01", 0.0)), float(b.get("sl01", 0.0)), float(b.get("lot01", 0.0))
    if any(k in b for k in ("band_bonus", "reentry_bonus", "conviction_bonus", "hug_bonus", "hug_miss")):
        import dataclasses
        cfg = dataclasses.replace(cfg, band_stack_bonus=float(b.get("band_bonus", 0.0)),
                                  reentry_bonus=float(b.get("reentry_bonus", 0.0)),
                                  conviction_bonus=float(b.get("conviction_bonus", 0.0)),
                                  hug_pressure_bonus=float(b.get("hug_bonus", 0.0)),
                                  hug_miss_penalty=float(b.get("hug_miss", 0.0)))
    if "daily_target_pct" in b:                        # test-only: lower the goal so the +2.5% MUTE path fires
        import dataclasses
        cfg = dataclasses.replace(cfg, daily_target_pct=float(b["daily_target_pct"]))
    if sym_data is None:
        sym_data = _symbol_data(symbols, seed=seed)
    subs = build_portfolio_subs(sym_data, _reg, cfg=cfg, warmup=50, progress=False)
    env = PortfolioEnv(subs=subs, cfg=cfg, warmup=50, continue_after_pass=continue_after_pass,
                       bb_stop_enabled=bb_stop, risk_per_trade_pct=risk_pct, open_gate=open_gate,
                       bracket_enabled=bracket_on, trade_wheels=trade_wheels, exit_band_penalty=exit_band_pen)

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
        conviction_bonus=cfg.conviction_bonus, open_gate=1.0 if open_gate else 0.0,
        hug_pressure_bonus=cfg.hug_pressure_bonus, hug_miss_penalty=cfg.hug_miss_penalty,
        overtrade_soft_cap=cfg.overtrade_soft_cap, overtrade_penalty=cfg.overtrade_penalty,
        bracket_enabled=1.0 if bracket_on else 0.0,
        trade_wheels=1.0 if trade_wheels else 0.0, exit_band_penalty=exit_band_pen)

    cpu_obs, _ = env.reset()
    state = JPE.init_state(static, params, start=env.warmup, end=env.T - 1,
                           daily_target_frac=cfg.daily_target_pct / 100.0,
                           trailing_dd_frac=cfg.trailing_drawdown_pct / 100.0)
    jx_obs = np.asarray(JPE.reset_obs(state, static, params))
    np.testing.assert_allclose(jx_obs, cpu_obs, atol=ATOL_OBS, err_msg="RESET obs mismatch")

    step = jax.jit(JPE.step_portfolio, static_argnums=(6,))   # static = arg 5 after +tp01/sl01/lot01
    if actions is None:
        actions = np.random.default_rng(7).integers(0, 4, size=n_steps)
    mo = mr = 0.0
    events = {"banked": False, "won_day": False, "breach": False, "opens": 0}  # alpha-shaping proven by reward atol
    prev_streak = 0
    prev_pos = {s: 0 for s in symbols}
    for k, a in enumerate(actions):
        cpu_obs, cpu_r, cpu_term, cpu_trunc, info = env.step(int(a), tp01, sl01, lot01)
        state, jx_obs, jx_r, jx_term, jx_trunc = step(state, int(a), tp01, sl01, lot01, static, params)
        jx_obs = np.asarray(jx_obs); jx_r = float(jx_r)
        mo = max(mo, float(np.max(np.abs(jx_obs - cpu_obs)))); mr = max(mr, abs(jx_r - cpu_r))
        np.testing.assert_allclose(jx_obs, cpu_obs, atol=ATOL_OBS,
                                   err_msg=f"step {k} obs mismatch (action={a}, j just decided)")
        assert abs(jx_r - cpu_r) < ATOL_REW, f"step {k} reward: cpu={cpu_r:.10f} jax={jx_r:.10f} (action={a})"
        assert bool(jx_term > 0.5) == bool(cpu_term), f"step {k} terminated mismatch"
        assert bool(jx_trunc > 0.5) == bool(cpu_trunc), f"step {k} truncated mismatch"
        for s in symbols:                                    # count NEW opens (flat/opposite -> a direction)
            np_ = info["positions"].get(s, 0)
            if np_ != 0 and np_ != prev_pos[s]:
                events["opens"] += 1
            prev_pos[s] = np_
        events["banked"] |= bool(info.get("day_locked") or info.get("phase2_active"))
        if info.get("daily_pass_streak", 0) > prev_streak:
            events["won_day"] = True
        prev_streak = info.get("daily_pass_streak", 0)
        events["breach"] |= bool(cpu_term and not env.acc.episode_passed)
        if cpu_term or cpu_trunc:
            break
    events["bracket_closed"] = sum(1 for e in getattr(env, "_bracket_log", []) if e["event"] == "close")
    events["wheel_blocks"] = int(getattr(env, "_wheel_blocks", 0))   # opens the training-wheels vetoed
    events["exit_band_blocks"] = int(getattr(env, "_exit_band_blocks", 0))  # closes the exit-band penalty fired on
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
    the JAX env must STILL match the CPU PortfolioEnv bar-for-bar (563 obs + reward). An uptrend + BUY-heavy
    actions exercise risk-sized entries, hard-stop closes, and the band-stack bonus."""
    syms = ["EURUSD", "GBPUSD"]
    sym_data = _symbol_data(syms, seed=5, drift=6e-5)
    acts = np.where(np.arange(6000) % 7 < 5, 1, 0)
    behaviors = {"bb_stop": True, "risk_pct": 0.1, "band_bonus": 0.005, "reentry_bonus": 0.003,
                 "conviction_bonus": 0.1, "open_gate": True}
    mo, mr, ev = _run(syms, continue_after_pass=True, sym_data=sym_data, actions=acts, behaviors=behaviors)
    print(f"\n[portfolio trade-risk behaviors ON] max|obs|={mo:.2e} max|reward|={mr:.2e} events={ev}")
    assert mo < ATOL_OBS and mr < ATOL_REW, f"behaviors-on parity broke: max|obs|={mo:.2e} max|reward|={mr:.2e}"


def test_portfolio_parity_hug_pressure_on():
    """v1.10.0: with the HEAVY hugging-pressure reward ON (ride bonus + indices/metals miss-penalty) on
    INDEX + METAL symbols (US30, XAUUSD) fed real 1m High/Low (aux) on a trend, the JAX env must STILL match
    the CPU PortfolioEnv bar-for-bar (563 obs + reward) — including the >=3-TF hug gate, the conflict carve-out,
    and the post-+2.5%-goal MUTING (no penalty + zeroed hug obs). Random actions -> alignment varies so BOTH
    the ride-bonus and the miss-penalty paths fire."""
    syms = ["US30", "XAUUSD"]                                   # index + metal -> miss-penalty applies to both
    sym_data = _symbol_data(syms, seed=5, drift=1e-3, with_aux=True)   # steep uptrend -> hug fires + +2.5% reached
    # flat FIRST (sit out a clean index/metal hug -> MISS-PENALTY), then BUY-heavy (aligned -> RIDE bonus, and
    # equity crosses +2.5% -> the daily-goal MUTE). All three hug paths exercised under parity.
    acts = np.where(np.arange(6000) < 1500, 0, np.where(np.arange(6000) % 7 < 5, 1, 0))
    behaviors = {"hug_bonus": 0.01, "hug_miss": 0.02, "daily_target_pct": 0.1}   # tiny goal -> MUTE path fires
    mo, mr, ev = _run(syms, continue_after_pass=True, sym_data=sym_data, actions=acts, behaviors=behaviors)
    print(f"\n[portfolio HUG-PRESSURE on] max|obs|={mo:.2e} max|reward|={mr:.2e} events={ev}")
    assert mo < ATOL_OBS and mr < ATOL_REW, f"hug-on parity broke: max|obs|={mo:.2e} max|reward|={mr:.2e}"
    assert ev["banked"], "expected the +2.5% goal to be reached (exercises the hug MUTE path)"


def test_portfolio_parity_brackets_on():
    """v1.12.0 Stage 2b: with the TP/SL/lot BRACKET model ON (fed fixed tp01/sl01/lot01), on a trend that makes
    the brackets actually FIRE, the JAX env must match the CPU PortfolioEnv bar-for-bar (563 obs + reward) --
    the locked TP/SL prices, the intrabar high/low exit at the LEVEL, the 1%-risk lot clamp, and the trade
    tallies. This is THE Stage-2b gate (CPU <-> JAX bracket execution to ~1e-7)."""
    syms = ["US30", "XAUUSD"]                                   # index + metal, with real 1m high/low
    sym_data = _symbol_data(syms, seed=7, drift=2e-4, with_aux=True)   # uptrend -> the close reaches the TP level
    # BUY for 2 steps every 120, HOLD the rest: a position opens, HOLDS long enough for the close to drift up to
    # its locked TP (~55 bars at this drift) -> the bracket FIRES; then flat until the next BUY. Many TP cycles.
    acts = np.where(np.arange(7000) % 120 < 2, 1, 0)
    behaviors = {"bracket": True, "tp01": 0.5, "sl01": 0.5, "lot01": 0.3}
    mo, mr, ev = _run(syms, continue_after_pass=True, sym_data=sym_data, actions=acts, behaviors=behaviors)
    print(f"\n[portfolio BRACKETS on] max|obs|={mo:.2e} max|reward|={mr:.2e} events={ev}")
    assert mo < ATOL_OBS and mr < ATOL_REW, f"bracket parity broke: max|obs|={mo:.2e} max|reward|={mr:.2e}"
    assert ev["bracket_closed"] > 0, "brackets never fired — test is vacuous (raise drift / lower tp01)"


def test_portfolio_parity_brackets_reversals():
    """v1.12.0 audit fix: on a REVERSAL (close-leg + bracket open in the SAME step) with another symbol holding
    an open position, the bracket 1%-equity CLAMP must use the PRE-STEP equity in BOTH envs (CPU eq_before, JAX
    eq_before) -> identical lot. Alternating BUY/SELL forces flips; 2 symbols means one holds while the other
    reverses. Must match bar-for-bar."""
    syms = ["US30", "XAUUSD"]
    sym_data = _symbol_data(syms, seed=7, drift=2e-4, with_aux=True)
    acts = np.where(np.arange(7000) % 8 < 4, 1, 2)             # BUY 4, SELL 4 -> reversals (close+open same step)
    behaviors = {"bracket": True, "tp01": 0.6, "sl01": 0.4, "lot01": 0.5}
    mo, mr, ev = _run(syms, continue_after_pass=True, sym_data=sym_data, actions=acts, behaviors=behaviors)
    print(f"\n[portfolio BRACKET REVERSALS] max|obs|={mo:.2e} max|reward|={mr:.2e} events={ev}")
    assert mo < ATOL_OBS and mr < ATOL_REW, f"reversal-clamp parity broke: max|obs|={mo:.2e} max|reward|={mr:.2e}"


def test_portfolio_parity_trade_wheels_on():
    """TRAINING WHEELS: with the operator's hard directional open-gate ON, a NEW BUY may only open where
    buy_allowed and a NEW SELL only where sell_allowed (the conditions in trade_permission.py). The JAX env
    must match the CPU PortfolioEnv bar-for-bar (563 obs + reward), AND the wheels must actually BLOCK some
    opens (else the test is vacuous) — proven by comparing open-counts wheels-OFF vs wheels-ON on identical
    actions/data."""
    syms = ["EURUSD", "GBPUSD"]
    sym_data = _symbol_data(syms, seed=13)
    acts = np.random.default_rng(3).integers(0, 4, size=3000)  # mixed BUY/SELL/CLOSE/HOLD -> both wheels tested
    mo, mr, ev = _run(syms, continue_after_pass=True, sym_data=sym_data, actions=acts,
                      behaviors={"trade_wheels": True})
    print(f"\n[portfolio TRAINING WHEELS] max|obs|={mo:.2e} max|reward|={mr:.2e} "
          f"wheel_blocks={ev['wheel_blocks']} opens={ev['opens']} events={ev}")
    assert mo < ATOL_OBS and mr < ATOL_REW, f"wheels parity broke: max|obs|={mo:.2e} max|reward|={mr:.2e}"
    assert ev["wheel_blocks"] > 0, (
        f"wheels vetoed NOTHING (wheel_blocks=0) — mask path never exercised, test is vacuous")


def test_portfolio_parity_exit_band_penalty_on():
    """EXIT-BAND (v1.13.0): with the exit-band penalty ON, a close landing OUTSIDE the direction's 1m BB(20,0.5)
    band (a BUY judged vs the band on High, a SELL vs the band on Low) is punished -- on the agent's own
    close/flip AND on a TP/SL bracket exit. The JAX env must match the CPU PortfolioEnv bar-for-bar (563 obs +
    reward), AND the penalty must actually FIRE (else the test is vacuous)."""
    syms = ["EURUSD", "GBPUSD"]
    sym_data = _symbol_data(syms, seed=17, with_aux=True, drift=-6e-5)  # downtrend + BUY -> SL bracket exits fire
    pat = np.arange(4000) % 8
    acts = np.where(pat < 6, 1, np.where(pat == 6, 3, 0))      # BUY-heavy (persist -> tight SL hits) + periodic CLOSE
    mo, mr, ev = _run(syms, continue_after_pass=False, sym_data=sym_data, actions=acts,
                      behaviors={"exit_band_penalty": 0.05, "bracket": True, "tp01": 0.9, "sl01": 0.03})
    print(f"\n[portfolio EXIT-BAND penalty] max|obs|={mo:.2e} max|reward|={mr:.2e} "
          f"exit_band_blocks={ev['exit_band_blocks']} bracket_closed={ev['bracket_closed']} events={ev}")
    assert mo < ATOL_OBS and mr < ATOL_REW, f"exit-band parity broke: max|obs|={mo:.2e} max|reward|={mr:.2e}"
    assert ev["exit_band_blocks"] > 0, (
        "exit-band penalty fired on NOTHING (exit_band_blocks=0) -- penalty path never exercised, test is vacuous")


def test_portfolio_parity_breach():
    """Strong downtrend + BUY-and-hold -> forces a drawdown BREACH (episode terminates). Parity must
    hold through the breach + penalty."""
    syms = ["EURUSD", "GBPUSD"]
    sym_data = _symbol_data(syms, seed=9, drift=-1.5e-4)       # downward drift
    acts = np.ones(6000, dtype=int)                           # BUY and hold into the drawdown
    mo, mr, ev = _run(syms, continue_after_pass=False, sym_data=sym_data, actions=acts)
    print(f"\n[portfolio downtrend breach] max|obs|={mo:.2e} max|reward|={mr:.2e} events={ev}")
    assert ev["breach"], "expected a drawdown breach on the downtrend"
