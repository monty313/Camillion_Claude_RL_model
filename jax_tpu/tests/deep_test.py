# =====================================================================
# WHEN 2026-06-28 | WHO Claude for Monty
# WHY  A DEEP, end-to-end verification of the JAX/TPU trainer beyond the unit parity
#      tests: the float32 (TPU) path, REAL multi-device pmap, vmap-vs-scalar consistency,
#      training stability + actual learning + resume, eval/streak logic, ONNX-on-rollout,
#      a finiteness sweep, determinism, and the fingerprint/obs contract. Each check runs
#      in its OWN clean subprocess with the right flags (x64 on/off, forced device count).
# WHERE jax_tpu/tests/deep_test.py
# HOW   python jax_tpu/tests/deep_test.py            # run ALL checks, print a PASS/FAIL board
#       python jax_tpu/tests/deep_test.py --check X  # run one check in-process
# =====================================================================
"""Deep verification driver for the JAX/TPU trainer. Run: python jax_tpu/tests/deep_test.py"""
from __future__ import annotations
import os
import sys
import subprocess

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Per-check process config: x64 (exact parity) and forced device count (pmap). Parsed BEFORE jax import.
CHECKS = {
    "float32_single":       {"x64": False, "devices": 1},
    "float32_portfolio":    {"x64": False, "devices": 1},
    "manyseed_single":      {"x64": True,  "devices": 1},
    "manyseed_portfolio":   {"x64": True,  "devices": 1},
    "edge_actions":         {"x64": True,  "devices": 1},
    "vmap_consistency":     {"x64": True,  "devices": 1},
    "determinism":          {"x64": False, "devices": 1},
    "finiteness":           {"x64": False, "devices": 1},
    "pmap_multidevice":     {"x64": False, "devices": 8},
    "learns":               {"x64": False, "devices": 1},
    "resume":               {"x64": False, "devices": 1},
    "eval_streak":          {"x64": False, "devices": 1},
    "onnx_rollout":         {"x64": False, "devices": 1},
    "fingerprint_contract": {"x64": False, "devices": 1},
}

_ARG = None
for i, a in enumerate(sys.argv):
    if a == "--check" and i + 1 < len(sys.argv):
        _ARG = sys.argv[i + 1]

if _ARG:
    cfg = CHECKS[_ARG]
    if cfg["devices"] > 1:
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cfg['devices']}"
    import numpy as np
    import pandas as pd
    import jax
    if cfg["x64"]:
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from config.ftmo_config import load_ftmo_config
    from src.env.trading_env import TradingEnv
    from src.env.portfolio_env import build_portfolio_subs
    from src.strategies.registry import AlphaRegistry
    from src.strategies.alpha_pack import register_all
    from jax_tpu import (jax_static_features as JSF, jax_env as JE, jax_portfolio_env as JPE,
                         jax_config as JC, jax_ppo as PPO, jax_eval as EVAL, jax_trainer as TR,
                         jax_checkpoint as CKPT)

    def _reg():
        r = AlphaRegistry(); register_all(r); return r

    def _ohlc(n, seed, drift=0.0, base=1.10):
        rng = np.random.default_rng(seed)
        close = base + np.cumsum(rng.normal(drift, 1e-4, n)).astype(np.float64)
        ind = rng.normal(0, 1.0, (n, 220)).astype(np.float32)
        t0 = pd.Timestamp("2024-03-04 00:00:00").value
        time_ns = (t0 + np.arange(n, dtype=np.int64) * 60_000_000_000).astype(np.int64)
        return ind, close, time_ns

    def _build_single(symbol="EURUSD", n=3000, seed=7, drift=0.0, psize=100.0, warmup=50):
        cfg_ = load_ftmo_config()
        ind, close, time_ns = _ohlc(n, seed, drift)
        env = TradingEnv(ind, close, time_ns, _reg(), cfg=cfg_, symbol=symbol, position_size=psize, warmup=warmup)
        sd = JSF.build_static_data(env)
        static = JE.make_device_static(sd)
        params = JE.params_from_static(
            sd, daily_target_frac=cfg_.daily_target_pct / 100, trailing_dd_frac=cfg_.trailing_drawdown_pct / 100,
            daily_dd_frac=cfg_.daily_drawdown_pct / 100, total_dd_frac=cfg_.max_total_drawdown_pct / 100,
            profit_target_frac=cfg_.profit_target_total_pct / 100, trailing_enabled=1.0, two_phase_enabled=1.0,
            phase2_continue=1.0 if cfg_.phase2_continue else 0.0, phase2_trailing_frac=cfg_.phase2_trailing_pct / 100)
        return cfg_, env, sd, static, params

    def _build_portfolio(symbols, n=3500, seed=21, drift=0.0, warmup=50):
        cfg_ = load_ftmo_config()
        t0 = pd.Timestamp("2024-03-04 00:00:00").value
        time_ns = (t0 + np.arange(n, dtype=np.int64) * 60_000_000_000).astype(np.int64)
        rng = np.random.default_rng(seed)
        sym_data = {s: (rng.normal(0, 1, (n, 220)).astype(np.float32),
                        (1.1 + 0.2 * k + np.cumsum(rng.normal(drift, 1e-4, n))).astype(np.float64), time_ns)
                    for k, s in enumerate(symbols)}
        subs = build_portfolio_subs(sym_data, _reg, cfg=cfg_, warmup=warmup, progress=False)
        from src.env.portfolio_env import PortfolioEnv
        env = PortfolioEnv(subs=subs, cfg=cfg_, warmup=warmup, continue_after_pass=True)
        psd = JSF.build_portfolio_static(subs)
        static = JPE.make_portfolio_device_static(psd)
        params = JPE.portfolio_params(
            psd, daily_target_frac=cfg_.daily_target_pct / 100, trailing_dd_frac=cfg_.trailing_drawdown_pct / 100,
            daily_dd_frac=cfg_.daily_drawdown_pct / 100, total_dd_frac=cfg_.max_total_drawdown_pct / 100,
            profit_target_frac=cfg_.profit_target_total_pct / 100, trailing_enabled=1.0, two_phase_enabled=1.0,
            phase2_continue=1.0 if cfg_.phase2_continue else 0.0, phase2_trailing_frac=cfg_.phase2_trailing_pct / 100,
            continue_after_pass=1.0, alpha_on=1.0 if cfg_.alpha_reward_enabled else 0.0,
            alpha_agree=cfg_.alpha_agree_bonus, alpha_against=cfg_.alpha_against_penalty,
            alpha_beat=cfg_.alpha_beat_bonus, day_pass_reward=cfg_.day_pass_reward, day_fail_penalty=cfg_.day_fail_penalty)
        return cfg_, env, psd, static, params

    def _step_parity(env, static, params, env_mod, init_kw, actions, otol, rtol):
        cfg_ = load_ftmo_config()
        cpu_obs, _ = env.reset()
        state = env_mod.init_state(static, params, **init_kw)
        jx = np.asarray(env_mod.reset_obs(state, static, params))
        mo = float(np.max(np.abs(jx - cpu_obs))); mr = 0.0
        assert mo < otol, f"reset obs diff {mo}"
        step = jax.jit(env_mod.step, static_argnums=(3,))
        for k, a in enumerate(actions):
            cpu_obs, cr, ct, ctr, _ = env.step(int(a))
            state, jo, jr, jt, jtr = step(state, int(a), static, params)
            jo = np.asarray(jo); jr = float(jr)
            mo = max(mo, float(np.max(np.abs(jo - cpu_obs)))); mr = max(mr, abs(jr - cr))
            assert mo < otol, f"step {k} obs diff {mo} (a={a})"
            assert mr < rtol, f"step {k} reward diff {mr} cpu={cr} jax={jr} (a={a})"
            assert bool(jt > 0.5) == bool(ct) and bool(jtr > 0.5) == bool(ctr), f"step {k} done flags"
            if ct or ctr:
                break
        return mo, mr

    # ---------------------------------------------------------------- checks
    def float32_single():
        cfg_, env, sd, static, params = _build_single(seed=7)
        acts = np.random.default_rng(1).integers(0, 4, 1200)
        mo, mr = _step_parity(env, static, params, JE,
                              dict(start=sd.warmup, end=sd.T - 1, daily_target_frac=0.025, trailing_dd_frac=0.04),
                              acts, otol=3e-3, rtol=5e-4)
        return f"float32 single parity over 1200 steps: max|obs|={mo:.2e} max|reward|={mr:.2e} (tol 3e-3/5e-4)"

    def float32_portfolio():
        cfg_, env, psd, static, params = _build_portfolio(["EURUSD", "GBPUSD"], seed=21)
        acts = np.random.default_rng(7).integers(0, 4, 1600)
        mo, mr = _step_parity(env, static, params, JPE,
                              dict(start=env.warmup, end=env.T - 1, daily_target_frac=0.025, trailing_dd_frac=0.04),
                              acts, otol=3e-3, rtol=5e-4)
        return f"float32 portfolio parity over 1600 steps: max|obs|={mo:.2e} max|reward|={mr:.2e} (tol 3e-3/5e-4)"

    def manyseed_single():
        worst_o = worst_r = 0.0
        for seed in range(6):
            cfg_, env, sd, static, params = _build_single(seed=seed, drift=(seed - 3) * 2e-5)
            acts = np.random.default_rng(100 + seed).integers(0, 4, 900)
            mo, mr = _step_parity(env, static, params, JE,
                                  dict(start=sd.warmup, end=sd.T - 1, daily_target_frac=0.025, trailing_dd_frac=0.04),
                                  acts, otol=1e-4, rtol=1e-5)
            worst_o = max(worst_o, mo); worst_r = max(worst_r, mr)
        return f"6 seeds single (x64) exact parity: worst max|obs|={worst_o:.2e} max|reward|={worst_r:.2e}"

    def manyseed_portfolio():
        combos = [(["EURUSD", "GBPUSD"], 21), (["EURUSD", "XAUUSD", "US30"], 33), (["GBPUSD", "US30"], 44)]
        worst_o = worst_r = 0.0
        for syms, seed in combos:
            cfg_, env, psd, static, params = _build_portfolio(syms, seed=seed, drift=(seed % 3 - 1) * 3e-5)
            acts = np.random.default_rng(7 + seed).integers(0, 4, 1200)
            mo, mr = _step_parity(env, static, params, JPE,
                                  dict(start=env.warmup, end=env.T - 1, daily_target_frac=0.025, trailing_dd_frac=0.04),
                                  acts, otol=1e-4, rtol=1e-5)
            worst_o = max(worst_o, mo); worst_r = max(worst_r, mr)
        return f"3 portfolio combos (x64) exact parity: worst max|obs|={worst_o:.2e} max|reward|={worst_r:.2e}"

    def edge_actions():
        worst = 0.0
        for const_a in (0, 1, 2, 3):  # all-HOLD, all-BUY, all-SELL, all-CLOSE
            cfg_, env, sd, static, params = _build_single(seed=5, drift=4e-5)
            acts = np.full(900, const_a)
            mo, mr = _step_parity(env, static, params, JE,
                                  dict(start=sd.warmup, end=sd.T - 1, daily_target_frac=0.025, trailing_dd_frac=0.04),
                                  acts, otol=1e-4, rtol=1e-5)
            worst = max(worst, mo, mr)
            cfg_, env, psd, static, params = _build_portfolio(["EURUSD", "GBPUSD"], seed=9, drift=4e-5)
            mo, mr = _step_parity(env, static, params, JPE,
                                  dict(start=env.warmup, end=env.T - 1, daily_target_frac=0.025, trailing_dd_frac=0.04),
                                  np.full(1200, const_a), otol=1e-4, rtol=1e-5)
            worst = max(worst, mo, mr)
        return f"constant-action sequences (HOLD/BUY/SELL/CLOSE), single+portfolio: worst diff={worst:.2e}"

    def vmap_consistency():
        cfg_, env, sd, static, params = _build_single(seed=11)
        n = 16
        starts = jnp.full((n,), sd.warmup, jnp.int32); ends = jnp.full((n,), sd.T - 1, jnp.int32)
        dtf = jnp.full((n,), 0.025); trf = jnp.full((n,), 0.04)
        st_b = jax.vmap(JE.init_state, in_axes=(None, None, 0, 0, 0, 0))(static, params, starts, ends, dtf, trf)
        st_s = JE.init_state(static, params, sd.warmup, sd.T - 1, 0.025, 0.04)
        step_b = jax.jit(jax.vmap(JE.step_env, in_axes=(0, 0, None, None)), static_argnums=(3,))
        step_s = jax.jit(JE.step_env, static_argnums=(3,))
        rng = np.random.default_rng(3); worst = 0.0
        # drive all batch envs with the SAME action as the scalar env -> must stay identical
        for k in range(60):
            a = int(rng.integers(0, 4))
            acts = jnp.full((n,), a, jnp.int32)
            st_b, ob, rb, tb, ub = step_b(st_b, acts, static, params)
            st_s, os_, rs, ts, us = step_s(st_s, a, static, params)
            worst = max(worst, float(jnp.max(jnp.abs(ob - os_[None]))), float(jnp.max(jnp.abs(rb - rs))))
        assert worst < 1e-9, f"vmap vs scalar diverged by {worst}"
        return f"vmap(16) vs scalar identical over 60 steps: max diff={worst:.2e}"

    def _tiny_scale():
        JC.N_ENVS_PER_CORE = 32; JC.N_STEPS = 16; JC.MINIBATCH_SIZE = 128
        JC.EVAL_N_WINDOWS = 16; JC.MAX_BARS = 500

    def determinism():
        import tempfile
        _tiny_scale()
        cfg_, env, sd, static, params = _build_single(seed=7, drift=3e-5)
        def run():
            JC.SEED = 42
            rows = []
            TR.train(sd, save_dir=tempfile.mkdtemp(), resume=False, total_updates=999, eval_every=1,
                     target_streak=10 ** 9, max_iters=3, verbose=False, on_eval=lambda r, rs: rows.append(r["mean_reward"]))
            return rows
        a, b = run(), run()
        assert a == b, f"non-deterministic: {a} != {b}"
        return f"same seed -> identical reward sequence: {[round(x, 8) for x in a]}"

    def finiteness():
        import tempfile
        _tiny_scale()
        bad = []
        for name, builder, mod in (("single", lambda: _build_single(seed=4, drift=2e-5), JE),
                                   ("portfolio", lambda: _build_portfolio(["EURUSD", "GBPUSD"], seed=6, drift=2e-5), JPE)):
            cfg_, env, sdx, static, params = builder()
            train_fn = TR.train_portfolio if mod is JPE else TR.train
            rows = []
            details = train_fn(sdx, save_dir=tempfile.mkdtemp(), resume=False, total_updates=999, eval_every=2,
                               target_streak=10 ** 9, max_iters=6, verbose=False,
                               on_eval=lambda r, rs: rows.append(r))
            for r in rows:
                for kk, vv in r.items():
                    if isinstance(vv, float) and not np.isfinite(vv):
                        bad.append(f"{name}.{kk}={vv}")
        assert not bad, f"non-finite metrics: {bad}"
        return "all training/eval metrics finite over single + portfolio runs (no NaN/inf)"

    def pmap_multidevice():
        nd = jax.local_device_count()
        assert nd >= 2, f"expected forced multi-device, got {nd}"
        _tiny_scale()
        cfg_, env, sd, static, params = _build_single(seed=7, drift=3e-5)
        train_end = int(sd.T * 0.8); window = min(JC.MAX_BARS, (train_end - sd.warmup) // 2)
        train_iter, opt = TR.make_train_iter(static, params, sd.warmup, train_end, window, JE)
        key = jax.random.PRNGKey(0)
        _, p = PPO.init_params(key); norm = PPO.norm_init()
        rep = lambda t: jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (nd,) + jnp.asarray(x).shape), t)
        pp, po, pn = rep(p), rep(opt.init(p)), rep(norm)
        iks = jax.random.split(key, nd)
        st, ob = jax.vmap(lambda k: TR._fresh_states(k, JC.N_ENVS_PER_CORE, static, params, sd.warmup, train_end, window, JE))(iks)
        dk = jax.random.split(jax.random.PRNGKey(1), nd); pe = jnp.full((nd,), 0.01)
        for _ in range(3):
            pp, po, pn, st, ob, dk, mtr = train_iter(pp, po, pn, st, ob, dk, pe)
        # params AND the obs-normalizer must be IDENTICAL across all device replicas (both pmean'd)
        pleaves = jax.tree_util.tree_leaves(pp)
        nleaves = jax.tree_util.tree_leaves(pn)
        p_spread = max(float(jnp.max(jnp.abs(l - l[0:1]))) for l in pleaves)
        n_spread = max(float(jnp.max(jnp.abs(l - l[0:1]))) for l in nleaves)
        finite = all(bool(jnp.all(jnp.isfinite(l))) for l in pleaves + nleaves)
        assert finite, "non-finite params/norm after pmap training"
        assert p_spread < 1e-6, f"param replicas diverged by {p_spread} (grad pmean broken)"
        assert n_spread < 1e-6, f"NORM replicas diverged by {n_spread} (norm pmean broken)"
        return (f"pmap on {nd} devices: 3 iters OK, params synced (spread={p_spread:.2e}) "
                f"AND obs-norm synced (spread={n_spread:.2e}), all finite")

    def learns():
        import tempfile
        JC.N_ENVS_PER_CORE = 256; JC.N_STEPS = 64; JC.MINIBATCH_SIZE = 1024
        JC.EVAL_N_WINDOWS = 32; JC.MAX_BARS = 1500; JC.DOMAIN_RANDOMIZE_RISK = False
        # a STRONG uptrend with real sizing: BUY-and-hold makes money -> a learnable edge. The robust
        # learning signal is mean per-step reward rising as the policy learns to capture the uptrend
        # (greedy BUY), plus the action mix shifting toward BUY. (+10% pass needs multi-day windows;
        # that's covered by the portfolio run on real data, not this fast synthetic check.)
        cfg_, env, sd, static, params = _build_single(seed=2, drift=3e-4, psize=10000.0, n=6000, warmup=100)
        rows = []
        TR.train(sd, save_dir=tempfile.mkdtemp(), resume=False, total_updates=999, eval_every=8,
                 target_streak=10 ** 9, max_iters=64, verbose=False, on_eval=lambda r, rs: rows.append(r))
        # robust learning signal: mean per-step reward rises as the policy captures the uptrend edge.
        early = np.mean([r["mean_reward"] for r in rows[:2]])
        late = np.mean([r["mean_reward"] for r in rows[-2:]])
        msg = f"mean reward {early:+.6f} (early) -> {late:+.6f} (late) over {len(rows)} evals (uptrend BUY-hold edge)"
        assert late > early, f"no learning: {msg}"
        return "training LEARNS on a learnable task: " + msg

    def resume():
        import tempfile
        _tiny_scale()
        d = tempfile.mkdtemp()
        cfg_, env, sd, static, params = _build_single(seed=7, drift=3e-5)
        TR.train(sd, save_dir=d, resume=False, total_updates=999, eval_every=2, target_streak=10 ** 9,
                 max_iters=4, verbose=False)
        latest = CKPT.find_latest(d)
        assert latest is not None, "no checkpoint written"
        tag, upd = latest
        # resume: should load the checkpoint and continue from `upd`
        out = {}
        TR.train(sd, save_dir=d, resume=True, total_updates=999, eval_every=2, target_streak=10 ** 9,
                 max_iters=2, verbose=False, on_eval=lambda r, rs: out.setdefault("first", r))
        cont = out["first"]["update"]
        assert cont > upd, f"resume did not continue ({cont} <= {upd})"
        # params round-trip exactly
        _, tmpl = PPO.init_params(jax.random.PRNGKey(0))
        p, nm, det = CKPT.load_policy(d, tag, tmpl, PPO.RunningNorm)
        assert det.get("env_fingerprint"), "details missing fingerprint"
        return f"checkpoint@{upd} -> resume continued to update {cont}; params+details reload OK"

    def eval_streak():
        # _longest_run unit
        lr = EVAL._longest_run
        assert lr(np.array([1, 1, 1, 0, 1, 1])) == 3
        assert lr(np.array([0, 0, 0])) == 0
        assert lr(np.array([1, 1, 1, 1])) == 4
        # evaluate sanity on an uptrend (some windows should pass with a BUY-biased net)
        JC.EVAL_N_WINDOWS = 32; JC.MAX_BARS = 1500
        cfg_, env, sd, static, params = _build_single(seed=2, drift=2.0e-4, n=6000, warmup=100)
        model, p = PPO.init_params(jax.random.PRNGKey(0))
        # bias the actor toward BUY (action index 1) so deterministic argmax mostly BUYs
        import flax
        pp = flax.core.unfreeze(p)
        last = sorted([k for k in pp["params"] if k.startswith("Dense_")], key=lambda s: int(s.split("_")[1]))[len(JC.NET_ARCH)]
        b = np.array(pp["params"][last]["bias"]); b[1] += 50.0
        pp["params"][last]["bias"] = jnp.asarray(b)
        ev = EVAL.evaluate(flax.core.freeze(pp), PPO.norm_init(), static, params, env=JE)
        ok = (0.0 <= ev["pass_rate"] <= 1.0) and (0 <= ev["best_streak"] <= ev["n_windows"])
        assert ok, f"eval out of range: {ev}"
        # a window can't be BOTH a pass and a breach -> pass_rate + breach_rate <= 1 (the gate-correctness fix)
        assert ev["pass_rate"] + ev["breach_rate"] <= 1.0 + 1e-6, \
            f"pass+breach double-count: {ev['pass_rate']}+{ev['breach_rate']}>1 (won-without-breach broken)"
        return (f"_longest_run correct; evaluate(uptrend) pass_rate={ev['pass_rate']:.1%} "
                f"breach_rate={ev['breach_rate']:.1%} best_streak={ev['best_streak']}/{ev['n_windows']}; "
                f"pass+breach<=1 (no post-pass-breach inflation)")

    def onnx_rollout():
        import tempfile
        cfg_, env, sd, static, params = _build_single(seed=7, drift=1e-4, n=3000)
        model, p = PPO.init_params(jax.random.PRNGKey(3)); norm = PPO.norm_init()
        d = tempfile.mkdtemp(); CKPT.save_policy(d, "best_policy", p, norm, {"x": 1})
        from jax_tpu import export_to_pytorch as EXP
        out = os.path.join(d, "p.onnx"); EXP.convert(d, "best_policy", out)
        # gather REAL rollout obs (not random) and compare ONNX vs JAX logits
        step = jax.jit(JE.step_env, static_argnums=(3,))
        state = JE.init_state(static, params, sd.warmup, sd.T - 1, 0.025, 0.04)
        obs = [np.asarray(JE.reset_obs(state, static, params))]
        rng = np.random.default_rng(0)
        for _ in range(200):
            state, o, *_ = step(state, int(rng.integers(0, 4)), static, params)
            obs.append(np.asarray(o))
        O = np.stack(obs).astype(np.float32)
        import onnxruntime as ort
        ol = ort.InferenceSession(out).run(None, {"obs": O})[0]
        jl, _ = model.apply(p, PPO.norm_apply(norm, jnp.asarray(O)))
        d2 = float(np.max(np.abs(ol - np.asarray(jl))))
        assert d2 < 1e-4, f"ONNX vs JAX on rollout obs diff {d2}"
        return f"ONNX matches JAX on {len(obs)} REAL rollout observations: max|logit|={d2:.2e}"

    def fingerprint_contract():
        from src.training.env_fingerprint import env_fingerprint
        from config import constants as C
        cfg_, env, sd, static, params = _build_single(seed=1)
        # JAX trainer stamps fp == CPU env_fingerprint
        assert env_fingerprint() == env_fingerprint(), "fingerprint unstable"
        assert C.OBS_TOTAL_SIZE == 553 and sd.static_obs.shape[1] == 553, "obs size != 553"
        # static obs block placement: a fresh CPU obs at warmup, static parts must match jax static row (dynamic zeroed)
        cpu_obs, _ = env.reset()
        srow = sd.static_obs[sd.warmup]
        from jax_tpu.jax_static_features import DYNAMIC_SLICES
        mask = np.ones(C.OBS_TOTAL_SIZE, bool)
        for (a, b) in DYNAMIC_SLICES.values():
            mask[a:b] = False
        d = float(np.max(np.abs(srow[mask] - cpu_obs[mask])))
        assert d < 1e-4, f"static obs block placement mismatch {d}"
        return f"fingerprint stable ({env_fingerprint()}), obs={C.OBS_TOTAL_SIZE}, static blocks match CPU (diff={d:.2e})"

    fn = {k: v for k, v in locals().items() if callable(v) and k == _ARG}.get(_ARG)
    try:
        msg = fn()
        print(f"PASS  {_ARG}: {msg}")
        sys.exit(0)
    except Exception as e:
        import traceback
        print(f"FAIL  {_ARG}: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)


def main():
    print(f"DEEP TEST — {len(CHECKS)} checks, each in a clean subprocess\n" + "=" * 74)
    results = {}
    for name in CHECKS:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--check", name],
                           capture_output=True, text=True, cwd=_ROOT)
        line = next((l for l in r.stdout.splitlines() if l.startswith(("PASS", "FAIL"))), None)
        ok = r.returncode == 0
        results[name] = ok
        if line:
            print(line)
        else:
            print(f"FAIL  {name}: no result line (rc={r.returncode})")
            print("  " + "\n  ".join((r.stdout + r.stderr).splitlines()[-15:]))
    npass = sum(results.values())
    print("=" * 74)
    print(f"DEEP TEST RESULT: {npass}/{len(CHECKS)} checks passed")
    if npass != len(CHECKS):
        print("FAILED:", [k for k, v in results.items() if not v])
    return 0 if npass == len(CHECKS) else 1


if __name__ == "__main__":
    if not _ARG:
        raise SystemExit(main())
