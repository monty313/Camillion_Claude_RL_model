# Camillion JAX/TPU Trainer — THE PLAN (grounded in the actual repo)

> **Everything JAX/TPU lives in this one folder (`jax_tpu/`)** so it never gets confused with the
> CPU code. The CPU trainer (`src/training/trainer.py` + `src/env/*`) is the **reference / source of
> truth** and is **never changed**. This is a *second implementation of the same environment* that
> must match the CPU env **bar-for-bar** (same 479-obs, same reward, same FTMO rules, same
> `env_fingerprint`). It exists to run **thousands of trading lifetimes in parallel on a TPU** and
> report the **probability of passing** the FTMO challenge per (daily-target, risk) setting.

---

## 0. The key architectural decision (why this is tractable AND high-parity)

The 479-float observation splits cleanly into two groups (verified by reading the CPU env):

| group | blocks | size | depends on | how JAX gets it |
|---|---|---|---|---|
| **STATIC** (per-bar) | indicators, alpha_values, alpha_mask, alpha_summary, signal_memory, signal_accuracy, time, cross_asset, alpha_streak | **439** | only the bar index | **precompute ONCE on host** (the CPU env already does this), pack into a `(T, 439)` device tensor, **share read-only across all envs** |
| **DYNAMIC** | account_daily, account_episode, portfolio, sizing, recent_context | **40** | the evolving account/position state | **recomputed every JAX step** from `EnvState` + a few per-bar host scalars (`jnp` port of `win_loss_features`) |

This is exactly the **"build once + share the precomputed table across envs"** design already written
into `config/constants.py` (the MAX_STRATEGIES scaling note) and the project memory. Consequences:

- **The static 439 floats are byte-identical to the CPU env** — they are literally the same arrays
  (`sub.ind[i]`, `sub.alpha_matrix[i]`, `summarize(...)`, `sub.sig_acc[i]`, `sub.time_feats[i]`,
  `sub.cross_asset_matrix[i]`, `min(streak,50)/50`, …). Zero parity risk on 439/479 of the obs.
- **We do NOT need TA-Lib / pandas / indicators inside the JAX hot loop** (CLAUDE.md rule #3 holds).
  The JAX step only does: index the static row → evolve account state → compute 40 dynamic floats →
  reward + FTMO. All branchless `jnp`.
- **Memory at scale is solved**: the `(T, 439)` tensor is shared once; each of the thousands of envs
  carries only a tiny `EnvState` (~35 scalars). The alpha/streak tables can be `int8` later.

`jax_indicators.py` (indicators rewritten in `jnp`) is included as a **parity-tested OPTIONAL** module
for the future "fully on-device, generate-data-on-the-fly" ideal — it is **not** on the critical path
to a working TPU trainer (the static tensor is).

---

## 1. The non-negotiable invariants (copied from the blueprint, enforced here)

1. **Same observation contract** — 479 float32, `v1.5.0`, block order from `config/constants.py`
   (`OBS_BLOCK_ORDER`). Real constant names are `OBS_TOTAL_SIZE` (=479) and
   `OBSERVATION_CONTRACT_VERSION` (="v1.5.0") — *not* `OBS_SIZE`/`OBS_VERSION` (blueprint was wrong;
   we use the real names, no aliases added).
2. **Same FTMO numbers** — 2.5%/day target, 5% daily DD, 10% total DD, 4% phase-1 trailing wall,
   two-phase (+2.5% → bank → 1% trail), +10% pass. All read from `config/ftmo_config.py` at the call
   site (so they stay runtime-tunable).
3. **Same alpha-vs-action separation** — `alpha=0` (no setup) ≠ `ACTION_HOLD` ≠ empty slot (mask 0).
4. **Same fingerprint** — `env_fingerprint()` is **config-derived** (contract version, obs_total,
   asset classes, sorted alpha names, 9 FTMO keys, 4 reward knobs). JAX reading the same config →
   the hash matches **automatically**. The real work is **step-parity**, not the hash.
5. **Step-parity test is the gate** — step the CPU `TradingEnv` and the JAX env on identical bars +
   identical actions; assert obs match (atol 1e-4) and reward matches (atol 1e-5). Nothing past the
   gate ships until this is green.
6. **Same policy interface** — 479 obs in, `Discrete(4)` out, 3×256 tanh MLP → a JAX policy is ranked
   head-to-head with CPU policies by pass-rate, and can be exported to the same ONNX format for MT5.

---

## 2. Exact numbers the JAX code must reproduce (from reading the repo)

**Observation block layout (479):**
```
indicators       [  0:220]   alpha_values  [220:284]   alpha_mask    [284:348]
alpha_summary    [348:352]   signal_memory [352:357]   signal_accur. [357:359]
account_daily    [359:366]   account_epis. [366:373]   time          [373:379]
portfolio        [379:387]   alpha_streak  [387:451]   sizing        [451:461]
cross_asset      [461:471]   recent_context[471:479]
```
Builder = ordered concat → `np.nan_to_num(nan=0,posinf=0,neginf=0)` → validate. **No per-block scaling.**

**FTMO rules** (`src/risk/ftmo_rules.py`, fractions of starting balance unless noted):
- `daily_target_hit`: `(equity - day_start_balance) >= starting_balance * 2.5%`
- `daily_drawdown_breached`: `(day_start_balance - equity) >= day_start_balance * 5%`
- `total_drawdown_breached`: `(starting_balance - equity) >= starting_balance * 10%`
- `trailing_breached` (enabled): `(episode_peak_equity - equity) >= episode_peak_equity * 4%`
- `breached = daily OR total OR trailing`; `should_auto_flat = two_phase AND daily_target_hit`

**Account update on close** (`trade_history.record_close`): `balance += pnl; daily_realized += pnl;
episode_realized += pnl; mark_equity(balance)` (→ equity=balance, bump day/episode peaks);
`trades++`; win→`wins++, consec_losses=0`; loss→`losses++, consec_losses++`.

**Single-symbol step** (`trading_env.step`, the parity reference):
1. act at `close[t]`: on position change, realize `pos*(close[t]-entry)*size - cost_frac*close[t]*size`
   via `record_close`; open → set entry, charge entry cost `cost_frac*close[t]*size` to balance &
   daily/episode realized.
2. advance to `close[t+1]`, mark-to-market: `equity = balance + pos*(close[t+1]-entry)*size`.
3. `reward = (equity - equity_before)/starting_balance * reward_scale`.
4. day boundary (`_dates[t+1] != cur_date`): pay NY index bonus if the ending day passed; `reset_day`;
   reset phase2 + NY state; `days_elapsed++`.
5. breach → `terminated`, `reward -= breach_penalty (1.0)`; else equity ≥ start*1.10 → pass,
   `reward += pass_bonus (1.0)`.
6. two-phase: `should_auto_flat` → flatten+bank; `phase2_continue=True` → 1% trail from banked peak.
- `cost_frac = 0.000035` (per side), `daily_target_pct=2.5`, `breach_penalty=pass_bonus=1.0`.
- **Day rollover uses precomputed `_dates` (pandas-normalized timestamps), NOT bar-index modulo** →
  JAX uses a precomputed `is_new_day[t]` boolean array (host) — see §0/§4.

**PPO hyperparameters (match these exactly)** — `src/training/trainer.py`:
`gamma=0.997, gae_lambda=0.97, clip=0.2, ent_coef 0.01→0 (linear anneal over training), vf_coef=0.5,
lr=3e-4, max_grad_norm=0.5, n_epochs=10, n_steps=2048(ref), batch_size=256(ref), net_arch=[256,256,256]
tanh`, **`VecNormalize(norm_obs=True, norm_reward=False, clip_obs=10.0)`** (running obs mean/std; the
JAX PPO replicates this with an online normalizer, frozen at eval).

---

## 3. Folder layout (everything in `jax_tpu/`)

```
jax_tpu/
  PLAN.md                 ← this file (the plan + the exact numbers)
  README.md               ← how to run it (Colab + local parity)
  __init__.py
  jax_config.py           ← hyperparams (mirror CPU) + SCALE knobs for 70–80% TPU utilization
  jax_ftmo.py             ← branchless FTMO breach + two-phase banking (jnp / lax.select)
  jax_obs_blocks.py       ← the 40 DYNAMIC obs floats in jnp (port of win_loss_features)
  jax_static_features.py  ← HOST: build/cache the (T,439) static obs tensor + per-bar scalars
  jax_env.py              ← EnvState pytree + branchless step_env (indexes the static tensor)
  jax_indicators.py       ← OPTIONAL on-device indicators in jnp (parity-tested vs CPU funcs)
  jax_ppo.py              ← Flax 3×256 policy/value + GAE + clipped PPO loss + online obs-norm
  jax_trainer.py          ← vmap+pmap training loop, scaled for 70–80% TPU + domain-randomized risk
  jax_eval.py             ← held-out walk-forward PASS-RATE (= P(pass) readout)
  jax_checkpoint.py       ← orbax save/load + checkpoint_meta.json (shape/version guard)
  export_to_pytorch.py    ← JAX/Flax params → PyTorch 3×256 → ONNX (MT5 path)
  tests/
    __init__.py
    conftest.py           ← repo-root on sys.path; skip cleanly if jax missing
    test_jax_parity.py    ← THE GATE: ftmo + dynamic-blocks + STEP parity (CPU vs JAX)
    run_parity.py         ← stdlib runner (works without pytest): python jax_tpu/tests/run_parity.py
  notebooks/
    Camillion_JAX_TPU_Train.ipynb   ← Colab TPU notebook (mirrors run_training.py data path)
```
`pyproject.toml testpaths` gets `"jax_tpu"` appended (1-line, documented) so `pytest -q` collects the
parity test too (it self-skips when JAX isn't installed).

---

## 4. Build order (strict; each step is verified locally with `jax[cpu]` before moving on)

0. **Scaffold** the folder + `jax_config.py` + `README.md` (this commit).
1. **Install `jax[cpu] flax optax` locally** so every parity step is actually *run*, not just written.
2. **`jax_ftmo.py`** → parity vs `ftmo_rules`/`breach_detector` on a grid of account states. **RUN.**
3. **`jax_obs_blocks.py`** (5 dynamic blocks) → parity vs `win_loss_features` on random states. **RUN.**
4. **`jax_static_features.py`** — host builder: from a `TradingEnv` (or feature_cache), emit the
   `(T,439)` static tensor + per-bar scalar arrays (`close, is_new_day, minute_of_day, ref_move,
   week_avg, prev_day, prev2, today_sofar`) + per-symbol scalars (`typical_range, value_per_point,
   position_size, is_index`). The 439 columns are assembled in the **same order** the CPU `_obs()` uses.
5. **`jax_env.py`** — `EnvState` NamedTuple (pytree) + `step_env` (branchless, indexes the static
   tensor, evolves account, calls `jax_ftmo` + `jax_obs_blocks`). Include NY-index-bonus state for full
   parity (fires only on indices). → **STEP-PARITY GATE** vs CPU `TradingEnv` on synthetic bars +
   scripted actions. **RUN — must be green before §6.**
6. **`jax_indicators.py`** + parity vs CPU indicator funcs (run locally; pandas path, no TA-Lib). **RUN.**
7. **`jax_ppo.py`** — Flax `CamillionPolicy` (3×256 tanh, actor 4-logits + critic), GAE
   (`gamma=0.997,λ=0.97`), clipped loss (`clip=0.2,vf=0.5,ent anneal`), online obs-normalizer
   (`clip_obs=10`), `optax.chain(clip_by_global_norm(0.5), adam(3e-4))`.
8. **`jax_trainer.py`** — `vmap` envs → `pmap` across 8 TPU cores; `lax.scan` rollout; domain-randomize
   `(daily_target, trailing_dd)` per env; **scale to 70–80% TPU utilization** (see §5); fingerprint
   logged to the ledger with `trainer="jax-tpu"`.
9. **`jax_eval.py`** — deterministic held-out walk-forward; `pass_rate = passes/M`; report a
   `(target, risk)` grid + CI; the **P(pass)** deliverable.
10. **`jax_checkpoint.py`** (orbax + meta guard) and **`export_to_pytorch.py`** (→ ONNX for MT5).
11. **`notebooks/Camillion_JAX_TPU_Train.ipynb`** — TPU runtime check → install → clone → build
    data_cache via the **existing `run_training.prepare_caches`** → build static tensor → train → eval.
12. **Docs** — `docs/UPDATE_LOG.md` IRAC entry, this folder's `README.md`, pointer from
    `docs/JAX_GPU_TPU_TRAINER_BLUEPRINT.md` to "now partially built in `jax_tpu/`".

---

## 5. Hitting 70–80% TPU utilization (operator requirement — "even if we do more work")

TPU v2-8 is fed by making the **per-update batch huge** so the matrix engine is saturated evaluating
the small MLP for a massive army (blueprint Rule 5). Knobs in `jax_config.py`:

- **`pmap` across all 8 cores** + **`vmap`** within each core. Effective batch =
  `N_ENVS_PER_CORE × 8 × N_STEPS`. Default target **N_ENVS_PER_CORE=2048, N_STEPS=128** →
  `2048×8×128 ≈ 2.1M states / update` (raise until HBM ~80% or step time plateaus).
- **bf16 compute, fp32 for money**: matmuls in bf16 (TPU-native), but equity/reward/FTMO kept fp32
  (`jax_default_matmul_precision="float32"` for the env math) — sub-0.002% equity moves must not
  vanish (blueprint C3).
- **`donate_argnums` + `lax.scan`** for the rollout so there's zero host↔device traffic in the loop.
- **An autoscale + utilization probe** in the notebook: print HBM used / device, step throughput, and a
  suggested `N_ENVS_PER_CORE` to push utilization into the 70–80% band. We deliberately oversize the
  env army (more parallel lifetimes) rather than shrink the net — extra envs = better gradient + higher
  pass-rate signal, which is "more work" spent usefully.
- Static shapes everywhere (fixed `MAX_BARS` episodes + masking) so XLA never recompiles mid-run.

**Honest note:** actual utilization can only be *measured on the TPU*. The notebook prints the numbers
and the tuning knob; defaults are sized to land in-band on v2-8, and the probe tells you which way to
nudge `N_ENVS_PER_CORE`.

---

## 5.5 Stop condition + progress saving (operator requirement)

**Train until 40 consecutive challenge PASSES on held-out data**, then stop. Definitions:

- Every `EVAL_EVERY` updates, `jax_eval.py` runs a stream of **held-out walk-forward challenge
  windows** (unseen data). Each window **passes** iff it reaches +10% without breaching daily/total/
  trailing DD (same rule as `walk_forward.py`).
- A running **consecutive-pass streak** is tracked across those windows. Stop the moment the streak
  reaches **`TARGET_CONSECUTIVE_PASSES = 40`** (config knob). One failed window resets the streak to 0
  (consistency, not a lucky spike). This is the FTMO-style "40 challenges in a row" the operator asked
  for, and it's the same idea as the existing "4 won days in a row" bonus, scaled up.
- **Why held-out:** a high train streak with a low held-out streak = overfitting → keep training. We
  only count the streak on **unseen** windows so 40-in-a-row means real generalization.

**Progress + policies saved to Colab/Drive the whole way** (`jax_checkpoint.py` + the trainer):
- `SAVE_DIR = /content/drive/MyDrive/Camillion/jax_models` (created in the notebook).
- On **every eval**: write `jax_ckpt_<update>/` (orbax params) + `checkpoint_meta.json`
  (obs_size, contract_version, env_fingerprint, N_ENVS, update, streak, pass_rate, daily_target, risk).
- Append one line per eval to **`SAVE_DIR/jax_progress.jsonl`** (the run ledger: update, timesteps,
  mean_reward, pass_rate, consecutive_passes, best_streak, HBM%, throughput) so progress is *documented*
  and resumable. Also mirror a row into `docs/TRAINING_LEDGER.md`-style record where available.
- Keep a rolling **`SAVE_DIR/best_policy/`** = the checkpoint with the longest held-out streak so far,
  with its details JSON, so the "current best" is always on Drive even if Colab disconnects.
- **Resume**: `train(resume=True)` reads `latest_step.txt`/`jax_progress.jsonl`, restores params +
  streak, and continues toward 40 — so a disconnect never loses progress.
- At 40-in-a-row: save `SAVE_DIR/passed_40_in_a_row/` (final policy + full details + the window-by-window
  pass log) and stop.

---

## 6. Build status (everything below is DONE — nothing silently dropped)

- ✅ **PortfolioEnv parity** (shared-pot, symbol-cycling, alpha-shaping, midnight day-scoring,
  4-in-a-row bonus, pot-level two-phase banking): built in `jax_portfolio_env.py` and verified
  bar-for-bar against the CPU `PortfolioEnv` (`tests/test_jax_portfolio_parity.py`), including the
  forced banking / won-day / breach branches. Train it with `jax_trainer.train_portfolio(...)` — the
  user's core goal. The single-symbol env was the foundation; both share one parity-tested interface.
- ✅ **Live FTMO-consistency progress** (`jax_progress.py`): every eval streams `P(pass) @ 2.5%/4%`,
  held-out return, breach rate, and the `N/40 in a row` bar; the Colab `LiveDashboard` redraws live.
- ✅ **On-device indicator generation** (`jax_indicators.py`) — built & parity-tested; the trainer
  uses the precomputed static tensor by default (faster + exact). Available for a future fully-on-device run.
- ⏭️ **Domain-randomized risk eval grid** — `jax_eval.evaluate` already takes `(daily_target, trailing_dd)`;
  the full grid sweep (P(pass) as a function of target/risk) is a one-loop notebook run over `evaluate`.
