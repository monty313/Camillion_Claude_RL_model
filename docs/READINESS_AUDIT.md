# READINESS AUDIT — Camillion (pre-alpha correctness audit)

**Date:** 2026-06-21 · **Scope:** prove the PPO/MLP + workflow are correct, learnable,
leak-free, and ready to add alphas. **No new features added.** **Not pushed (per request).**

> **Sandbox limitation (honest disclosure):** the build environment has **no PyTorch /
> Stable-Baselines3 and no PyPI access**. Checks that require *running* a neural net
> (`check_env`, single-batch overfit, PPO-update sanity, save/load) are marked
> **SKIP — run in Colab**, with the exact runnable test shipped under `tests/`. Everything
> that runs on numpy/pandas is proven here **with real numbers**.

---

## ⚠️ FINDINGS FIRST (mentor flags — read these)

These do **not** block *adding alphas*, but two of them **must be addressed before a serious
training run**. None of them is a hidden bug in the plumbing — the plumbing is correct.

**F1 — HIGH · The observation is raw and unnormalized.** Measured obs range over 5,000 steps
was **[-413.77, +381.86]**, and the gym space is `Box(-inf, +inf, (367,), float32)`. Feature
scales span orders of magnitude (US30 ~46,000 vs RSI 0–100 vs alphas ±1 vs account fractions
0–1). An MLP/PPO learns poorly when inputs aren't on comparable scales, and SB3's `check_env`
will warn about the unbounded space. **This was your deliberate "do not normalize the
indicators" choice — and it's correct for the *cache*.** The fix does **not** touch the 367
contract or the cached raw values: normalize what the *policy* sees at train time with
`VecNormalize` (a training wrapper). The shipped `tests/test_single_batch_overfit.py` already
wraps the env in `VecNormalize` to demonstrate this. **Recommendation: train with
`VecNormalize(norm_obs=True)`. Flagging, not changing, per your rules.**

**F2 — RESOLVED · Reward scale depends on `position_size` — default is now realistic.**

**Historical note (original audit finding):**
The original audit measured reward std over a random rollout at `position_size=1.0`:
- `position_size=1.0`: **std = 5.38e-7** — a near-dead signal (unusable for learning).
- `position_size=100000` (realistic ~1 lot notional): **std = 0.424** — healthy.

**Current verified state (2026-06-21):**
The `TradingEnv` default is now `position_size=100000.0`. This was verified by inspecting the
actual runtime default via `inspect.signature(TradingEnv.__init__)`. The single-symbol training
notebook also sets `POSITION_SIZE=100000.0` explicitly. No action required — the baseline is
already configured with a realistic notional.

**Tests using `position_size=1.0`:** Some tests intentionally use 1.0 for fast verification of
specific behaviors (reward independence, gate logic), not for training. This is intentional and
correct.

**F3 — LOW · `env.step()` ≈ 310µs (~3,300 steps/s single env).** Fine for CPU PPO with
vectorised envs (×8 ≈ 26k/s); the cost is the per-step observation assembly, optimizable
later if needed. Not a blocker.

**F4 — NOTE · `breach_penalty` (1.0) dominates the per-step reward (~±0.004 at 100k notional).**
Intentional (a breach is catastrophic), but tune the ratio deliberately when you set scale.

---

## SECTION 1 — ENVIRONMENT CORRECTNESS
- **1.2/1.3/1.4 PASS** — `reset()` → `(obs, info)`; `step()` → `(ndarray, float, bool, bool, dict)`;
  obs `dtype=float32`, `shape=(367,)`, finite. Action space `{HOLD:0, BUY:1, SELL:2, CLOSE:3}`.
- **1.5 PASS** — 5,000 random steps: **0 non-finite events**, obs range [-413.77, +381.86],
  2 truncations (window end), 0 spurious terminations. Test: `tests/test_env_5k_no_nan.py`.
- **1.1 SKIP (Colab)** — `stable_baselines3.common.env_checker.check_env`. Test shipped:
  `tests/test_env_checker.py` (expect a *warning* about the unbounded obs space → see F1).

## SECTION 2 — OBSERVATION INTEGRITY
- **2.6 PASS** — block map sums to **367** (v1.1.0):
  `indicators[0:200] alpha_values[200:264] alpha_mask[264:328] alpha_summary[328:332]
  signal_memory[332:337] signal_accuracy[337:339] account_daily[339:346]
  account_episode[346:353] time[353:359] portfolio[359:367]`.
- **2.7 PASS** — zero alphas → `alpha_values` and `alpha_mask` all 0; env runs clean (alphas optional).
- **2.8 PASS** — `ALPHA_BUY=+1, ALPHA_SELL=-1, ALPHA_INACTIVE=0`, empty slot = mask 0; `ACTION_HOLD=0`
  lives in a separate action space (see `CLAUDE.md` "Alpha-state 0 vs action HOLD").
- **2.9 PASS** — 3 alphas: mask sum 3; summary `[buy%, sell%, active%, net%]` mathematically correct
  (e.g., one active sell → `[0, 1, 0.333, -1]`).
- **2.10 PASS** — 64 alphas: shape still `(367,)`, mask sum 64, summary computes (`[0.512,0.488,0.672,0.023]`).
- **2.11 PASS** — signal memory shifts correctly (lag0→lag1 each step), e.g. net `[…,1,-1,0.3]`:
  `t=3 → [-1,1,-0.5,0.2,0]`, `t=4 → [0.3,-1,1,-0.5,0.2]`.
- **2.12 PASS** — multi-timeframe leakage: corrupting **all** bars > t=1500 leaves aligned 5m/30m/4h/1d
  values at `[:1501]` byte-identical (last-closed-bar rule). Test: `tests/test_cache_no_leakage.py`.

## SECTION 3 — REWARD CORRECTNESS
- **3.13 PASS** — formula (verbatim from `env.step`):
  `reward = (self.acc.equity - equity_before) / self.cfg.starting_balance`; then
  `reward -= self.breach_penalty` on a breach. No alpha/accuracy/signal terms.
- **3.14 PASS** — `tests/test_trading_env.py::test_reward_independent_of_alphas`: identical prices +
  identical actions, different alphas → byte-identical rewards.
- **3.15 PASS** — reward std ≈ 0.42 at default position_size=100000 (realistic).
- **3.16 PASS** — grep of the reward path: no `alpha`/`accuracy`/`signal`/`reliab` terms.
- **3.17 PASS** — FTMO: +$2,600 → daily 2.60% → `target_hit=True, auto_flat=True` (two-phase).
  FREE: same → `target_hit=True, auto_flat=False` (no two-phase). Breach: equity 95,800 from peak
  100,000 (-4.2%) vs 4% wall → `breached=True ['trailing_drawdown']`. Day reset: daily_pnl 500 → 0.

## SECTION 4 — PPO/MLP LEARNS (sanity)
- **4.18 PASS (structural)** — policy input dim = `observation_space` = **367**; `MlpPolicy`
  `net_arch=[256,256,256]`, `gamma=0.997, gae_lambda=0.97, ent_coef=0.0, lr=3e-4`.
- **4.22 PASS** — introspector is read-only: **no `import torch`, no `.backward(`, no `optimizer`**
  in `policy_introspector.py` (the word "torch" appears only in a docstring note). It calls the
  policy forward-only via block ablation.
- **4.19 / 4.20 / 4.21 SKIP (Colab)** — single-batch overfit, PPO-update finiteness (loss/KL/entropy),
  and critic `value_loss` movement need torch. Shipped: `tests/test_single_batch_overfit.py` (uses
  `VecNormalize` per F1). **Until this passes in Colab, "the network can learn" is by-design, not yet proven.**

## SECTION 5 — REPRODUCIBILITY & SPEED
- **5.23 PASS** — same seed + same actions → identical obs and rewards. Test: `tests/test_seed_reproducibility.py`.
  (Full torch-level determinism is a Colab concern.)
- **5.24 PASS** — `env.step()` source contains no pandas/TA-Lib; **~310µs/step (~3,300/s)**; indicators
  are precomputed once into the cache and only read in the loop.
- **5.25 PASS** — `env.step()` µs: 0 alphas 309.7 · 3 alphas 304.1 · 64 alphas 316.9 → **64 vs 0 = 1.02×**
  (< 5×). Step time is alpha-count-independent because alphas are precomputed.

## SECTION 6 — DIAGNOSTICS & END-TO-END
- **6.26 PASS** — cache → 0/3 alphas → read-only eval harness → Policy Doctor + Barbershop ran with no errors.
- **6.27 PASS (mock) / SKIP-real (Colab)** — on a mock policy that depends on the alpha block, the block
  saliency correctly attributed **alpha = 0.82**, leader-chasing flag computed. Real-trained-model run is the Colab cell.
- **6.28 PASS** — scoreboard, day_replay, trade_autopsy, risk_doctor, signal_doctor, feature_doctor all produce output.
- **6.29 PASS** — walk-forward ran (3 windows, pass-rate produced; 0.0 here because the policy is a *mock*).

## SECTION 7 — READY TO ADD ALPHAS
- **7.30/7.32 PASS** — obs shape stays `(367,)` at 1, 5, and 64 alphas; mask sums 1/5/64; STATE_DIM unchanged.
- **7.31 PASS** — `tests/test_add_one_alpha_shape.py`: register one alpha → shape (367,), slot+mask filled, summary recomputes.
- **How you add an alpha (no retrain):** subclass `BaseStrategy` in `src/strategies/examples/`, implement
  `compute_signal(ctx) -> +1/-1/0`, then `registry.register(MyStrategy())` — it lands in the next free
  fixed slot. The 367 contract never changes, so old models keep working.

## SECTION 8 — SAVE / LOAD / RESUME WITH NEW ALPHAS
- **8.33–8.36 PASS (by construction) + Colab round-trip** — because the alpha slots are **fixed at 64** and
  unassigned slots = 0, a model trained with N alphas consumes the same **367**-dim observation as one with M
  alphas (the extra slots are simply 0). There is **no shape mismatch by construction** when you add alphas and
  load an old model. The literal SB3 `model.save`/`PPO.load` round-trip should be confirmed once in Colab.

---

## SUMMARY TABLE (1–36)

| # | Check | Result | One-line proof / reason |
|---|-------|--------|--------------------------|
| 1 | check_env | SKIP | needs SB3 → `tests/test_env_checker.py` (Colab); expect unbounded-obs warning (F1) |
| 2 | obs dtype/bounds/no-NaN | PASS | float32 (367,), 5k steps 0 non-finite |
| 3 | action space | PASS | {HOLD,BUY,SELL,CLOSE}={0,1,2,3} |
| 4 | reset/step API | PASS | reset→(obs,info), step→5-tuple |
| 5 | 5k-step no-NaN test | PASS | 0 non-finite over 5,000 steps |
| 6 | block map = 367 | PASS | sums to 367, matches contract v1.1.0 |
| 7 | zero-alpha clean | PASS | alpha block all 0, env runs |
| 8 | 3-state distinct | PASS | +1/-1/0/empty ≠ action HOLD |
| 9 | 3 alphas values/mask/summary | PASS | mask=3, summary math correct |
| 10 | 64 alphas shape+summary | PASS | (367,), summary ok |
| 11 | signal memory shift | PASS | lag0→lag1 demonstrated |
| 12 | MTF leakage | PASS | corrupt future → aligned[:t] unchanged |
| 13 | reward formula | PASS | equity-delta/start − breach_penalty |
| 14 | reward ⊥ alphas | PASS | byte-identical rewards, different alphas |
| 15 | reward std > 0 | PASS | std ≈ 0.42 at default position_size=100000 (realistic) |
| 16 | no alpha in reward | PASS | grep clean |
| 17 | FTMO/FREE day logic | PASS | worked numbers (target/auto-flat/breach/reset) |
| 18 | net arch / input 367 | PASS | MlpPolicy [256,256,256], in=367 |
| 19 | single-batch overfit | SKIP | needs torch → `tests/test_single_batch_overfit.py` (Colab) |
| 20 | PPO update finite/KL/entropy | SKIP | needs torch (Colab) |
| 21 | critic value_loss moves | SKIP | needs torch (Colab) |
| 22 | introspector read-only | PASS | no torch import / backward / optimizer |
| 23 | seed reproducibility | PASS | identical obs+rewards same seed |
| 24 | cache path / step speed | PASS | no pandas/talib in step; 310µs |
| 25 | 0/3/64 alpha timing <5× | PASS | 1.02× |
| 26 | full pipeline runs | PASS | cache→eval→doctor→barbershop ok |
| 27 | Policy Doctor on a model | PASS(mock)/SKIP(real) | saliency alpha=0.82 on mock; real=Colab |
| 28 | barbershop modules | PASS | all 6 produce output |
| 29 | walk-forward pass-rate | PASS | 3 windows, pass-rate produced |
| 30 | add-alpha shapes 1/5/64 | PASS | all (367,) |
| 31 | add-one-alpha test | PASS | `tests/test_add_one_alpha_shape.py` |
| 32 | no retrain / STATE_DIM | PASS | 367 fixed |
| 33 | save with 3 alphas | PASS(constr)/Colab | shape fixed at 367 |
| 34 | add 5 → 8 alphas | PASS(constr) | shape 367 |
| 35 | load old model runs | PASS(constr)/Colab | new slots = 0 |
| 36 | no crash/shape mismatch | PASS(constr) | 367 invariant |

**Tests:** 53/53 green (48 prior + 5 new audit tests). **Fixed this audit:** only the Phase-0
barbershop smoke test (it expected placeholder stubs that are now real modules). **Nothing in the
observation contract, reward, or leakage was changed** — F1 is flagged (not altered), F2 is resolved
(the default was already realistic).

---

## VERDICT

- **Ready to ADD ALPHAS: ✅ GO.** The alpha-slot plumbing is proven correct and shape-stable
  (367 fixed at 1/5/64 alphas), unassigned = 0, mask and summary correct, reward provably
  independent of alphas, and the multi-timeframe cache is leak-free. Adding alphas is safe and
  needs no retraining or schema change.

- **Ready to TRAIN seriously: ⚠️ CONDITIONAL — two to-dos first:**
  1. **F1** — train with `VecNormalize(norm_obs=True)` (raw 367 obs is unscaled).
  2. Run the **4 Colab proof cells** (check_env, single-batch overfit, PPO-update sanity,
     save/load) — I cannot run torch here, so learnability is **by-design, not yet numerically proven**.
  
  **F2 (reward scale) is resolved** — the default `position_size=100000.0` is now realistic.

Do F1 + F2 and pass the Colab overfit test, and you have a green light to train.
