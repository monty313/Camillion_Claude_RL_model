# JAX GPU/TPU TRAINER BLUEPRINT — the full-rewrite plan (build only when CPU is the wall)

> **UPDATE 2026-06-28 — partially BUILT in `jax_tpu/`.** A first implementation now lives in the repo's
> `jax_tpu/` folder (operator-requested ahead of the CPU wall, for TPU scale + train-to-40-in-a-row).
> It follows this blueprint's invariants: a **step-parity gate** proves the JAX env matches the CPU env
> bar-for-bar (max|obs|≈1e-7, max|reward|≈1e-20), indicators are parity-tested in jnp, and it trains on
> a TPU with a custom Flax PPO until **40 consecutive held-out challenge passes**, checkpointing to
> Drive. Key design choice vs this doc: indicators/alphas are **precomputed once on the host and shared**
> as a static `(T,479)` tensor (the "build once + share" plan), so the hot loop indexes it and only the
> 40 account-dependent obs floats are recomputed in jnp — see `jax_tpu/PLAN.md`. The single-symbol env
> is done; the shared-pot **PortfolioEnv** parity is the next phase in that folder. The text below is the
> original full-rewrite plan and remains the contract.

> **Status: BLUEPRINT, not built.** This is the plan for a *future* from-scratch rewrite
> that runs the WHOLE training loop on-device (GPU or TPU) in JAX/Flax. The CPU env
> (`src/env/trading_env.py`) stays the **reference / source of truth**. Do not start this
> until training **time** is the proven bottleneck (we have not even done the full Colab
> run yet). When you do, this file + `docs/ENVIRONMENT_STATE.md` §4 + the GPU-rules header
> in `src/training/env_fingerprint.py` are the contract.

---

## 0. The goal (why this rewrite exists)

Run **vast amounts of market data, over and over, across thousands of parallel
simulations**, until we have a policy that **passes the FTMO challenge consistently** —
and along the way:

1. **Daily target and risk are runtime inputs you can change anytime** (no retrain).
2. **The trainer reports the *likelihood* of passing** — a pass-probability readout, per
   (daily-target, risk) setting, so you can see "at 2.5% target / 4% trailing, the bot
   passes ~X% of the time."

Everything below serves those three things: scale, runtime-tunable risk, pass-likelihood.

---

## 1. Scope and honest cost (read before committing)

This is a **complete rebuild**, not an optimization:
- **You abandon PyTorch + Stable-Baselines3** and write a **custom PPO in pure JAX/Flax**.
- You **rewrite the env** as branchless, fixed-shape array math that lives on the device.
- You **lose the readability** the rest of the repo prizes (CLAUDE.md "simple, commented,
  Colab-friendly"). On-device JAX is dense and hard for a non-programmer to edit. That is
  the price of the speed — accept it deliberately or don't start.
- The payoff is real and large: published JAX on-device RL (Brax / PureJaxRL / gymnax)
  routinely shows **100–1000× more environment-steps per unit time** than a CPU loop,
  because thousands of envs run in lockstep with **zero host↔device transfer**. Treat
  "minutes instead of weeks" as the *direction*, not a guarantee — measure it.

**Decision rule:** only build this if (a) the optimized-CPU rung in
`docs/TRAINING_SPEED_PLAN.md` is exhausted AND (b) training wall-clock is blocking
progress toward a consistent pass-rate.

---

## 2. NON-NEGOTIABLE invariants (the rewrite must preserve these exactly)

**The governing rule — VERSION PAIRING.** CPU, GPU, and TPU are **not three different bots.
They are ONE bot written three ways.** They share the **same contract version, the same
`env_fingerprint()`, the same behaviour, and the same policy-file format** — only the *code*
differs per hardware (Python/NumPy on CPU, JAX on GPU/TPU). So they carry **one version
number across all three** (e.g. all are "env v1.5.0 / fp `abc123`"). If behaviour changes,
**every implementation bumps together, in the same PR.** This is *why* there is nothing to
confuse: a policy is tagged by version+fingerprint, not by which machine produced it, so
CPU/GPU/TPU runs are ranked side-by-side in the one ledger. The items below are how you
enforce that pairing.

The new engine is a **second implementation of the same environment**. It MUST match the
CPU reference or it is a bug, not a trainer:

1. **Same observation contract.** Emit the **identical** observation (currently
   `v1.5.0`, **479** float32, block order in `config/constants.py`). Same blocks, same
   indices, same meaning. A change here follows the obs-contract protocol (bump version +
   `docs/OBSERVATION_CONTRACT.md` + shape tests) — on BOTH implementations, same PR.
2. **Same FTMO numbers.** 2.5%/day target, 4% phase-1 trailing wall, two-phase
   (+2.5% → bank → 1% trailing), +10% challenge target. Never silently change them.
3. **Same alpha-vs-action separation.** `alpha = 0` (no setup) ≠ `ACTION_HOLD` (policy
   took no action) ≠ empty slot (mask 0). Keep them distinct.
4. **Same fingerprint.** `env_fingerprint()` of the JAX env MUST equal the CPU env's for
   the same config. A matching hash = "same environment"; only then are policies
   comparable in the ledger.
5. **Step-parity test.** Step the CPU env and the JAX env on identical bars; assert
   **observations and rewards match within float tolerance**. Matching hash + mismatched
   steps = a bug. This is the single most important test in the rewrite.
6. **Same policy interface.** Identical obs size + action space {HOLD, BUY, SELL, CLOSE},
   so a JAX-trained policy is ranked head-to-head with CPU/GPU policies by pass-rate.
7. **Per-slot alphas + the int8/shared-table memory plan** (see `ENVIRONMENT_STATE.md`
   §3) carry over — the alpha table is just a static device tensor here.

---

## 3. The paradigm: pure JAX/Flax **co-location**

Both the **market environment** and the **model** live inside device memory (GPU VRAM /
TPU HBM). Nothing crosses the CPU↔device bus in the hot loop, so the transfer bottleneck
that kills naive RL-on-accelerator drops to ~zero.

```
                         [ Inside GPU/TPU memory ]
 [ JAX Matrix Env 1 ] [ JAX Matrix Env 2 ] ... [ JAX Matrix Env 10,000 ]
          \                    |                      /
           v                   v                     v
        [ One fixed-size tensor batch of market states ]
                              |
                              v
                     [ Flax policy/value network ]
                              |
                              v
            [ jit-compiled PPO update — one XLA graph ]
```

**One codebase, two device targets.** JAX runs on **both** CUDA GPUs and TPUs. The same
rewrite serves "fully GPU" and "fully TPU"; the device is a flag. The discipline below is
*mandatory on TPU* and *strongly beneficial on GPU* (see §6 for the differences).

---

## 4. The five rebuild rules (how the CPU-only blockers become wins)

These map 1:1 to the five issues that make our env hostile to accelerators.

**Rule 1 — Indicators become `jax.numpy`, fused into the graph.**
Rewrite RSI/CCI/ATR/SMA/BB in native `jnp` ops (no TA-Lib/pandas/Numba). When jit-compiled,
XLA **operator-fusion** merges indicator math + network layers into one execution block,
killing memory-bandwidth stalls. (Precompute what is static once; compute the rest fused.)

**Rule 2 — Fixed-length episodes + masking (no variable shapes).**
Force every episode to a strict max length (e.g., exactly 500 bars). If an account breaches
or banks early at bar 120, the env keeps stepping to 500 with an internal `active=False`
flag. Pass a parallel **mask** (1 = real, 0 = dead) to the loss so it ignores dead bars.
The device always sees a rigid static shape → **no recompilation stalls**.

**Rule 3 — Branchless FTMO logic via `lax.select` / `lax.cond`.**
Delete Python `if/else` from the step. Rewrite every FTMO decision (breached? banked?
day passed? stop hit?) as `jax.lax.select`/`jax.lax.cond` over arrays. The device computes
both paths and **mask-selects** the answer — slow branching becomes raw vectorized math.

**Rule 4 — Custom PPO in JAX/Flax (goodbye SB3).**
Write the PPO loop (rollout, GAE, clipped objective, optimizer step) in pure JAX/Flax.
This unlocks `jax.vmap` (replicate one env to thousands) and `jax.pmap` (shard across all
cores) — one line each. Keep PPO hyperparameters matched to the CPU trainer initially so
results are comparable.

**Rule 5 — The tiny MLP finally gets fed.**
The net stays small, but **10,000 envs in parallel** push the effective batch from ~64 to
**256,000+ states per gradient step**, so the matrix engine is fully utilized evaluating
the small MLP for a huge army at once. *This is what makes the accelerator worth it despite
our small network.*

---

## 5. Runtime-changeable daily target & risk (a first-class requirement)

We must be able to **dial daily target and risk at inference, anytime, with no retrain** —
and have the bot still behave well. Two mechanisms, both preserved/strengthened from the
CPU design:

1. **Keep risk as PERCENTAGE observations, never baked into weights.** The obs already
   exposes target/DD as fractions (progress-to-target, % DD used/remaining) — that's *why*
   `config/variables.py` risk knobs are "change at runtime, no retrain." The JAX env passes
   `daily_target` and `risk` as **part of the per-env state**, feeding those same %
   features. The network reads the situation, not the absolute dollars.
2. **Domain-randomize target/risk across the 10,000 envs during training.** Sample a
   different `(daily_target, risk)` for each parallel env from realistic ranges. The policy
   then learns to handle the **whole range**, so at inference you can set any value and it
   adapts. This is the on-device payoff applied to robustness, not just speed.

Result: one policy, robust across target/risk settings; you change the dial live and read
the resulting pass-likelihood (next section).

---

## 6. Fully GPU vs fully TPU (what actually differs)

Same JAX codebase; the device changes the constraints and the knobs:

| | **Fully GPU** (JAX-on-CUDA) | **Fully TPU** (JAX-on-XLA/TPU) |
|---|---|---|
| Static shapes | Strongly preferred | **Mandatory** (dynamic shape → recompile death) |
| Parallelism | `vmap` (usually 1 GPU on Colab) | `pmap` across the 8 TPU cores + `vmap` |
| Precision | fp32 / tf32 / fp16 | **bf16** native |
| Branching | `lax.select` preferred | `lax.select` **required** |
| Custom kernels | possible (Pallas/Triton) | avoid; stay pure XLA |
| Tolerance for "messy" code | higher | **lowest** — discipline is non-optional |
| Best when | one strong device, faster to get right | you want the absolute max parallel throughput |

**Practical order:** get it working **fully GPU first** (more forgiving, easier to debug
and to step-parity against the CPU env), then flip the device flag and tighten for TPU.
Both must still match the CPU fingerprint.

---

## 7. The pass-likelihood readout ("tell us the chance of passing")

This is a required output, not a nice-to-have. Because thousands of envs run at once, the
**pass-rate falls out for free** and *is* the probability estimate:

- **Definition:** for a given `(daily_target, risk)`, run `M` **held-out walk-forward**
  windows (unseen data). A window **passes** if it reaches the +10% challenge target
  **without** breaching the daily/total drawdown. `pass_rate = passes / M`.
- **`pass_rate` = the estimated probability of passing the FTMO challenge** under that
  setting. Report it with a confidence interval (M is large here, so it's tight).
- **Report a grid**, not a single number: `pass_rate` as a function of `(daily_target,
  risk)`. That's the deliverable — "at 2.5%/4% you pass ~X%; at 3%/4% ~Y%."
- **Log every evaluation to the ledger** (`docs/TRAINING_LEDGER.md` /
  `records/run_ledger.jsonl`) with the fingerprint + `trainer="jax-gpu"|"jax-tpu"`, so the
  "current best policy" stays = highest held-out pass-rate at the current fingerprint.

---

## 8. The training loop ("over and over until consistently passing")

```
repeat:
    1. roll out N steps across 10,000 envs (randomized target/risk, varied data slices)
    2. compute GAE + clipped PPO loss (masked for dead bars)
    3. one jit-compiled optimizer step
    every K iterations:
        4. evaluate pass_rate on held-out walk-forward windows (§7)
        5. log to ledger; checkpoint policy
    stop when: pass_rate >= TARGET (e.g. >= 90% at 2.5%/4%) on held-out data
               for C consecutive evaluations  (consistency, not a lucky spike)
```

- **Generalization, not memorization:** train on many data slices/symbols + randomized
  risk; **judge only on held-out** windows. A high train pass-rate with a low held-out
  pass-rate = overfitting → keep going / vary more, don't ship.
- **Consistency gate:** require the pass-rate to hold for `C` consecutive held-out evals
  before declaring success — one good eval is noise.

---

## 9. Build order (so a future agent can execute)

1. **Skeleton + parity harness first.** Stand up a minimal JAX env for ONE symbol that
   emits the v1.5.0 observation; write the **step-parity test vs the CPU env** (§2.5)
   before anything else. Nothing proceeds until steps match.
2. **Indicators in `jnp`** (Rule 1) — parity-tested against the cached CPU indicators.
3. **Fixed-length + masking** (Rule 2) and **branchless FTMO** (Rule 3) — re-run parity.
4. **`vmap` to thousands of envs** (Rule 4/5) on GPU; confirm fingerprint unchanged.
5. **Custom PPO in JAX/Flax**; match CPU hyperparameters; first end-to-end train on GPU.
6. **Pass-likelihood eval + ledger logging** (§7).
7. **Domain-randomized target/risk** (§5).
8. **Flip to TPU** (§6): enforce static shapes/bf16/`pmap`; re-parity; benchmark.
9. **Document** every step in `UPDATE_LOG.md` (IRAC) and update `ENVIRONMENT_STATE.md`.

---

## 10. When NOT to do this
- If the CPU/optimized-CPU path already trains fast enough to iterate → don't. The speed is
  worthless if it costs us readability and parity we don't need yet.
- If the observation contract or FTMO rules are still changing weekly → wait; you'd be
  re-doing the parity work each change, on two codebases.
- If no one can maintain dense JAX → the bus-factor risk outweighs the speed.

**Bottom line:** this rewrite turns the bot into a hyper-parallel engine that plays
thousands of trading lifetimes at once, lets you change target/risk on the fly, and reports
your odds of passing — but only earn it by matching the CPU reference bar-for-bar, and only
build it when training *time* is the thing standing between us and a consistent pass-rate.



####If you completely rebuild it from scratch to fit the hardware, then yes—you can absolutely benefit from the TPU, and the performance payoff would be massive.
By discarding your current PyTorch/SB3 stack and rewriting the entire pipeline, you can turn your trading bot into a hyper-parallelized engine capable of processing millions of market bars per second across thousands of symbols simultaneously.
Here is the exact blueprint of how a complete rebuild unlocks the TPU, how it changes your metrics, and the strict engineering rules you must follow.
------------------------------
## The Paradigm Shift: Pure JAX/Flax Co-Location
To benefit from the TPU, you must use JAX and Flax (Google's TPU-native deep learning library). The core strategy is Co-Location: both your FTMO market environment and your RL model must live inside the TPU's high-bandwidth memory (HBM).

               [ Inside TPU Memory ]
 [ JAX Matrix Env 1 ] [ JAX Matrix Env 2 ] ... [ JAX Matrix Env 10,000 ]
         \                    |                    /
          \                   |                   /
           v                  v                  v
     [ Combined Fixed-Size Tensor Batch of Market States ]
                              |
                              v
                   [ Flax Neural Network ]

Because everything happens inside the TPU chip, the CPU-to-TPU communication bottleneck drops to zero.
------------------------------
## How a Rebuild Solves Your 5 Issues
If you commit to a full rewrite in JAX, your 5 issues turn into massive performance wins:
## 1. Preprocessing is Flattened Into the Tensor Graph

* How it changes: Instead of using CPU libraries, you write your indicators (RSI, CCI) using native jax.numpy operations.
* The TPU Benefit: When you compile this code, JAX uses Operator Fusion. It combines your indicator math and your neural network layers into a single, massive mathematical execution block on the TPU hardware, eliminating memory-bandwidth delays.

## 2. Variable Days are Eliminated via Masking and Vectorization

* How it changes: You force every trading episode to have a strict, maximum length (e.g., exactly 500 bars). If an account breaches or banks a profit on bar 120, the JAX environment keeps running to bar 500 but sets an internal state flag to "inactive."
* The TPU Benefit: The TPU always receives a perfectly rigid, static matrix shape. You pass a parallel array of 0s and 1s (masks) to the loss function so it ignores the dead data. The TPU never stalls for re-compilation.

## 3. Conditionals are Converted into Mathematical Selects

* How it changes: You completely delete standard Python if/else branching code. You rewrite your FTMO logic using jax.lax.select or jax.lax.cond.
* The TPU Benefit: Instead of jumping around code branches, the TPU calculates both paths simultaneously and uses a lightning-fast mathematical mask to select the correct answer. It replaces slow logic with raw speed.

## 4. Re-architecting the Stack (Goodbye PyTorch/SB3)

* How it changes: You abandon PyTorch and SB3. You write a custom PPO or DQN algorithm in pure JAX/Flax.
* The TPU Benefit: You gain native access to jax.vmap (Vectorized Map) and jax.pmap (Parallel Map). With a single line of code, JAX takes your single-symbol trading environment and replicates it 10,000 times across all TPU cores instantly.

## 5. Overcoming the Tiny MLP Payoff (Massive Scaling)

* How it changes: Your network is still small, but your data throughput becomes infinite.
* The TPU Benefit: Because the TPU is processing 10,000 environments in parallel, your effective batch size scales from 64 to 256,000+ states per gradient step. The TPU's matrix engine is suddenly fully utilized because it is evaluating the small MLP for a massive army of environments all at the exact same time.

------------------------------
## The Ultimate Payoff
If you do this rewrite, you can load every single minute bar of historical data for 500 symbols over 10 years directly into the TPU memory. The bot can play out thousands of lifetimes of trading configurations simultaneously, finding a highly generalized policy in minutes instead of weeks.
------------------------------
To determine if this massive engineering lift makes sense for your project, let me know:

* How long does your current CPU training run take right now?
* Do you have the engineering runway to spend several weeks rewriting pure Python logic into complex matrix math?

I can provide a code skeleton showing how a trading environment step function is written in pure JAX using jax.lax.select.

####