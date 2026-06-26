# ENVIRONMENT STATE — the living record  (single source of truth)

> **What this is:** the always-current description of *everything the bot's environment includes*
> — observations, reward, FTMO rules, alphas, sizing. If it's not written here, it doesn't count.
> **Golden rule:** every change to the environment updates THIS file in the same PR.
>
> The machine-readable version is `src/training/env_fingerprint.py:env_spec()`, and its 12-char
> hash `env_fingerprint()` stamps every training run. **Same fingerprint = same environment.**

---

## 1. Current state (as of contract v1.5.0)

| Part | Value |
|---|---|
| Observation contract | **v1.5.0**, **479 float32** (full block table in `docs/OBSERVATION_CONTRACT.md`) |
| Observation blocks | indicators(220) · alpha_values(64) · alpha_mask(64) · alpha_summary(4) · signal_memory(5) · signal_accuracy(2) · account_daily(7) · account_episode(7) · time(6) · portfolio(8) · alpha_streak(64) · sizing(10) · cross_asset(10) · recent_context(8) |
| Action space | 4 — HOLD / BUY / SELL / CLOSE |
| Reward | equity-change/step **+ deliberate shaping**: −breach_penalty, +pass_bonus(+10%), +NY index bonus *(when merged)* |
| FTMO rules | +2.5%/day of **initial**; phase-1 **4% trailing**; two-phase bank → optional **1% trailing**; **+10%** challenge pass; daily 5% / total 10% hard lines |
| Per-asset sizing | `config/asset_specs.py` — each symbol sized so ~one daily range ≈ +2.5%, full adverse day < 4% |
| Alpha roster | gravity + the pack *(15 on `main`; 16 once the ORB PR merges)* |
| Transaction cost | per-side fraction of notional (`TRANSACTION_COST_FRAC_PER_SIDE`) |

> Print the exact live values any time with `env_spec()` / `env_fingerprint()` — never trust memory.

---

## 2. UPDATE RULES — do this for EVERY environment change

1. **If the observation shape/contents change** → bump `OBSERVATION_CONTRACT_VERSION`, update
   `docs/OBSERVATION_CONTRACT.md` (block table + total) **and** every shape test (the locked-shape
   tests). Append-only blocks keep old indices stable; reordering/removing is a breaking change.
2. **If FTMO rules or reward change** → say so explicitly (it's a deliberate decision), update the
   relevant `config/variables.py` + this table, and note that the **fingerprint will change**.
3. **If the alpha roster changes** → it fills a fixed slot (no shape change), but the fingerprint
   changes — that's correct, those policies are a new experiment.
4. **Always** → append a dated IRAC entry to `docs/UPDATE_LOG.md`, and keep `pytest -q` green.
5. **The fingerprint is the truth.** After any change, a *different* `env_fingerprint()` means
   policies trained before/after are **NOT comparable** — start a fresh line in the ledger.

---

## 3. Scaling alphas toward ~1000 (how we function, so the obs stays stable)

Settled with the operator. Every future agent follows this so growing the alpha
library never destabilises the observation:

1. **Keep per-slot — do NOT switch to aggregates.** Each alpha owns one fixed
   slot → 3 obs inputs (value, mask, streak). That is what lets the **policy
   learn an individual weight per alpha**, which is the point. Aggregate /
   consensus features would throw that away; only use them if the operator
   explicitly asks.
2. **Filling a slot never changes the obs shape.** Assigning an alpha flips its
   slot's value `0 → ±1`; the number of inputs is unchanged, so trained policies
   keep working. Adding alphas up to `MAX_STRATEGIES` is free and shape-stable.
3. **Empty slots do NOT hurt learning.** A permanently-0 slot is invisible to the
   network (its weight never trains). 900 empties don't make the bot dumber.
4. **The only real cost is memory — and we solve it, not avoid it.** The env
   precomputes a `(bars × slots)` alpha table + streak table once; the hot loop
   just indexes a row (per-step CPU barely grows), and the per-alpha weighting
   lives in the **network (GPU work, scales fine)**. What grows is the table's
   RAM, especially when copied per parallel env. So: **(a)** store alpha/streak
   tables as **int8** (values are −1/0/+1 → 1 byte, 4× cut); **(b)** compute the
   table **once and share it read-only across envs**; **(c)** grow the filled
   count over time — empties then cost only cheap memory.
5. **Raising `MAX_STRATEGIES` is a deliberate contract bump** (it resizes 3 obs
   blocks). Follow §2: bump the version, update the contract doc + this file +
   shape tests; the fingerprint rolls automatically. Set it **once with
   headroom** — cheap while pre-training, costly once a real policy exists.

TL;DR: per-slot stays; never reshape the obs casually; beat memory with **int8 +
a shared precomputed table**, not by aggregating away per-alpha weighting.

---

## 4. Rules for building the GPU trainer (when we create it)

**The principle we expect to use (data-parallel RL).** The GPU trainer is **ONE shared
policy (one brain) learning from THOUSANDS of simulations running at once.** The GPU's
rule is that every sim runs the **same math instruction at the same moment** — but each
sim runs it on **different market data** (different day/slice/start + exploration noise),
so each produces a **different experience** (different trades, wins, losses). Those
thousands of diverse experiences are **pooled into one weight update.** So "all do the
same thing" means **same operation, different worlds** — it does NOT mean they learn the
same thing; it means one brain learns from a thousand different lives at once. This is
*why* the GPU is worth it: far more varied experience per unit of wall-clock → faster,
steadier learning. The hard part (and the whole job of the rewrite) is turning the
**branchy FTMO logic** (if breached / if banked / if day passed) into that **lockstep
math** — every per-sim decision becomes a mask/array op applied across all sims at once.
The *output is identical in kind* to the CPU trainer: the same policy file (see §3 of the
CPU-vs-GPU framing — same destination, different vehicle; use the GPU when training *time*
becomes the wall).

The CPU env is the reference. A GPU env/trainer is a *second implementation of the same thing*, so:

1. **Match the fingerprint.** The GPU env MUST yield the **same `env_fingerprint()`** as the CPU env
   for the same config (same obs contract, alphas, FTMO rules, reward).
2. **Match the steps.** Add a parity test that steps the CPU and GPU envs on identical bars and
   asserts the **observations and rewards match** (within float tolerance). Matching hash but
   mismatched steps = a bug.
3. **Same policy format.** Keep obs size + action space identical, so a GPU-trained policy is the
   **same file** as a CPU-trained one and they can be ranked together (see the ledger).
4. **Tag the trainer.** Log every run with `trainer="cpu"` or `"gpu"` and the fingerprint, so we
   compare like-for-like and never confuse which environment produced a policy.
5. **Don't diverge silently.** Any intended behaviour change goes through §2 (bump + docs + tests),
   on **both** implementations, in the same PR.

---

## 5. Tracking which policy to follow
Every training run is recorded in the ledger (`docs/TRAINING_LEDGER.md`, data in
`records/run_ledger.jsonl`). The current best policy = highest walk-forward **pass-rate** among
non-rejected runs **of the current fingerprint**: `run_log.best_run(fingerprint=env_fingerprint())`.
