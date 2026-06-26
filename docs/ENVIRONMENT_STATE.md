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

## 3. Rules for building the GPU trainer (when we create it)

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

## 4. Tracking which policy to follow
Every training run is recorded in the ledger (`docs/TRAINING_LEDGER.md`, data in
`records/run_ledger.jsonl`). The current best policy = highest walk-forward **pass-rate** among
non-rejected runs **of the current fingerprint**: `run_log.best_run(fingerprint=env_fingerprint())`.
