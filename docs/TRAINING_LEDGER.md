# TRAINING LEDGER — every run, every result, which policy to follow

> **Purpose:** never get confused about which trained policy is the real one. Every training run is
> recorded with its **environment fingerprint** and its **FTMO walk-forward pass-rate**, so we can
> always answer *"which policy do we follow?"* — objectively, not from memory.

## Where it lives
- **Data:** `records/run_ledger.jsonl` — one JSON line per run (committed; it's the record).
- **Code:** `src/training/run_log.py` — `log_run(...)`, `load_runs()`, `best_run(...)`.
- **Identity:** `src/training/env_fingerprint.py` — the env hash stamped on every run.

## The rule (do this for EVERY run)
1. Right before training, compute `fp = env_fingerprint()`.
2. After training + walk-forward eval, call:
   ```python
   from src.training.run_log import log_run
   log_run(run_id="2026-06-25_seed7_gpu", fingerprint=fp, contract_version=C.OBSERVATION_CONTRACT_VERSION,
           trainer="gpu", symbols=["EURUSD","US30","XAUUSD","GBPUSD"], seed=7,
           total_timesteps=10_000_000, pass_rate=0.62, breaches=3, n_windows=40,
           status="candidate", model_path="drive/.../ppo_seed7.zip", notes="ORB on")
   ```
3. **Commit the updated `records/run_ledger.jsonl`** (or sync it off Drive) so the record persists.

## Which policy to follow
```python
from src.training.env_fingerprint import env_fingerprint
from src.training.run_log import best_run
best = best_run(fingerprint=env_fingerprint())   # highest pass-rate, SAME environment, not rejected
```
- **Always filter by fingerprint** so you only compare policies trained on the *same* environment.
- `status`: `candidate` (default) → `best` (the one you're using) / `rejected` (ruled out).

## CPU vs GPU (and seeds)
- Log each run with `trainer="cpu"` or `"gpu"`. **Same fingerprint ⇒ comparable**, so a CPU run, a
  GPU run, and 20 seeds all rank together by pass-rate — pick the winner regardless of how it was made.
- A **different** fingerprint means a different environment: it starts its own line; don't compare
  across fingerprints.

## Fields
`run_id · timestamp · fingerprint · contract_version · trainer · symbols · seed · total_timesteps ·
pass_rate · breaches · n_windows · status · model_path · notes`

> `pass_rate` is the headline: the fraction of unseen walk-forward windows the policy passed
> (reached +10% with no breach). That is the number that decides which policy goes to FTMO.
