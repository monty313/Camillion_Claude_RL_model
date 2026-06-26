# records/

Append-only training records (the data behind `docs/TRAINING_LEDGER.md`).

- `run_ledger.jsonl` — one JSON line per training run: environment fingerprint + FTMO
  walk-forward pass-rate + where the model lives. Written by `src/training/run_log.py`.
  **Commit this file** (or sync it off Drive) after each run so the record is never lost.

Model weights / VecNormalize stats are NOT stored here (they're large + gitignored) — only the
lightweight records that tell us which policy to trust.
