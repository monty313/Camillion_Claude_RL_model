# Camillion Claude RL Model

A modular reinforcement-learning **forex** framework where each strategy emits a
simple directional **alpha** (`+1` buy / `-1` sell / `0` inactive) and an RL agent
learns how to **combine those alphas** — while it can also trade directly from raw
indicators — all under **FTMO-style** challenge rules.

> Mission: not to maximize PnL, but to **repeatedly pass FTMO-style challenges**
> (2.5% daily target / 4% trailing wall). Evolved from the `Quantra` repo.

## Status — Phase 0 (skeleton)
Foundation only: configs, the locked observation contract, indicator registry +
stubs, `BaseStrategy` + `StrategyRegistry` (64 fixed slots), the signal/observation
builders, account + risk scaffolds, and the shape-contract tests. **No RL training,
no UI yet** (those are Phases 1 and 2).

## The observation (locked: 357 float32, contract `v1.0.0`)
| Block | Size | What it is |
|---|---|---|
| indicators | 190 | raw TA-Lib values, 5 TFs x 38 (NOT normalized) |
| alpha_values | 64 | strategy outputs `+1/-1/0` (fixed slots) |
| alpha_mask | 64 | occupancy: `1` assigned, `0` empty slot |
| alpha_summary | 4 | buy% / sell% / active% / net% |
| signal_memory | 5 | last-5-bar net signal balance |
| signal_accuracy | 2 | rolling 1-bar / 3-bar accuracy (no leakage) |
| account_daily | 7 | daily win%, pnl%, dd%, target%, risk%, trades%, streak% |
| account_episode | 7 | episode win%, pnl%, dd%, target%, pass%, risk%, streak% |
| time | 6 | time-of-day / day-of-week / session flags |
| portfolio | 8 | open positions, exposure, unrealized pnl, equity/balance |

**Adding strategies fills slots — the shape never changes.** See
`docs/OBSERVATION_CONTRACT.md`.

## Run the tests
```bash
# with pytest (Colab):
pip install pytest && pytest -q
# or stdlib-only (no install):
python tools/run_tests.py
```

## Repo layout
`config/` frozen constants + tunables + FTMO/FREE + speed knobs ·
`src/indicators` TA-Lib registry/stubs · `src/strategies` base + 64-slot registry ·
`src/signals` summary / memory / accuracy · `src/observation` contract + builder ·
`src/account` state + win/loss features · `src/risk` FTMO/FREE/breach ·
`src/env` RL env (Phase 1) · `src/training` trainer (Phase 1) ·
`src/jarvis` + `src/barbershop` UI/diagnostics (Phase 2) · `tests/` · `docs/`.

## Disclaimer
Research software for simulated prop-firm challenges. Not financial advice.
