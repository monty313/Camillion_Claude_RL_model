# Camillion Claude RL Model

A modular reinforcement-learning **forex** framework where each strategy emits a
simple directional **alpha** (`+1` buy / `-1` sell / `0` inactive) and an RL agent
learns how to **combine those alphas** — while it can also trade directly from raw
indicators — all under **FTMO-style** challenge rules.

> Mission: not to maximize PnL, but to **repeatedly pass FTMO-style challenges**
> (2.5% daily target / 4% trailing wall). Evolved from the `Quantra` repo.

## Status — JAX/TPU trainer + multi-head "super-scalper" actor
The full pipeline is built and CPU↔JAX bar-for-bar parity-verified: a shared-pot **PortfolioEnv** (all symbols,
one FTMO pot), an on-device **PPO trainer** (`jax_tpu/`), a rich **observation** (v1.12.0, **557** float32),
and a **multi-head actor** that outputs direction + continuous **TP / SL / lot** (bracket orders with a hard
1%-equity risk clamp), trained via a **freeze/unlock curriculum** and a self-discovering **R:R reward**.
The bracket actor ships **default-OFF** (`bracket_enabled=0`) so the proven discrete bot is unchanged.

## The observation (locked: **557** float32, contract `v1.12.0`)
Raw indicators (220 = 44×5 TFs) + alpha values/mask/summary/memory/accuracy + account (daily/episode) + time +
portfolio + alpha streak + sizing + cross-asset + recent-context + raw OHLC + trade-risk + consistency +
**momentum**-perception + **hug-pressure** + **dual-BB interactions** + **1m scalp-momentum**. Blocks are
**append-only** and version-bumped; adding strategies fills the 64 alpha slots without changing the shape.
Full per-block table + version history: **`docs/OBSERVATION_CONTRACT.md`**.

## QUICKSTART — train the bot
Training runs on a Colab **TPU** via **`jax_tpu/notebooks/Camillion_JAX_TPU_Train.ipynb`** (Steps 0→8b; Step 8c
is the "prove it learned the principle" report). Two knobs select the mode:
- **`bracket_enabled`** — a Step-8b `env_param_kwargs` value (`0`=discrete bot, `1`=TP/SL/lot bracket actor).
- **`ACTOR_CURRICULUM_STAGE`** — in `config/constants.py` (`1`=freeze tp/sl/lot, `2`=unlock lot, `3`=unlock all).

```text
a) BASELINE (discrete actor — the proven path; recommended first run)
   Run the notebook Steps 0→8b as shipped. bracket_enabled is unset -> 0. Read Step 8c (proof report).

b) STAGE 1 CURRICULUM (learn WHEN to trade; tp/sl/lot FROZEN at ~1:1 R:R + 1x lot)
   - config/constants.py:        ACTOR_CURRICULUM_STAGE = 1
   - Step 8b env_param_kwargs:    add   bracket_enabled=1.0
   Run 8b. Watch direction accuracy / win-rate stabilize (not PnL yet).

c) FLIP TO STAGE 2, then 3   (nothing else changes — the flag drives the rollout + PPO)
   - Stage 2 (unlock lot):   config/constants.py  ACTOR_CURRICULUM_STAGE = 2
   - Stage 3 (unlock all):   config/constants.py  ACTOR_CURRICULUM_STAGE = 3
   Keep bracket_enabled=1.0. After a real run, `src/analysis/rr_histogram.py` on the bracket log shows the
   LEARNED R:R per alignment (conviction) quartile — the research output.
```
**Deploy (MT5/ONNX):** `jax_tpu/export_to_pytorch.py` emits ONNX with outputs
`direction_logits[4], tp_pct[1], sl_pct[1], lot_mult[1]` (heads clipped+mapped to final units; the deployer
applies the 1% lot clamp using live equity). ⚠️ The **MT5 EA (`.mq5`) is not in this repo** — its inference
side must read these **4 outputs** (was 1 before v1.12.0), or it will silently break.

## Run the tests
```bash
pytest -q                                    # full CPU suite (300 tests)
JAX_ENABLE_X64=1 pytest -q jax_tpu/tests/    # CPU<->JAX parity gates
python tools/run_tests.py                    # stdlib-only fallback (no install)
```

## Repo layout
`config/` frozen constants + tunables + FTMO/FREE + speed knobs ·
`src/indicators` TA-Lib registry/stubs · `src/strategies` base + 64-slot registry ·
`src/signals` summary / memory / accuracy · `src/observation` contract + builder ·
`src/account` state + win/loss features · `src/risk` FTMO/FREE/breach ·
`src/env` shared-pot PortfolioEnv (+ TP/SL/lot bracket model) · `src/observation` obs blocks
(momentum / hug / bb-interactions / scalp) · `src/analysis` R:R histogram ·
`jax_tpu/` on-device PPO trainer + envs + **multi-head policy** + CPU↔JAX parity + ONNX export ·
`src/jarvis` read-only cockpit · `tests/` · `docs/`.

## Disclaimer
Research software for simulated prop-firm challenges. Not financial advice.
