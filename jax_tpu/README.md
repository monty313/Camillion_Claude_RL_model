# jax_tpu/ — Camillion's JAX/TPU trainer (one folder, everything JAX)

Train **one bot on a TPU** by playing **thousands of trading lifetimes at once**, until it passes
**40 FTMO challenges in a row on held-out data** — saving every policy + its details + a progress
ledger to Google Drive so a Colab disconnect never loses work.

This is the **same bot as the CPU trainer, written for the TPU**. It is a *second implementation of
the same environment* and must match the CPU env (`src/env/trading_env.py`) **bar-for-bar**: same 479
observation, same reward, same FTMO rules, same `env_fingerprint`. A parity test proves it before any
training runs. Read **`PLAN.md`** for the full design + the exact numbers it reproduces.

## How to run it (Colab TPU)
Open **`notebooks/Camillion_JAX_TPU_Train.ipynb`**, set Runtime → **TPU**, run top to bottom. It:
1. checks the TPU, mounts Drive, installs `jax[tpu]`/flax/optax;
2. **runs the parity gate** (TPU env == CPU env) — training is blocked if it fails;
3. builds market features from your Drive CSVs via the **existing** `run_training.prepare_caches`;
4. probes TPU utilization and suggests a scale for **70–80%** usage;
5. **trains to 40-in-a-row**, checkpointing to `MyDrive/Camillion/jax_models/` every eval;
6. plots the progress ledger and **exports the best policy to ONNX** for MT5.

## How to verify locally (CPU, no TPU needed)
```bash
pip install "jax[cpu]" flax optax           # + onnxscript onnxruntime for the ONNX export
python jax_tpu/tests/run_parity.py           # stdlib runner (no pytest)
pytest jax_tpu/tests -q                       # or via pytest (self-skips if jax absent)
```

## What's here
| file | what it is |
|---|---|
| `PLAN.md` | the full build plan + every exact number the JAX code reproduces |
| `jax_config.py` | PPO hyperparams (mirror the CPU trainer) + TPU scale knobs + the 40-in-a-row stop + Drive paths |
| `jax_ftmo.py` | FTMO breach + two-phase banking as branchless `jnp` (1:1 with `src/risk/ftmo_rules.py`) |
| `jax_obs_blocks.py` | the 40 **dynamic** obs floats in `jnp` (1:1 with `src/account/win_loss_features.py`) |
| `jax_static_features.py` | host builder: the shared `(T,479)` **static** obs tensor + per-bar scalars from a CPU `TradingEnv` |
| `jax_env.py` | `EnvState` pytree + branchless `step_env` (indexes the static tensor; recomputes the 40 dynamic floats) |
| `jax_portfolio_env.py` | **(core goal)** shared-pot, symbol-cycling env: per-symbol decisions, alpha-shaping, midnight day-scoring + 4-in-a-row, pot-level breach/pass/two-phase banking — 1:1 with `src/env/portfolio_env.py` |
| `jax_indicators.py` | **optional** on-device indicators in `jnp` (parity-tested vs `src/indicators/*`); not on the critical path |
| `jax_ppo.py` | Flax 3×256 tanh policy/value + GAE + clipped PPO loss + a `VecNormalize`-style obs normalizer |
| `jax_trainer.py` | pmapped rollout+update at scale, domain-randomized risk, **40-in-a-row stop**, Drive checkpoints + resume; env-agnostic (`train()` single-symbol, `train_portfolio()` shared pot) |
| `jax_eval.py` | held-out walk-forward **pass-rate (= P(pass))** + the consecutive-pass streak (works for either env) |
| `jax_progress.py` | **live FTMO-consistency progress**: per-eval readout (P(pass) + the streak-to-40 bar) + a Colab `LiveDashboard` you watch as it trains |
| `jax_checkpoint.py` | Drive persistence: params + obs-norm + details JSON + the `jax_progress.jsonl` ledger + best/passed dirs |
| `export_to_pytorch.py` | JAX/Flax → PyTorch 3×256 → **ONNX** (the MT5 deploy path), bit-verified |
| `tests/` | `test_jax_parity.py` (single-symbol **step gate**), `test_jax_portfolio_parity.py` (**portfolio gate**), `test_jax_indicators.py`, `run_parity.py` |

## The architecture in one paragraph
The 479-float observation splits into **439 static per-bar floats** (indicators, alphas, masks, summaries,
signal memory/accuracy, time, cross-asset, alpha-streak — precomputed once by the CPU env and **shared
read-only** across all envs) and **40 dynamic floats** (account/portfolio/sizing/recent-context — they
depend on the evolving account state). The JAX step **indexes** the shared static tensor and recomputes
only the 40 dynamic floats + reward + FTMO in branchless `jnp`. So 439/479 of the obs is byte-identical
to the CPU env, the hot loop has **no TA-Lib/pandas** (CLAUDE.md rule #3), and each of the thousands of
parallel envs carries only a ~34-scalar `EnvState` — which is what lets a TPU hold them all at once.

## Status — COMPLETE
- ✅ Single-symbol step-parity gate **green** (CPU vs JAX: max |obs| ≈ 1e-7, max |reward| ≈ 1e-20).
- ✅ **Portfolio (shared-pot) step-parity gate green** — incl. forced two-phase banking, won days +
  4-in-a-row, and breach paths (max |obs| ≈ 2e-7, max |reward| ≈ 1e-10). The core-goal env is verified.
- ✅ Indicator parity **green** (RSI/ATR/CCI/SMA/Bollinger vs the CPU pandas reference).
- ✅ Full pipeline runs end-to-end for BOTH envs (`train()` / `train_portfolio()`): pmap rollout+PPO,
  held-out eval, **live FTMO-consistency dashboard**, Drive checkpoint+resume, ONNX export.
- ✅ 13/13 JAX parity tests + the full repo suite green.

**You watch progress live**: every eval prints `P(pass) @ 2.5%/4%`, the held-out return, the breach rate,
and a bar of `N/40 challenge passes in a row`; in Colab the `LiveDashboard` redraws those curves as it trains.
