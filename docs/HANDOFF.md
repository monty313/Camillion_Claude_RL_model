# Camillion — Handoff

**Repo:** `monty313/Camillion_Claude_RL_model` (private) · **Branch:** `main`
**Mission:** an RL bot that passes the FTMO Challenge consistently. A PPO/MLP policy
learns to combine many directional "alpha" signals under FTMO risk rules.
**Status:** alpha pack v1.2.0 in; 75/75 stdlib tests green; not yet trained on real data.

---

## What exists now
- **15 alphas** (each emits +1 / -1 / 0; a *suggestion*, the policy decides):
  - slot 0 `gravity_30m_4h_agree`
  - 1-4 Regime Pulse (trend/pullback × 5m-30m, 30m-4h)
  - 5-8 CCI Surge (trend/pullback × 5m-30m, 30m-4h)
  - 9-12 SMA Stack (trend/pullback × 5m-30m, 30m-4h)
  - 13-14 SMA Reversion (5m-30m, 30m-4h)
  Each is its own `src/strategies/<name>.py` + `register_<name>.py`; `alpha_pack.register_all()` wires all 15.
- **Observation contract v1.2.0 = 451 float32** (was 367): indicators 220, alpha_values 64, alpha_mask 64,
  alpha_summary 4, signal_memory 5, signal_accuracy 2, account_daily 7, account_episode 7, time 6,
  portfolio 8, **alpha_streak 64** (per-alpha signal-streak, normalized). Leak-free, auto-generated from `config/constants.py`.
- **FTMO objective aligned to the real 2-Step Challenge:** **+10% pass target** (episode terminates as PASS),
  5% daily / 10% static max; the old 2.5%/day auto-flat and 4% trailing are **OFF by default** (runtime-editable in `config/variables.py`).
- **Transaction cost in the reward:** `TRANSACTION_COST_FRAC_PER_SIDE = 0.000035` (~0.8 pip round-trip on EURUSD),
  charged on every open and close. Reward is still equity-change-only (now net of cost) + breach penalty - / + pass bonus.
- **Single-symbol training notebook** `notebooks/Camillion_Single_Symbol_Train.ipynb`: trains gravity + 14 alphas on
  one real symbol, with an out-of-sample FTMO verdict and a per-alpha **edge diagnostic**.

## How to run training (Colab)
1. Put your 1m CSV on Drive (auto-found): e.g. `Camillion_data/EURUSD_M1_*.csv`.
2. Open `Camillion_Single_Symbol_Train.ipynb` in Colab → add a Secret `GITHUB_TOKEN` (your PAT) → set `SYMBOL`.
3. Runtime → Run all. Start with `TOTAL_TIMESTEPS=20_000` (smoke), then full. `WARMUP=50_000` (needed so the 4h
   BB200/CCI100 alphas aren't silent). Read the OOS return / maxDD / wall verdict and the alpha-edge table.

## How to push code to GitHub
Open `pusher_phase3.html`, pick `manifest.json` → it overlays files (base_tree, never deletes).
Confirmation = green `PUSHED commit ...`.

## Honest open issues / next steps (priority)
1. **No alpha is validated to have edge yet.** Run the edge-diagnostic cell on real EURUSD; keep alphas with
   `edge(bps) > 0 & hit% > 50`, cut the noise.
2. **Cost is a flat fraction.** Upgrade to per-bar spread from the CSV `SPREAD` column for fidelity.
3. **Run a real Colab training run** to get an honest OOS FTMO number (the single number that matters).
4. **Parked / not integrated:** the 1.7 GB PPO `.pt` (its manifest shows pass_rate 0 — likely not worth it) and the
   Quantra DQN `.h5` (small, code available) as model-alphas; Phase-4 multi-symbol portfolio env (built, parked, NOT pushed).
5. **Diversity:** the 15 alphas lean trend-following; consider genuinely different signals (vol regime, extremes).

## Key files
- contract: `config/constants.py`, `src/observation/observation_contract.py`, `src/observation/builder.py`
- env/reward/FTMO: `src/env/trading_env.py`, `config/variables.py`, `config/ftmo_config.py`, `src/risk/*`
- alphas: `src/strategies/*` (+ `alpha_pack.py`); diagnostics: `src/barbershop/*` (+ `alpha_edge.py`)
- training: `src/training/*`; tests: `tests/` (run `python tools/run_tests.py`, no pytest needed)

## Notes
- Collaborator docs on GitHub (`PROJECT_SPEC.md`, `OPERATOR_CONTROLS.md`, `FTMO_REPO_AUDIT.md`, `READINESS_AUDIT.md`)
  are preserved by the pusher (excluded from the manifest), not overwritten.
