# REPO GUIDE — how everything in this folder works (read this first)

> Plain-language but complete. Monty is not a programmer — this explains every folder, the data
> flow, how to run things, and the traps. For *fixing* problems, see `docs/TROUBLESHOOTING.md`
> (and just **ask JARVIS** in the cockpit — he reads the same knowledge). For the cockpit itself,
> see `docs/JARVIS_GUIDE.md`.

---

## 0. What this is, in one paragraph
Camillion is a **reinforcement-learning trading bot** that learns to combine many simple trading
signals ("alphas") into one policy whose only job is to **pass the FTMO challenge consistently**:
make **+2.5% of the initial balance per day → +10% in ~4 days**, while never breaching the **5%
daily / 10% total** drawdown walls (a self-imposed **4% trailing wall** trips first). One bot is
meant to trade **all broker assets** from one shared equity/drawdown pot.

## 1. The three rules that override everything (from `CLAUDE.md`)
1. **Never silently change the observation shape.** It is locked at **479 float32** (contract
   `v1.5.0`). Adding strategies fills empty slots; the shape never changes. Changing it is a
   deliberate "contract bump" that makes old trained models incompatible.
2. **Never change the FTMO numbers** (2.5% / 4% / 5% / 10% / +10%) without saying so explicitly.
3. **Never put TA-Lib / MT5 / pandas inside `env.step()`.** The hot loop only reads cached
   float32 arrays. Speed is the #1 priority (training runs `env.step()` millions of times).

## 2. The folder at a glance
```
config/          the "contract" + tunable knobs (numbers the bot lives by)
src/
  data/          builds the cached market features (once, leak-free)
  indicators/    the technical indicators (RSI/CCI/Bollinger/ATR/SMA, + ADX alpha-private)
  strategies/    the alphas (signal generators) + the 64-slot registry
  env/           TradingEnv: the observation + actions + FTMO engine + reward
  signals/       turns alpha votes into the net signal / memory / leak-free accuracy
  observation/   assembles + locks the 479-float observation contract
  account/       AccountState + closed-trade ledger + account feature blocks
  risk/          breach detection (daily / total / trailing), FTMO vs FREE rules
  training/      PPO trainer, parallel envs, the fingerprint + the run ledger
  interpret/     read-only look inside the policy (action dist, value, entropy)
  barbershop/    read-only "Policy Doctor" diagnostics
  jarvis/        the read-only cockpit BACKEND (bridge + council + knowledge)  -> JARVIS_GUIDE.md
config files at root: jarvis_bridge.py (the server), CLAUDE.md (rules), pyproject.toml
tools/           run_tests.py (stdlib test runner, no pytest needed)
tests/           the test suite (locks the contract, FTMO rules, leakage, alphas, bridge)
docs/            every design + operating doc (this file, troubleshooting, contract, blueprints)
notebooks/       one-click Colab training notebooks
records/         the training ledger data (one line per run)
```

## 3. The data → decision pipeline (the spine)
```
 1m OHLCV ─► src/data/cache_builder ─► (bars × 220) float32 indicator cache  [ONCE, leak-free]
                                          │
 TradingEnv.__init__ precomputes ONCE ◄───┘  alphas (bars×64), net signal, leak-free accuracy, time
                                          │
 env.step(action) [HOT LOOP, millions/run]│  reads ONLY cached float32:
   • mark position to price, update equity (record_close = the single P&L truth)
   • check FTMO breach / two-phase banking
   • assemble the 479 observation, reward = equity change (+ deliberate FTMO/NY shaping)
                                          │
 PPO (Stable-Baselines3, MlpPolicy 3×256) + VecNormalize ─► trained policy
                                          │
 walk-forward eval (load frozen norm stats) ─► pass-rate ─► run ledger (which policy to trust)
                                          │
 JARVIS cockpit (read-only) ─► /state + council advice  [never trades]
```

## 4. Each area, in detail

### `config/` — the contract + the knobs
- **`constants.py`** — the **frozen contract**. Timeframes `(1m,5m,30m,4h,1d)`, the indicator specs
  (which SMA/CCI/RSI/ATR/Bollinger lines exist), `MAX_STRATEGIES = 64` alpha slots, the
  observation block sizes, `OBSERVATION_CONTRACT_VERSION = "v1.5.0"`, `OBS_TOTAL_SIZE = 479`,
  `ACTIONS = (HOLD,BUY,SELL,CLOSE)`. **No logic.** Every number here is baked into trained models —
  editing it is a contract bump.
- **`variables.py`** — the **safe-to-edit knobs**: `FTMO_DAILY_TARGET_PCT=2.5`,
  `FTMO_TRAILING_DRAWDOWN_PCT=4.0`, `FTMO_TWO_PHASE_ENABLED`, `OPEN_GATE_CCI_THRESHOLD`, the symbol
  universe, costs. These are **percentages/toggles**, so the policy reads them as fractions and
  **you can change them without retraining**.
- **`ftmo_config.py`** — builds the active rules object (`FTMOConfig` or `FreeModeConfig`) from
  `variables.py`. `load_active_config()` is what the env/risk use; `update_risk_settings(...)`
  changes them live.
- **`asset_specs.py`** — per-asset sizing. `value_per_point` differs hugely (EURUSD 100000, XAUUSD
  100, US30 1), so `calibrated_position_size(symbol)` sizes each asset so **~one daily range ≈
  +2.5%** and a full bad day stays under 4%. `infer_asset_class` covers the whole FTMO broker.
- **`training_speed_config.py`** — speed knobs (parallel env count, window length, cache format) and
  the hard rule flags (no TA-Lib/MT5 in step).

### `src/data/` + `src/indicators/` — the feature cache
- **`cache_builder.py`** — the most important data file. `build_aligned_indicators(df1m)` resamples
  1m → 5m/30m/4h/1d, computes every indicator, and **aligns higher timeframes by the LAST CLOSED
  bar** (so the bot never peeks at an unfinished bar → **leak-free**). Output: a `(bars × 220)`
  float32 array the env reads. `build_cache`/`load_cache` persist it. `load_ohlcv_csv` reads your
  CSV with flexible column names.
- **`indicators/`** — real `sma.py`, `cci.py`, `rsi.py`, `bollinger.py`, `atr.py` (Wilder, TA-Lib
  fast-path) and `adx.py` (ADX, used **alpha-private** — read by alphas, kept out of the obs).
  `base.py` enumerates the canonical column order so the cache and the live obs always line up.

### `src/strategies/` — the alphas
- An **alpha = a signal generator**: `compute_signal(ctx) -> +1` buy / `-1` sell / `0` inactive.
  Two kinds: **directional** (vote in the consensus) and **non-directional gates** (`1/0`, e.g. a
  movement filter) that are **excluded** from the directional consensus.
- **`registry.py`** — 64 fixed slots. The policy only ever sees each slot's **output** (+1/-1/0) +
  the occupancy mask — never the strategy's internals. `directional_mask()` separates gates.
- **`base.py`** (`BaseStrategy`, `DIRECTIONAL` flag), **`context.py`** (`MarketContext` = the per-bar
  snapshot an alpha reads), **`alpha_pack.py`** (`register_all` wires the roster), and the alpha
  families: `gravity_*`, `regime_pulse_*`, `cci_surge_*`, `sma_stack_*`, `sma_reversion_*`,
  `orb_ny_breakout_*`. **Roster on `main` = 16** (gravity + 14-pack + ORB); **18 with PR #12** (the
  two `dual_movement_filter` gates).

### `src/env/` + `src/signals/` + `src/observation/` — perception + the engine
- **`trading_env.py`** — the heart. `__init__` precomputes the alpha matrix, net signal, leak-free
  accuracy, and time features once. `step(action)` is the hot loop: act at the bar, update the
  account (`record_close` is the single P&L truth), check breach + the two-phase daily engine
  (hit +2.5% → bank & stop), assemble the **479** observation, and return reward =
  **equity change** (+ deliberate breach penalty / +10% pass bonus / NY index bonus).
- **`signals/`** — `signal_summary.py` (buy%/sell%/active%/**net%** over *directional* alphas),
  `signal_memory.py` (last-5 net), `signal_accuracy.py` (leak-free rolling accuracy).
- **`observation/`** — `observation_contract.py` (the locked block table + names) and `builder.py`
  (concatenates the blocks in order, sanitizes NaN→0, validates the 479 shape).

### `src/account/` + `src/risk/` — money + walls
- **`account_state.py`** — `AccountState`: balance, equity, daily & episode tallies, peaks, loss
  streaks. **`trade_history.py`** — `record_close()` is the **single source of truth** for P&L.
  **`win_loss_features.py`** — the daily/episode/sizing/cross-asset/recent-context feature blocks.
- **`risk/breach_detector.py`** — `detect(acc, cfg)` enforces 5% daily / 10% total / 4% trailing,
  in both FTMO and FREE modes; flags the two-phase auto-flat at +2.5%.

### `src/training/` + `src/interpret/` + `src/barbershop/` — learning + looking inside
- **`trainer.py`** — PPO (SB3) with `VecNormalize` (standardizes the obs at train time; the stats
  are saved next to the model and **must** be reloaded for eval). `train`, `train_multi_symbol`,
  `resume`, `load_for_eval`, `sb3_policy_fn`.
- **`vector_env_factory.py`** — parallel envs (`SubprocVecEnv`); multi-symbol round-robin with
  per-asset calibrated size. **`gym_adapter.py`** — wraps `TradingEnv` as a Gym env.
- **`env_fingerprint.py`** — the **comparability anchor**: same config → same `env_fingerprint()`;
  CPU/GPU must match. **`run_log.py`** — the training ledger; `best_run(fingerprint=...)` = the
  policy with the highest walk-forward pass-rate at the *current* fingerprint.
- **`interpret/policy_introspector.py`** — read-only look inside the policy (action distribution,
  value, entropy, which obs block matters). **`barbershop/policy_doctor.py`** — a read-only
  scoreboard (is it learning? is it leader-chasing?).

### `src/jarvis/` — the cockpit backend → see **`docs/JARVIS_GUIDE.md`**
The read-only bridge + the OMEGA/JUSTICE/JARVIS council + the knowledge base.

### `tools/`, `tests/`, `docs/`, `notebooks/`, `records/`
- **`tools/run_tests.py`** — runs the whole suite with **no pytest** (`python tools/run_tests.py`).
- **`tests/`** — locks the obs contract (479), FTMO breach/two-phase rules, leakage (no future
  peeking), alpha truth tables, multi-symbol balancing, and the JARVIS bridge contract.
- **`docs/`** — this guide, `JARVIS_GUIDE.md`, `TROUBLESHOOTING.md`, `OBSERVATION_CONTRACT.md`,
  `ENVIRONMENT_STATE.md`, `TRAINING_LEDGER.md`, the GPU/TPU blueprint, the live-wiring patch.
- **`notebooks/`** — one-click Colab training. **`records/`** — the run ledger data (commit it).

## 5. The 479 observation (what the bot "sees" each bar)
`indicators(220) + alpha_values(64) + alpha_mask(64) + alpha_summary%(4) + signal_memory(5) +
signal_accuracy(2) + account_daily(7) + account_episode(7) + time(6) + portfolio(8) +
alpha_streak(64) + sizing(10) + cross_asset(10) + recent_context(8) = 479`. Adding an alpha fills a
slot (0→±1); the **shape never changes**. Everything risk-related is a **percentage**, so changing
the dollar limits never confuses the bot.

## 6. How to run things
- **Prove it works:** `python tools/run_tests.py` (or `pytest -q`). All green = healthy.
- **Change risk (no retrain):** edit `config/variables.py` (target, trailing, mode), restart.
- **Train:** open `notebooks/Camillion_One_Click_Train.ipynb` in Colab → load data, build cache,
  `register_all`, `trainer.train(...)`. The model + its `_vecnorm.pkl` are saved — **commit both**.
- **Pick the policy to trust:** highest walk-forward pass-rate at the current fingerprint
  (`run_log.best_run(fingerprint=env_fingerprint())`); see `docs/TRAINING_LEDGER.md`.
- **Run the cockpit:** `pip install -r requirements-jarvis.txt` then
  `uvicorn jarvis_bridge:app --port 8000`. (Details in `JARVIS_GUIDE.md`.)
- **Add an alpha (no retrain):** write a `BaseStrategy`, register it in `alpha_pack.py` — it fills a
  slot; the obs shape doesn't change. Need a new indicator? Add it **alpha-private** so the obs
  still doesn't change.

## 7. The traps (read before editing)
- **The 479 obs is locked.** Don't change a block size casually — it's a contract bump + retrain.
- **Nothing heavy in `env.step()`.** Precompute in the cache; the step only reads float32.
- **`record_close` is the only place P&L is booked.** Never also do `acc.balance += pnl` (that
  double-counts — a real bug we fixed).
- **Eval must reload the frozen VecNormalize stats**, or results are garbage.
- **Leakage = #1 false-optimism source.** Trust held-out walk-forward, not in-sample.
- **Fingerprint changes when you change the env** (obs/alphas/FTMO/reward) — that's correct; it just
  starts a new experiment line. Don't compare pass-rates across fingerprints.
- **`alpha = 0` (no setup) ≠ `ACTION_HOLD` (policy took no trade) ≠ empty slot.** Keep them distinct.

## 8. "Where do I look when…"
| You want to… | Go to |
|---|---|
| change risk / target / trailing | `config/variables.py` |
| understand the obs | `docs/OBSERVATION_CONTRACT.md`, `config/constants.py` |
| add/inspect an alpha | `src/strategies/` + `alpha_pack.py` |
| see the trading engine / reward | `src/env/trading_env.py` |
| build/feed real data | `src/data/cache_builder.py` |
| train / pick a model | `src/training/` + `docs/TRAINING_LEDGER.md` |
| run / wire the cockpit | `jarvis_bridge.py` + `docs/JARVIS_GUIDE.md` |
| **fix a problem** | **`docs/TROUBLESHOOTING.md` or just ask JARVIS** |
