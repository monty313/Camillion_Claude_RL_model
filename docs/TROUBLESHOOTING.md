# Common Problems & Fixes (training + trading)

> Generated from `src/jarvis/knowledge.py` — the SAME knowledge JARVIS uses. Ask JARVIS "how do I fix <X>?" in the cockpit and he answers from this, grounded in the real system.

Camillion is a PPO/MLP reinforcement-learning bot built to pass the FTMO challenge CONSISTENTLY:
+2.5% of the INITIAL balance per day -> +10% in ~4 days, while never breaching the 5% daily / 10%
total drawdown walls (a self-imposed 4% trailing wall trips first).

PIPELINE (data -> decision):
  config/            locked numbers + tunable risk knobs (constants.py = frozen contract; variables.py
                     = runtime-editable risk; ftmo_config.py = the active rules; asset_specs.py = per-asset sizing).
  src/data/          cache_builder.py precomputes every indicator ONCE, leak-free (last-closed-bar), to
                     float32 arrays. The env step() only reads these (NEVER TA-Lib/MT5/pandas in the hot loop).
  src/indicators/    real RSI/CCI/Bollinger/ATR/SMA (+ ADX as an ALPHA-PRIVATE indicator).
  src/strategies/    alphas = signal generators (+1 buy / -1 sell / 0 inactive). 64 fixed slots; the policy
                     learns a weight per slot. Two kinds: DIRECTIONAL (vote in the consensus) and
                     non-directional GATES (1/0, e.g. movement filters) that are excluded from the consensus.
  src/env/           TradingEnv: the locked 479-float observation (contract v1.5.0) + actions
                     {HOLD,BUY,SELL,CLOSE}. REWARD = equity change only (+ deliberate FTMO/NY shaping).
  src/account/risk/  AccountState + breach detection + the two-phase daily engine (hit +2.5% -> bank & stop).
  src/training/      PPO (SB3) with VecNormalize; env_fingerprint = the CPU/GPU parity + comparability anchor;
                     run_log = the training ledger (best policy = highest walk-forward pass-rate at a fingerprint).
  src/interpret/ + src/barbershop/  read-only diagnostics (PolicyIntrospector, Policy Doctor).
  src/jarvis/        the cockpit BACKEND: state_contract (the /state JSON), state_provider (live snapshot),
                     consistency (the system-logic read), council (OMEGA->JUSTICE->JARVIS reasoning), this
                     knowledge base. jarvis_bridge.py serves it read-only. The HUD is JARVIS Cockpit.dc.html.

THREE RULES THAT OVERRIDE EVERYTHING (CLAUDE.md): (1) never silently change the 479 observation shape;
(2) never change the FTMO numbers without saying so; (3) never put TA-Lib/MT5/pandas inside env.step().

## Training problems

### the policy is not improving / reward stays flat / it only ever HOLDs
- **Likely cause:** observation not standardized, reward too small to learn from, or too few steps/envs
- **Fix:** train through VecNormalize (norm_obs=True) — trainer.py already does this; give it more n_steps/n_envs and total_timesteps; check the reward magnitude isn't ~0; run the Policy Doctor (src/barbershop/policy_doctor.py) to see action distribution + per-block importance.
- **Where:** `src/training/trainer.py, src/barbershop/policy_doctor.py`

### equity jumps by ~2x a trade's P&L / banked profit looks doubled
- **Likely cause:** realized P&L double-count — adding balance/daily/episode P&L manually AND via record_close
- **Fix:** record_close is the SINGLE source of truth: it updates balance + daily/episode realized P&L + equity + tallies. Never also do `acc.balance += ...` in the env. (This bug was fixed; if it reappears, grep the env for manual += on balance/realized_pnl.)
- **Where:** `src/env/trading_env.py (step/_flatten), src/account/trade_history.py`

### I want to add a strategy/alpha without breaking my trained model
- **Likely cause:** confusion between filling a slot (safe) and resizing the obs (a contract bump)
- **Fix:** adding an alpha FILLS a fixed slot (slot value flips 0->±1) and does NOT change the obs shape — safe, the policy adapts by continuing training. If the alpha needs a NEW indicator, add it as an ALPHA-PRIVATE indicator (read via ctx, kept OUT of the obs) so the obs never changes. Only raising MAX_STRATEGIES is a contract bump (retrain).
- **Where:** `src/strategies/alpha_pack.py, src/indicators/base.py (per_tf_alpha_private_columns), docs/ENVIRONMENT_STATE.md`

### training is very slow on Colab
- **Likely cause:** the workload is CPU-bound (small MLP + branchy sim); the GPU mostly idles
- **Fix:** use SubprocVecEnv with more cores (SB3 recommends CPU-first for MlpPolicy); keep indicators precomputed in the cache (rule #3: nothing heavy in step()); see docs/TRAINING_SPEED_PLAN.md. A GPU only helps a lot after the JAX parallel-sim rewrite (docs/JAX_GPU_TPU_TRAINER_BLUEPRINT.md).
- **Where:** `src/training/vector_env_factory.py, docs/TRAINING_SPEED_PLAN.md`

### each env step is slow / TA-Lib or pandas error inside training
- **Likely cause:** rule #3 violation — TA-Lib/MT5/pandas called inside env.step()
- **Fix:** precompute ALL indicators once in src/data/cache_builder.py; env.step() must only read cached float32 arrays. Move any pandas/TA-Lib into the precompute, never the hot loop.
- **Where:** `src/data/cache_builder.py, src/env/trading_env.py`

### backtest/training results look too good to be true
- **Likely cause:** future leakage — using an indicator bar that had not closed yet, or accuracy that peeks ahead
- **Fix:** the cache aligns higher TFs by the LAST CLOSED bar (close_time <= this 1m close); signal accuracy is computed leak-free; ALWAYS judge on held-out walk-forward windows, not in-sample.
- **Where:** `src/data/cache_builder.py (_align_to_1m), src/signals/signal_accuracy.py, src/training (walk-forward)`

### the bot stopped exploring and always picks HOLD / entropy collapsed to ~0 / the policy went deterministic too early / entropy is 0.05 after training
- **Likely cause:** the entropy bonus (ent_coef) is off or too small, so the policy stopped exploring and locked onto one action — usually HOLD, the safe default — before it learned anything
- **Fix:** raise ent_coef in PPO_HPARAMS (src/training/trainer.py) from 0.0 to a small positive value (~0.005-0.02) so the policy keeps exploring; watch that entropy stays above ~0.3 early in training. Also check the reward isn't so tiny/sparse that HOLD is a local minimum. A deterministic (low-entropy) policy is dangerous live — it cannot adapt when the market shifts.
- **Where:** `src/training/trainer.py (PPO_HPARAMS ent_coef), src/barbershop/policy_doctor.py`

### the bot never trades even when I add strategies / alpha=0 seems to force a HOLD / adding alphas doesn't make it take setups
- **Likely cause:** conflating alpha-space with action-space — treating alpha=0 (no setup) as if it were ACTION_HOLD
- **Fix:** alpha=0 means 'that strategy has no setup right now' (alpha-space); ACTION_HOLD means 'the policy chose not to trade this step' (action-space). They share the integer 0 but are DIFFERENT spaces — the env must NEVER force HOLD just because the alphas are 0; the policy decides the action independently. If the bot never trades, look at the open-gate (5m CCI threshold) / day-lock / two-phase bank, NOT the alphas. See CLAUDE.md 'Alpha-state 0 vs action HOLD'.
- **Where:** `CLAUDE.md, src/strategies/base.py, src/env/trading_env.py`

### the model trades well in training but poorly in eval
- **Likely cause:** eval did not load the saved VecNormalize stats, so the observation is mis-scaled
- **Fix:** use trainer.load_for_eval (training=False) which loads the frozen mean/std saved next to the model; never eval a normalized-trained model on raw obs.
- **Where:** `src/training/trainer.py (load_for_eval, _vecnorm_path)`

### I have several trained models — which one do I trust?
- **Likely cause:** comparing models across different environments (different fingerprints)
- **Fix:** the best policy = highest walk-forward PASS-RATE among non-rejected runs AT THE SAME env_fingerprint(). A different fingerprint means the environment changed and the runs are NOT comparable. Use run_log.best_run(fingerprint=env_fingerprint()).
- **Where:** `src/training/run_log.py, src/training/env_fingerprint.py, docs/TRAINING_LEDGER.md`

### the env fingerprint changed after I edited something
- **Likely cause:** you changed the obs contract, the alpha roster, the FTMO rules, or the reward
- **Fix:** that is correct and intended — it marks a NEW experiment line. Log the run with the new fingerprint; don't compare its pass-rate against runs from the old fingerprint.
- **Where:** `src/training/env_fingerprint.py, docs/ENVIRONMENT_STATE.md`

### NaNs in the observation early in an episode
- **Likely cause:** indicators are NaN during warmup (before enough bars exist)
- **Fix:** this is normal — alphas return 0 during warmup and the env starts at `warmup`. Ensure warmup is large enough for the slowest indicator (e.g. higher-TF ADX needs many bars). Don't train on warmup bars.
- **Where:** `src/env/trading_env.py (warmup), src/indicators/base.py`

### training on multiple symbols, the bot only trades the easy one
- **Likely cause:** no incentive to generalize; sizes/rewards not comparable across assets
- **Fix:** use make_multi_symbol_vec_env (per-asset CALIBRATED size so ~one daily range ≈ +2.5% on every asset) and the cross-asset observation features (ATR-normalized, asset-class one-hot) so one policy can compare opportunities in common units.
- **Where:** `src/training/vector_env_factory.py, config/asset_specs.py`

### out-of-memory / crash on Colab during training
- **Likely cause:** too many parallel envs × large precomputed arrays (or a huge alpha table)
- **Fix:** fewer envs or a smaller window; turn on High-RAM; as alphas scale, store alpha tables as int8 + share one precomputed table across envs (the road-to-1000-alphas plan).
- **Where:** `config/training_speed_config.py, docs/ENVIRONMENT_STATE.md (Scaling alphas)`

### I have several policies — which do I run, and how do I keep them organized?
- **Likely cause:** no single ranked view of policies by how consistently they pass
- **Fix:** register each trained model in the policy registry; JARVIS ranks them by a CONSISTENCY score (walk-forward pass-rate, low max-DD, low day-to-day concentration). champion() = the one to run, and only policies at the SAME env fingerprint are comparable. Ask JARVIS 'which policy should I run?'
- **Where:** `src/jarvis/policy_registry.py, src/training/run_log.py, src/training/env_fingerprint.py`

### how do I add a new policy so JARVIS knows about it?
- **Likely cause:** the policy isn't registered yet
- **Fix:** one line: `python -m src.jarvis.policy_registry add --id my-policy --path models/camillion_ppo --fingerprint <fp> --pass-rate 0.8 --max-dd 3.5 --largest-day 30`, or policy_registry.add_policy(...). It appears in GET /policies and in JARVIS's context immediately.
- **Where:** `src/jarvis/policy_registry.py`

### how do I see day-by-day results — did I make +2.5% of initial and stay inside the trailing DD?
- **Likely cause:** a final equity number hides whether you passed CONSISTENTLY
- **Fix:** run the day-by-day report: `python -m src.training.daily_report --data data_cache --symbol EURUSD --model models/camillion_ppo`. It prints one row per day — P&L%, +TGT? (made +2.5% of initial), TRAIL_DD% + <WALL? (inside the 4% trailing wall), DAILY_LOSS%, BREACH, cumulative % — plus a PASS/FAIL summary. Full steps in docs/TRAINING_INSTRUCTIONS.md.
- **Where:** `src/training/daily_report.py, docs/TRAINING_INSTRUCTIONS.md`

### should I train per symbol, or on all four at once?
- **Likely cause:** training one symbol at a time can't learn to balance the portfolio
- **Fix:** train on ALL FOUR Google-Drive symbols TOGETHER with train_multi_symbol — one policy over EURUSD/GBPUSD/XAUUSD/US30 with per-asset calibrated size + the cross-asset features. That is how the bot learns to BALANCE the book (read every asset in common units, allocate risk across them) instead of overfitting the easiest one. Mount Drive, build a cache per symbol, then train_multi_symbol({sym: load_cache(...)}, ...).
- **Where:** `src/training/trainer.py (train_multi_symbol), src/training/vector_env_factory.py, docs/TRAINING_INSTRUCTIONS.md`

### how do I train the bot? I don't trade and I don't want a bunch of confusing steps
- **Likely cause:** training used to be several manual steps
- **Fix:** ONE command: put your four 1-minute CSVs (filenames containing EURUSD/GBPUSD/XAUUSD/US30) in one folder, then run `python run_training.py --data <that_folder>`. It finds the files, prepares the features, trains ONE bot on all four from one shared account, prints the DAY-BY-DAY +2.5% / 4%-trailing results, and files the policy. In Colab it's one cell: `!python run_training.py --data /content/drive/MyDrive/Camillion_data`. Install the engine once with `pip install stable-baselines3 torch`.
- **Where:** `run_training.py, docs/TRAINING_INSTRUCTIONS.md`

## Trading / FTMO problems

### the account breached the daily-loss or max-drawdown wall
- **Likely cause:** lost more than 5% in a day / 10% total before the protective stop engaged
- **Fix:** the self-imposed 4% trailing wall should trip BEFORE FTMO's 5/10%. Check FTMO_TRAILING_ENABLED and the trailing pct in variables.py; size down; JARVIS posture goes STAND DOWN when headroom is thin. Protecting the challenge always beats one trade.
- **Where:** `config/variables.py (FTMO_TRAILING_*), src/risk/breach_detector.py`

### not reaching +2.5% per day / behind pace to pass
- **Likely cause:** size too small relative to the account, or too few quality setups
- **Fix:** calibrate lots so ~one daily range ≈ +2.5% of INITIAL (config/asset_specs.calibrated_position_size) while a full adverse day stays < 4%; raise selectivity to higher-consensus setups rather than over-trading. JARVIS will say the next step in its ruling.
- **Where:** `config/asset_specs.py, src/jarvis/consistency.py`

### hit the daily target then gave the gains back
- **Likely cause:** did not bank and stop — kept trading after +2.5%
- **Fix:** the two-phase engine: hit +2.5% of initial -> CLOSE ALL & BANK & stop for the day (FTMO_TWO_PHASE_ENABLED). If you opt to continue, a tight 1% trailing protects the banked gain. Consistency is built by NOT giving it back.
- **Where:** `config/variables.py (FTMO_TWO_PHASE_*, FTMO_PHASE2_*), src/env/trading_env.py`

### the bot won't open new trades
- **Likely cause:** the 5m CCI open-gate is blocking (neutral market), or the day is locked after banking
- **Fix:** the open-gate forbids NEW directional opens when EITHER 5m CCI is within ±threshold; it's a feature (avoid chop). Tune OPEN_GATE_CCI_THRESHOLD or turn the gate off. After banking the daily target the day is locked (no new opens until tomorrow) — also intended.
- **Where:** `config/variables.py (OPEN_GATE_CCI_THRESHOLD), src/env/trading_env.py (open_gate, _day_locked)`

### position size seems wrong on indices/gold vs forex
- **Likely cause:** one fixed lot size used across assets — wrong, because value-per-point differs hugely
- **Fix:** size PER ASSET: PnL = position × price_move × contract_size. Use asset_specs.calibrated_position_size(symbol); value_per_point differs (EURUSD 100000, XAUUSD 100, US30 1).
- **Where:** `config/asset_specs.py (value_per_point, calibrated_position_size)`

### a single bad day wiped a big chunk of the account
- **Likely cause:** lots too large for the asset's volatility
- **Fix:** calibrate so ~one typical daily range ≈ +2.5% and a full ADVERSE day stays under the 4% wall. If a day can lose >4%, the lots are too big — shrink them.
- **Where:** `config/asset_specs.py, config/variables.py`

### the NY index bonus didn't pay
- **Likely cause:** the bonus has strict conditions
- **Fix:** it only applies to INDICES, on CLOSED-IN-PROFIT P&L, during the NY session (open 13:30 UTC), reaching ≥50% (within 2h) / ≥100% (within 3h) of the daily target — and it is PAID AT DAY-END ONLY IF THE DAY PASSED (≥+2.5% of initial). It is erased on a failed day or a breach.
- **Where:** `config/variables.py (FTMO_NY_*), src/env/trading_env.py (_ny_qualify, _ny_day_end_bonus)`

### a movement-filter alpha shows as a BUY / inflates the bullish count
- **Likely cause:** a non-directional GATE outputs 1 (=movement on), which looks like a +1 buy
- **Fix:** gates are EXCLUDED from the directional consensus (registry.directional_mask). The bridge sends the directional-only net_signal + net_signal_basis; the HUD must use those, never divide by a hardcoded 15. A gate's 1 means 'the market is moving', not 'buy'.
- **Where:** `src/signals/signal_summary.py, src/jarvis/state_provider.py (directional_mask)`

### is the bot a single-asset trader, or does it trade everything at once?
- **Likely cause:** you may not know the shared-pot portfolio trainer exists
- **Fix:** it is a PORTFOLIO trader — ONE shared equity/drawdown pot across the WHOLE FTMO universe. Train it with train_portfolio on the shared-pot PortfolioEnv: one policy holds SIMULTANEOUS positions across ALL symbols in one account, decides one symbol at a time while seeing the pot's exposure, and is rewarded on the pot — so it learns to BALANCE risk. Because decisions are per-symbol with portfolio context, the obs stays 479 and it scales to the full FTMO broker list live. See docs/TRAINING_INSTRUCTIONS.md; watch the live book on the heatmap tab.
- **Where:** `src/env/portfolio_env.py, src/training/trainer.py (train_portfolio), docs/TRAINING_INSTRUCTIONS.md`

## Data & cache problems

### building the cache fails / wrong columns / no close
- **Likely cause:** the CSV columns weren't recognized
- **Fix:** load_ohlcv_csv accepts flexible names (datetime/timestamp/date+time; open/high/low/close/volume with common aliases). You at least need a datetime column and a close. Check the CSV header.
- **Where:** `src/data/cache_builder.py (load_ohlcv_csv)`

### training crashes with 'no datetime column found' / MetaTrader (MT5) export won't load / header looks like <DATE>\t<TIME>\t<OPEN>... / only one column / 1 bar per day
- **Likely cause:** MT5 history exports are TAB-separated with angle-bracket headers and SPLIT <DATE>/<TIME> columns and dotted dates (2021.01.13) — old loaders assumed a comma file with one datetime column
- **Fix:** the loader now auto-detects comma/TAB/semicolon delimiters, strips <...> from headers, COMBINES split <DATE>+<TIME> (keeping 1-minute resolution), parses dotted MT5 dates, and uses TICK volume (<VOL> is usually 0 on forex). Just point run_training at the folder and re-run. If it still fails, open the first row and confirm it has O/H/L/C plus a date and time.
- **Where:** `src/data/cache_builder.py (load_ohlcv_csv), tests/test_csv_loader.py`

### a symbol's sizing/asset-class is wrong or unknown
- **Likely cause:** the symbol isn't in SPECS / not classified
- **Fix:** add it to config/asset_specs.SPECS (contract_size, pip, typical_daily_range, asset_class) or rely on infer_asset_class for the class. Unknown symbols fall back to a sane default size.
- **Where:** `config/asset_specs.py (SPECS, infer_asset_class)`

## Observation-contract problems

### shape mismatch error / 'expected 479 got N' / a saved model won't load
- **Likely cause:** the observation contract changed (someone edited a block size or MAX_STRATEGIES)
- **Fix:** the obs is LOCKED at 479 float32 (contract v1.5.0). Rule #1: never change it silently. If a change is intended, bump OBSERVATION_CONTRACT_VERSION + update docs/OBSERVATION_CONTRACT.md + the shape tests, and RETRAIN (old models are incompatible). Check config/constants.py OBS_TOTAL_SIZE.
- **Where:** `config/constants.py, docs/OBSERVATION_CONTRACT.md`

## Cockpit / JARVIS problems

### JARVIS / the cockpit says model_attached: false
- **Likely cause:** no trained policy is attached, so the action is an honest alpha-consensus fallback
- **Fix:** train a model (or load one) and attach it: StateProvider.from_cache(dir, symbol, policy=sb3_policy_fn(*load_for_eval(...))). Until then JARVIS reports confidence 0 and flags the policy fields as fallback — it never fabricates a decision.
- **Where:** `src/jarvis/state_provider.py (from_cache), src/training/trainer.py (load_for_eval, sb3_policy_fn)`

### the cockpit shows movement but it's not my real market data
- **Likely cause:** the provider is running from_synthetic (placeholder prices)
- **Fix:** switch to real data: StateProvider.from_cache(your_cache_dir, symbol). Build the cache first with src/data/cache_builder.build_cache on your 1m OHLCV.
- **Where:** `src/jarvis/state_provider.py (from_cache), src/data/cache_builder.py`

### uvicorn jarvis_bridge:app fails / FastAPI not found
- **Likely cause:** the optional web deps aren't installed
- **Fix:** pip install -r requirements-jarvis.txt (fastapi, uvicorn). The contract + council logic need none of these and are tested without them (python tools/run_tests.py).
- **Where:** `requirements-jarvis.txt, jarvis_bridge.py`

### the HUD is blank or 404 at /JARVIS Cockpit.dc.html
- **Likely cause:** the HUD files aren't in the repo root, or pullLive() isn't wired
- **Fix:** drop JARVIS Cockpit.dc.html + support.js into the repo root, then apply the 4-method patch in docs/JARVIS_LIVE_WIRING.md (pullLive + councilLive + the net-signal/gate fix).
- **Where:** `docs/JARVIS_LIVE_WIRING.md`

### JARVIS's advice feels generic
- **Likely cause:** no LLM key set, so the council uses its deterministic (still system-grounded) text
- **Fix:** set ANTHROPIC_API_KEY (and pip install anthropic) on the bridge host to enable the LLM layer (claude-opus-4-8). Without it the council is still grounded + progressive, just less conversational.
- **Where:** `src/jarvis/council.py (llm_available), requirements-jarvis.txt`

### I can't make JARVIS place or close a trade from the cockpit
- **Likely cause:** JARVIS is READ-ONLY by design
- **Fix:** that's intentional and structural — the bridge has GET routes only (POST/PUT/PATCH/DELETE return 405). JARVIS observes, reasons and advises; he never touches the trading code or places orders.
- **Where:** `jarvis_bridge.py`

### where is the market heatmap / how do I see all symbols at once?
- **Likely cause:** the heatmap is its own cockpit tab fed by a separate endpoint
- **Fix:** GET /heatmap returns the buy/sell signal of EVERY FTMO symbol (direction, strength, buy/sell %, hottest alpha) — its own tab. Wire the tab per docs/JARVIS_LIVE_WIRING.md; the same rows are also on /state as `heatmap`.
- **Where:** `jarvis_bridge.py (/heatmap), src/jarvis/market_view.py, docs/JARVIS_LIVE_WIRING.md`
