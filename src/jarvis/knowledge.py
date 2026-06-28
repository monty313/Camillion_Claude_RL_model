# =====================================================================
# WHEN 2026-06-26 (Phase 2 JARVIS) | WHO Claude for Monty
# WHY  JARVIS's ALWAYS-ON knowledge base: a compact map of how the whole repo works
#      + a structured TROUBLESHOOTING brain for training & trading problems. The
#      council loads this into context every deliberation, so Monty can ask JARVIS
#      "how do I fix X?" and get a grounded, system-correct answer (not a guess).
# WHERE src/jarvis/knowledge.py
# HOW  Plain Python data (stdlib only). search(q) keyword-ranks the troubleshooting
#      entries; as_context(q) builds a compact LLM string; render_markdown() writes
#      the human guide (docs/TROUBLESHOOTING.md is generated from THIS single source).
# DEPENDS_ON: (stdlib only)
# USED_BY: src/jarvis/council.py (context + JARVIS prompt), jarvis_bridge.py
#          (GET /knowledge, GET /ask), docs/TROUBLESHOOTING.md (generated)
# CHANGE_NOTES(IRAC): I: operator wants to ask JARVIS directly how to fix any issue,
#   and JARVIS must ALWAYS have the system + fixes at hand. R: that request,
#   2026-06-26. A: one knowledge base (system summary + ~30 grounded fixes) the
#   council always sees + a search so the right fix surfaces for the question. C:
#   JARVIS coaches from the real system's logic toward a consistent pass.
# =====================================================================
"""JARVIS knowledge base: how the repo works + grounded fixes for training/trading issues."""
from __future__ import annotations
import re

# A compact map of the system, always in JARVIS's context.
SYSTEM_SUMMARY = """\
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
  src/env/           TradingEnv: the locked 499-float observation (contract v1.6.0) + actions
                     {HOLD,BUY,SELL,CLOSE}. REWARD = equity change only (+ deliberate FTMO/NY shaping).
  src/account/risk/  AccountState + breach detection + the two-phase daily engine (hit +2.5% -> bank & stop).
  src/training/      PPO (SB3) with VecNormalize; env_fingerprint = the CPU/GPU parity + comparability anchor;
                     run_log = the training ledger (best policy = highest walk-forward pass-rate at a fingerprint).
  src/interpret/ + src/barbershop/  read-only diagnostics (PolicyIntrospector, Policy Doctor).
  src/jarvis/        the cockpit BACKEND: state_contract (the /state JSON), state_provider (live snapshot),
                     consistency (the system-logic read), council (OMEGA->JUSTICE->JARVIS reasoning), this
                     knowledge base. jarvis_bridge.py serves it read-only. The HUD is JARVIS Cockpit.dc.html.

THREE RULES THAT OVERRIDE EVERYTHING (CLAUDE.md): (1) never silently change the 499 observation shape;
(2) never change the FTMO numbers without saying so; (3) never put TA-Lib/MT5/pandas inside env.step()."""


# Each fix is grounded in the real system. area in {training, trading, data, bridge, obs}.
TROUBLESHOOTING = [
    # ---------------- TRAINING ----------------
    {"id": "train-not-learning", "area": "training",
     "symptom": "the policy is not improving / reward stays flat / it only ever HOLDs",
     "cause": "observation not standardized, reward too small to learn from, or too few steps/envs",
     "fix": "train through VecNormalize (norm_obs=True) — trainer.py already does this; give it more "
            "n_steps/n_envs and total_timesteps; check the reward magnitude isn't ~0; run the Policy "
            "Doctor (src/barbershop/policy_doctor.py) to see action distribution + per-block importance.",
     "refs": "src/training/trainer.py, src/barbershop/policy_doctor.py"},
    {"id": "train-double-pnl", "area": "training",
     "symptom": "equity jumps by ~2x a trade's P&L / banked profit looks doubled",
     "cause": "realized P&L double-count — adding balance/daily/episode P&L manually AND via record_close",
     "fix": "record_close is the SINGLE source of truth: it updates balance + daily/episode realized P&L "
            "+ equity + tallies. Never also do `acc.balance += ...` in the env. (This bug was fixed; if it "
            "reappears, grep the env for manual += on balance/realized_pnl.)",
     "refs": "src/env/trading_env.py (step/_flatten), src/account/trade_history.py"},
    {"id": "train-obs-shape", "area": "obs",
     "symptom": "shape mismatch error / 'expected 499 got N' / a saved model won't load",
     "cause": "the observation contract changed (someone edited a block size or MAX_STRATEGIES)",
     "fix": "the obs is LOCKED at 499 float32 (contract v1.6.0). Rule #1: never change it silently. If a "
            "change is intended, bump OBSERVATION_CONTRACT_VERSION + update docs/OBSERVATION_CONTRACT.md + "
            "the shape tests, and RETRAIN (old models are incompatible). Check config/constants.py OBS_TOTAL_SIZE.",
     "refs": "config/constants.py, docs/OBSERVATION_CONTRACT.md"},
    {"id": "train-add-alpha", "area": "training",
     "symptom": "I want to add a strategy/alpha without breaking my trained model",
     "cause": "confusion between filling a slot (safe) and resizing the obs (a contract bump)",
     "fix": "adding an alpha FILLS a fixed slot (slot value flips 0->±1) and does NOT change the obs shape — "
            "safe, the policy adapts by continuing training. If the alpha needs a NEW indicator, add it as an "
            "ALPHA-PRIVATE indicator (read via ctx, kept OUT of the obs) so the obs never changes. Only "
            "raising MAX_STRATEGIES is a contract bump (retrain).",
     "refs": "src/strategies/alpha_pack.py, src/indicators/base.py (per_tf_alpha_private_columns), docs/ENVIRONMENT_STATE.md"},
    {"id": "train-slow-colab", "area": "training",
     "symptom": "training is very slow on Colab",
     "cause": "the workload is CPU-bound (small MLP + branchy sim); the GPU mostly idles",
     "fix": "use SubprocVecEnv with more cores (SB3 recommends CPU-first for MlpPolicy); keep indicators "
            "precomputed in the cache (rule #3: nothing heavy in step()); see docs/TRAINING_SPEED_PLAN.md. A "
            "GPU only helps a lot after the JAX parallel-sim rewrite (docs/JAX_GPU_TPU_TRAINER_BLUEPRINT.md).",
     "refs": "src/training/vector_env_factory.py, docs/TRAINING_SPEED_PLAN.md"},
    {"id": "train-step-slow", "area": "training",
     "symptom": "each env step is slow / TA-Lib or pandas error inside training",
     "cause": "rule #3 violation — TA-Lib/MT5/pandas called inside env.step()",
     "fix": "precompute ALL indicators once in src/data/cache_builder.py; env.step() must only read cached "
            "float32 arrays. Move any pandas/TA-Lib into the precompute, never the hot loop.",
     "refs": "src/data/cache_builder.py, src/env/trading_env.py"},
    {"id": "train-leakage", "area": "training",
     "symptom": "backtest/training results look too good to be true",
     "cause": "future leakage — using an indicator bar that had not closed yet, or accuracy that peeks ahead",
     "fix": "the cache aligns higher TFs by the LAST CLOSED bar (close_time <= this 1m close); signal "
            "accuracy is computed leak-free; ALWAYS judge on held-out walk-forward windows, not in-sample.",
     "refs": "src/data/cache_builder.py (_align_to_1m), src/signals/signal_accuracy.py, src/training (walk-forward)"},
    {"id": "run-the-audit", "area": "ops",
     "symptom": "am I ready / is the bot safe to run / pre-flight check / how do I know it all works / "
                "give me a GO or NO-GO",
     "cause": "you want one plain-English health check across PPO math, FTMO rules, the env, JARVIS and "
              "code quality before committing to a challenge",
     "fix": "run ONE command: `python tools/run_full_audit.py`. It runs the repo's full unit suite (~150 "
            "tests) PLUS 44 system checks and writes audit_results/audit_report.html (open it in a browser) "
            "plus .md/.json, ending in a big GO or NO-GO with the exact things to fix first. Any failing unit "
            "test forces NO-GO. Exit code 0 = GO, 1 = NO-GO.",
     "refs": "tools/run_full_audit.py, audit_results/audit_report.html, tests/test_full_audit.py"},
    {"id": "entropy-collapse", "area": "training",
     "symptom": "the bot stopped exploring and always picks HOLD / entropy collapsed to ~0 / the policy went "
                "deterministic too early / entropy is 0.05 after training",
     "cause": "the entropy bonus (ent_coef) is off or too small, so the policy stopped exploring and locked onto "
              "one action — usually HOLD, the safe default — before it learned anything",
     "fix": "ent_coef in PPO_HPARAMS (src/training/trainer.py) is 0.01 by default now; if the live action-mix "
            "heartbeat shows ~HOLD 100% early, raise it toward 0.02 (useful range ~0.005-0.02); if it never "
            "settles, lower it. Watch that entropy stays above ~0.3 early in training. "
            "Also check the reward isn't so tiny/sparse that HOLD is a local minimum. A deterministic (low-entropy) "
            "policy is dangerous live — it cannot adapt when the market shifts.",
     "refs": "src/training/trainer.py (PPO_HPARAMS ent_coef), src/barbershop/policy_doctor.py"},
    {"id": "alpha-vs-hold", "area": "training",
     "symptom": "the bot never trades even when I add strategies / alpha=0 seems to force a HOLD / adding alphas "
                "doesn't make it take setups",
     "cause": "conflating alpha-space with action-space — treating alpha=0 (no setup) as if it were ACTION_HOLD",
     "fix": "alpha=0 means 'that strategy has no setup right now' (alpha-space); ACTION_HOLD means 'the policy "
            "chose not to trade this step' (action-space). They share the integer 0 but are DIFFERENT spaces — the "
            "env must NEVER force HOLD just because the alphas are 0; the policy decides the action independently. "
            "If the bot never trades, look at the open-gate (5m CCI threshold) / day-lock / two-phase bank, NOT the "
            "alphas. See CLAUDE.md 'Alpha-state 0 vs action HOLD'.",
     "refs": "CLAUDE.md, src/strategies/base.py, src/env/trading_env.py"},
    {"id": "train-eval-mismatch", "area": "training",
     "symptom": "the model trades well in training but poorly in eval",
     "cause": "eval did not load the saved VecNormalize stats, so the observation is mis-scaled",
     "fix": "use trainer.load_for_eval (training=False) which loads the frozen mean/std saved next to the "
            "model; never eval a normalized-trained model on raw obs.",
     "refs": "src/training/trainer.py (load_for_eval, _vecnorm_path)"},
    {"id": "train-which-policy", "area": "training",
     "symptom": "I have several trained models — which one do I trust?",
     "cause": "comparing models across different environments (different fingerprints)",
     "fix": "the best policy = highest walk-forward PASS-RATE among non-rejected runs AT THE SAME "
            "env_fingerprint(). A different fingerprint means the environment changed and the runs are NOT "
            "comparable. Use run_log.best_run(fingerprint=env_fingerprint()).",
     "refs": "src/training/run_log.py, src/training/env_fingerprint.py, docs/TRAINING_LEDGER.md"},
    {"id": "train-fingerprint-changed", "area": "training",
     "symptom": "the env fingerprint changed after I edited something",
     "cause": "you changed the obs contract, the alpha roster, the FTMO rules, or the reward",
     "fix": "that is correct and intended — it marks a NEW experiment line. Log the run with the new "
            "fingerprint; don't compare its pass-rate against runs from the old fingerprint.",
     "refs": "src/training/env_fingerprint.py, docs/ENVIRONMENT_STATE.md"},
    {"id": "train-nan-obs", "area": "training",
     "symptom": "NaNs in the observation early in an episode",
     "cause": "indicators are NaN during warmup (before enough bars exist)",
     "fix": "this is normal — alphas return 0 during warmup and the env starts at `warmup`. Ensure warmup "
            "is large enough for the slowest indicator (e.g. higher-TF ADX needs many bars). Don't train on warmup bars.",
     "refs": "src/env/trading_env.py (warmup), src/indicators/base.py"},
    {"id": "train-multi-symbol-specializes", "area": "training",
     "symptom": "training on multiple symbols, the bot only trades the easy one",
     "cause": "no incentive to generalize; sizes/rewards not comparable across assets",
     "fix": "use make_multi_symbol_vec_env (per-asset CALIBRATED size so ~one daily range ≈ +2.5% on every "
            "asset) and the cross-asset observation features (ATR-normalized, asset-class one-hot) so one "
            "policy can compare opportunities in common units.",
     "refs": "src/training/vector_env_factory.py, config/asset_specs.py"},
    {"id": "train-oom", "area": "training",
     "symptom": "out-of-memory / crash on Colab during training",
     "cause": "too many parallel envs × large precomputed arrays (or a huge alpha table)",
     "fix": "fewer envs or a smaller window; turn on High-RAM; as alphas scale, store alpha tables as int8 + "
            "share one precomputed table across envs (the road-to-1000-alphas plan).",
     "refs": "config/training_speed_config.py, docs/ENVIRONMENT_STATE.md (Scaling alphas)"},

    # ---------------- TRADING / FTMO ----------------
    {"id": "trade-breach", "area": "trading",
     "symptom": "the account breached the daily-loss or max-drawdown wall",
     "cause": "lost more than 5% in a day / 10% total before the protective stop engaged",
     "fix": "the self-imposed 4% trailing wall should trip BEFORE FTMO's 5/10%. Check FTMO_TRAILING_ENABLED "
            "and the trailing pct in variables.py; size down; JARVIS posture goes STAND DOWN when headroom is "
            "thin. Protecting the challenge always beats one trade.",
     "refs": "config/variables.py (FTMO_TRAILING_*), src/risk/breach_detector.py"},
    {"id": "trade-pace", "area": "trading",
     "symptom": "not reaching +2.5% per day / behind pace to pass",
     "cause": "size too small relative to the account, or too few quality setups",
     "fix": "calibrate lots so ~one daily range ≈ +2.5% of INITIAL (config/asset_specs.calibrated_position_size) "
            "while a full adverse day stays < 4%; raise selectivity to higher-consensus setups rather than "
            "over-trading. JARVIS will say the next step in its ruling.",
     "refs": "config/asset_specs.py, src/jarvis/consistency.py"},
    {"id": "trade-gave-it-back", "area": "trading",
     "symptom": "hit the daily target then gave the gains back",
     "cause": "did not bank and stop — kept trading after +2.5%",
     "fix": "the two-phase engine: hit +2.5% of initial -> CLOSE ALL & BANK & stop for the day (FTMO_TWO_PHASE_"
            "ENABLED). If you opt to continue, a tight 1% trailing protects the banked gain. Consistency is "
            "built by NOT giving it back.",
     "refs": "config/variables.py (FTMO_TWO_PHASE_*, FTMO_PHASE2_*), src/env/trading_env.py"},
    {"id": "trade-no-opens", "area": "trading",
     "symptom": "the bot won't open new trades",
     "cause": "the 5m CCI open-gate is blocking (neutral market), or the day is locked after banking",
     "fix": "the open-gate forbids NEW directional opens when EITHER 5m CCI is within ±threshold; it's a "
            "feature (avoid chop). Tune OPEN_GATE_CCI_THRESHOLD or turn the gate off. After banking the daily "
            "target the day is locked (no new opens until tomorrow) — also intended.",
     "refs": "config/variables.py (OPEN_GATE_CCI_THRESHOLD), src/env/trading_env.py (open_gate, _day_locked)"},
    {"id": "trade-size-per-asset", "area": "trading",
     "symptom": "position size seems wrong on indices/gold vs forex",
     "cause": "one fixed lot size used across assets — wrong, because value-per-point differs hugely",
     "fix": "size PER ASSET: PnL = position × price_move × contract_size. Use "
            "asset_specs.calibrated_position_size(symbol); value_per_point differs (EURUSD 100000, XAUUSD 100, US30 1).",
     "refs": "config/asset_specs.py (value_per_point, calibrated_position_size)"},
    {"id": "trade-blew-account", "area": "trading",
     "symptom": "a single bad day wiped a big chunk of the account",
     "cause": "lots too large for the asset's volatility",
     "fix": "calibrate so ~one typical daily range ≈ +2.5% and a full ADVERSE day stays under the 4% wall. "
            "If a day can lose >4%, the lots are too big — shrink them.",
     "refs": "config/asset_specs.py, config/variables.py"},
    {"id": "trade-ny-bonus", "area": "trading",
     "symptom": "the NY index bonus didn't pay",
     "cause": "the bonus has strict conditions",
     "fix": "it only applies to INDICES, on CLOSED-IN-PROFIT P&L, during the NY session (open 13:30 UTC), "
            "reaching ≥50% (within 2h) / ≥100% (within 3h) of the daily target — and it is PAID AT DAY-END "
            "ONLY IF THE DAY PASSED (≥+2.5% of initial). It is erased on a failed day or a breach.",
     "refs": "config/variables.py (FTMO_NY_*), src/env/trading_env.py (_ny_qualify, _ny_day_end_bonus)"},
    {"id": "trade-gate-looks-bullish", "area": "trading",
     "symptom": "a movement-filter alpha shows as a BUY / inflates the bullish count",
     "cause": "a non-directional GATE outputs 1 (=movement on), which looks like a +1 buy",
     "fix": "gates are EXCLUDED from the directional consensus (registry.directional_mask). The bridge sends "
            "the directional-only net_signal + net_signal_basis; the HUD must use those, never divide by a "
            "hardcoded 15. A gate's 1 means 'the market is moving', not 'buy'.",
     "refs": "src/signals/signal_summary.py, src/jarvis/state_provider.py (directional_mask)"},
    {"id": "trade-no-model", "area": "bridge",
     "symptom": "JARVIS / the cockpit says model_attached: false",
     "cause": "no trained policy is attached, so the action is an honest alpha-consensus fallback",
     "fix": "train a model (or load one) and attach it: StateProvider.from_cache(dir, symbol, "
            "policy=sb3_policy_fn(*load_for_eval(...))). Until then JARVIS reports confidence 0 and flags the "
            "policy fields as fallback — it never fabricates a decision.",
     "refs": "src/jarvis/state_provider.py (from_cache), src/training/trainer.py (load_for_eval, sb3_policy_fn)"},
    {"id": "trade-synthetic-data", "area": "bridge",
     "symptom": "the cockpit shows movement but it's not my real market data",
     "cause": "the provider is running from_synthetic (placeholder prices)",
     "fix": "switch to real data: StateProvider.from_cache(your_cache_dir, symbol). Build the cache first with "
            "src/data/cache_builder.build_cache on your 1m OHLCV.",
     "refs": "src/jarvis/state_provider.py (from_cache), src/data/cache_builder.py"},

    # ---------------- DATA ----------------
    {"id": "data-cache-fail", "area": "data",
     "symptom": "building the cache fails / wrong columns / no close",
     "cause": "the CSV columns weren't recognized",
     "fix": "load_ohlcv_csv accepts flexible names (datetime/timestamp/date+time; open/high/low/close/volume "
            "with common aliases). You at least need a datetime column and a close. Check the CSV header.",
     "refs": "src/data/cache_builder.py (load_ohlcv_csv)"},
    {"id": "data-mt5-format", "area": "data",
     "symptom": "training crashes with 'no datetime column found' / MetaTrader (MT5) export won't load / "
                "header looks like <DATE>\\t<TIME>\\t<OPEN>... / only one column / 1 bar per day",
     "cause": "MT5 history exports are TAB-separated with angle-bracket headers and SPLIT <DATE>/<TIME> columns "
              "and dotted dates (2021.01.13) — old loaders assumed a comma file with one datetime column",
     "fix": "the loader now auto-detects comma/TAB/semicolon delimiters, strips <...> from headers, COMBINES "
            "split <DATE>+<TIME> (keeping 1-minute resolution), parses dotted MT5 dates, and uses TICK volume "
            "(<VOL> is usually 0 on forex). Just point run_training at the folder and re-run. If it still fails, "
            "open the first row and confirm it has O/H/L/C plus a date and time.",
     "refs": "src/data/cache_builder.py (load_ohlcv_csv), tests/test_csv_loader.py"},
    {"id": "data-symbol-unknown", "area": "data",
     "symptom": "a symbol's sizing/asset-class is wrong or unknown",
     "cause": "the symbol isn't in SPECS / not classified",
     "fix": "add it to config/asset_specs.SPECS (contract_size, pip, typical_daily_range, asset_class) or rely "
            "on infer_asset_class for the class. Unknown symbols fall back to a sane default size.",
     "refs": "config/asset_specs.py (SPECS, infer_asset_class)"},

    # ---------------- BRIDGE / JARVIS ----------------
    {"id": "bridge-wont-start", "area": "bridge",
     "symptom": "uvicorn jarvis_bridge:app fails / FastAPI not found",
     "cause": "the optional web deps aren't installed",
     "fix": "pip install -r requirements-jarvis.txt (fastapi, uvicorn). The contract + council logic need none "
            "of these and are tested without them (python tools/run_tests.py).",
     "refs": "requirements-jarvis.txt, jarvis_bridge.py"},
    {"id": "bridge-hud-blank", "area": "bridge",
     "symptom": "the HUD is blank or 404 at /JARVIS Cockpit.dc.html",
     "cause": "the HUD files aren't in the repo root, or pullLive() isn't wired",
     "fix": "drop JARVIS Cockpit.dc.html + support.js into the repo root, then apply the 4-method patch in "
            "docs/JARVIS_LIVE_WIRING.md (pullLive + councilLive + the net-signal/gate fix).",
     "refs": "docs/JARVIS_LIVE_WIRING.md"},
    {"id": "bridge-generic-advice", "area": "bridge",
     "symptom": "JARVIS's advice feels generic",
     "cause": "no LLM key set, so the council uses its deterministic (still system-grounded) text",
     "fix": "set ANTHROPIC_API_KEY (and pip install anthropic) on the bridge host to enable the LLM layer "
            "(claude-opus-4-8). Without it the council is still grounded + progressive, just less conversational.",
     "refs": "src/jarvis/council.py (llm_available), requirements-jarvis.txt"},
    {"id": "bridge-readonly", "area": "bridge",
     "symptom": "I can't make JARVIS place or close a trade from the cockpit",
     "cause": "JARVIS is READ-ONLY by design",
     "fix": "that's intentional and structural — the bridge has GET routes only (POST/PUT/PATCH/DELETE return "
            "405). JARVIS observes, reasons and advises; he never touches the trading code or places orders.",
     "refs": "jarvis_bridge.py"},

    # ---------------- PORTFOLIO / POLICIES / HEATMAP ----------------
    {"id": "portfolio-trader", "area": "trading",
     "symptom": "is the bot a single-asset trader, or does it trade everything at once?",
     "cause": "you may not know the shared-pot portfolio trainer exists",
     "fix": "it is a PORTFOLIO trader — ONE shared equity/drawdown pot across the WHOLE FTMO universe. Train it "
            "with train_portfolio on the shared-pot PortfolioEnv: one policy holds SIMULTANEOUS positions across "
            "ALL symbols in one account, decides one symbol at a time while seeing the pot's exposure, and is "
            "rewarded on the pot — so it learns to BALANCE risk. Because decisions are per-symbol with portfolio "
            "context, the obs stays 499 and it scales to the full FTMO broker list live. See "
            "docs/TRAINING_INSTRUCTIONS.md; watch the live book on the heatmap tab.",
     "refs": "src/env/portfolio_env.py, src/training/trainer.py (train_portfolio), docs/TRAINING_INSTRUCTIONS.md"},
    {"id": "policy-organize", "area": "training",
     "symptom": "I have several policies — which do I run, and how do I keep them organized?",
     "cause": "no single ranked view of policies by how consistently they pass",
     "fix": "register each trained model in the policy registry; JARVIS ranks them by a CONSISTENCY score "
            "(walk-forward pass-rate, low max-DD, low day-to-day concentration). champion() = the one to run, "
            "and only policies at the SAME env fingerprint are comparable. Ask JARVIS 'which policy should I run?'",
     "refs": "src/jarvis/policy_registry.py, src/training/run_log.py, src/training/env_fingerprint.py"},
    {"id": "policy-add", "area": "training",
     "symptom": "how do I add a new policy so JARVIS knows about it?",
     "cause": "the policy isn't registered yet",
     "fix": "one line: `python -m src.jarvis.policy_registry add --id my-policy --path models/camillion_ppo "
            "--fingerprint <fp> --pass-rate 0.8 --max-dd 3.5 --largest-day 30`, or policy_registry.add_policy(...). "
            "It appears in GET /policies and in JARVIS's context immediately.",
     "refs": "src/jarvis/policy_registry.py"},
    {"id": "heatmap-tab", "area": "bridge",
     "symptom": "where is the market heatmap / how do I see all symbols at once?",
     "cause": "the heatmap is its own cockpit tab fed by a separate endpoint",
     "fix": "GET /heatmap returns the buy/sell signal of EVERY FTMO symbol (direction, strength, buy/sell %, "
            "hottest alpha) — its own tab. Wire the tab per docs/JARVIS_LIVE_WIRING.md; the same rows are also "
            "on /state as `heatmap`.",
     "refs": "jarvis_bridge.py (/heatmap), src/jarvis/market_view.py, docs/JARVIS_LIVE_WIRING.md"},
    {"id": "see-daily-results", "area": "training",
     "symptom": "how do I see day-by-day results — did I make +2.5% of initial and stay inside the trailing DD?",
     "cause": "a final equity number hides whether you passed CONSISTENTLY",
     "fix": "run the day-by-day report: `python -m src.training.daily_report --data data_cache --symbol "
            "EURUSD --model models/camillion_ppo`. It prints one row per day — P&L%, +TGT? (made +2.5% of "
            "initial), TRAIL_DD% + <WALL? (inside the 4% trailing wall), DAILY_LOSS%, BREACH, cumulative % — "
            "plus a PASS/FAIL summary. Full steps in docs/TRAINING_INSTRUCTIONS.md.",
     "refs": "src/training/daily_report.py, docs/TRAINING_INSTRUCTIONS.md"},
    {"id": "train-all-symbols-balance", "area": "training",
     "symptom": "should I train per symbol, or on all four at once?",
     "cause": "training one symbol at a time can't learn to balance the portfolio",
     "fix": "train on ALL FOUR Google-Drive symbols TOGETHER with train_multi_symbol — one policy over "
            "EURUSD/GBPUSD/XAUUSD/US30 with per-asset calibrated size + the cross-asset features. That is how "
            "the bot learns to BALANCE the book (read every asset in common units, allocate risk across them) "
            "instead of overfitting the easiest one. Mount Drive, build a cache per symbol, then "
            "train_multi_symbol({sym: load_cache(...)}, ...).",
     "refs": "src/training/trainer.py (train_multi_symbol), src/training/vector_env_factory.py, docs/TRAINING_INSTRUCTIONS.md"},
    {"id": "how-to-train-one-command", "area": "training",
     "symptom": "how do I train the bot? I don't trade and I don't want a bunch of confusing steps",
     "cause": "training used to be several manual steps",
     "fix": "ONE command: put your four 1-minute CSVs (filenames containing EURUSD/GBPUSD/XAUUSD/US30) in "
            "one folder, then run `python run_training.py --data <that_folder>`. It finds the files, prepares "
            "the features, trains ONE bot on all four from one shared account, prints the DAY-BY-DAY +2.5% / "
            "4%-trailing results, and files the policy. In Colab it's one cell: "
            "`!python run_training.py --data /content/drive/MyDrive/Camillion_data`. Install the engine once "
            "with `pip install stable-baselines3 torch`.",
     "refs": "run_training.py, docs/TRAINING_INSTRUCTIONS.md"},
]


def _score(entry: dict, terms: list[str]) -> int:
    # weight the SYMPTOM + id/area heavily; prefix-match so breaching~breach~breached all hit.
    fields = ((entry["symptom"], 4), (entry["id"], 3), (entry["area"], 2),
              (entry["cause"], 1), (entry["fix"], 1))
    score = 0
    for text, w in fields:
        words = re.findall(r"[a-z0-9]+", text.lower())
        for t in terms:
            t5 = t[:5]
            if any(wd.startswith(t5) or t.startswith(wd[:5]) for wd in words):
                score += w
    return score


def search(query: str, k: int = 6) -> list[dict]:
    """Keyword-rank the troubleshooting entries for a question (best first)."""
    terms = [t for t in re.findall(r"[a-z0-9]+", (query or "").lower()) if len(t) > 2]
    if not terms:
        return list(TROUBLESHOOTING[:k])
    scored = sorted(TROUBLESHOOTING, key=lambda e: _score(e, terms), reverse=True)
    hits = [e for e in scored if _score(e, terms) > 0]
    return (hits or list(TROUBLESHOOTING))[:k]


def as_context(query: str | None = None, k: int = 6) -> str:
    """A compact string for the LLM: the system summary + the most relevant fixes."""
    picks = search(query, k) if query else TROUBLESHOOTING[:k]
    lines = ["SYSTEM SUMMARY:", SYSTEM_SUMMARY, "", "RELEVANT FIXES (grounded in the repo):"]
    for e in picks:
        lines.append(f"  [{e['area']}] {e['symptom']} -> {e['fix']}  (see {e['refs']})")
    return "\n".join(lines)


def render_markdown() -> str:
    """Render the full troubleshooting guide (single source for docs/TROUBLESHOOTING.md)."""
    by_area: dict[str, list[dict]] = {}
    for e in TROUBLESHOOTING:
        by_area.setdefault(e["area"], []).append(e)
    titles = {"training": "Training problems", "trading": "Trading / FTMO problems",
              "data": "Data & cache problems", "bridge": "Cockpit / JARVIS problems", "obs": "Observation-contract problems"}
    out = ["# Common Problems & Fixes (training + trading)",
           "",
           "> Generated from `src/jarvis/knowledge.py` — the SAME knowledge JARVIS uses. Ask JARVIS "
           "\"how do I fix <X>?\" in the cockpit and he answers from this, grounded in the real system.",
           "", SYSTEM_SUMMARY, ""]
    for area in ("training", "trading", "data", "obs", "bridge"):
        if area not in by_area:
            continue
        out.append(f"## {titles.get(area, area)}")
        out.append("")
        for e in by_area[area]:
            out.append(f"### {e['symptom']}")
            out.append(f"- **Likely cause:** {e['cause']}")
            out.append(f"- **Fix:** {e['fix']}")
            out.append(f"- **Where:** `{e['refs']}`")
            out.append("")
    return "\n".join(out)
