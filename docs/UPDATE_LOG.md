# UPDATE LOG (IRAC)

Every change appends a dated IRAC entry. **Conclusion** states why it helps the bot
pass FTMO-style challenges more consistently.

## [2026-06-26] Portfolio cockpit: market heatmap + policy registry JARVIS organizes by consistency
- **I (Issue):** The bot is a PORTFOLIO trader (one pot, the whole FTMO universe at once), the
  cockpit needs a market heatmap as its own tab, we must be able to easily add a policy, and JARVIS
  must know each policy's detail relative to passing the FTMO challenge consistently.
- **R (Rule):** Operator clarification, 2026-06-26 — portfolio not single-asset; heatmap tab; easy
  policy add; JARVIS organizes policies by consistency.
- **A (Application):** `src/jarvis/market_view.py` (a read-only StateProvider per symbol -> the
  full-universe buy/sell heatmap + per-symbol positions + a portfolio view; honest that the shared-pot
  ENV is the next build). `src/jarvis/policy_registry.py` (persistent JSON registry: add_policy/list/
  champion/set_status, ranked by a CONSISTENCY score = pass-rate + low max-DD + low day concentration;
  CLI `python -m src.jarvis.policy_registry add ...`). The council + JARVIS prompts now carry the
  market summary + the policy roster; `answer()` handles "which policy should I run?". Bridge gains
  GET /heatmap + GET /policies and /state gains universe/positions/portfolio/heatmap; go_live.py is
  portfolio-first (`--symbols`). Knowledge + the two guides + the HUD wiring patch updated. +6 tests,
  140/140 green; still structurally read-only.
- **C (Conclusion):** One cockpit shows the whole FTMO book and one ranked, JARVIS-curated view of
  which policy passes most consistently — the operator picks the right policy and reads the whole
  market at a glance, all toward a consistent portfolio pass.

## [2026-06-26] Operator manuals (REPO_GUIDE, JARVIS_GUIDE) + JARVIS troubleshooting brain
- **I (Issue):** Monty wants an extremely detailed guide to how the whole folder works and how
  JARVIS works, a common-problems-and-fixes section for training & trading, and that knowledge placed
  where JARVIS ALWAYS has access so he can be asked directly how to fix any issue.
- **R (Rule):** That request, 2026-06-26 — the fixes must be grounded in the real system (no guessing),
  and JARVIS must carry them in context every deliberation.
- **A (Application):** `docs/REPO_GUIDE.md` (every folder/module, the data->decision pipeline, how to
  run, the locked invariants) and `docs/JARVIS_GUIDE.md` (the cockpit, the council, every endpoint,
  going live, how to ask for fixes), authored from a parallel directory-mapping pass. New
  `src/jarvis/knowledge.py` = the always-on knowledge base (system summary + ~30 grounded training/
  trading/data/bridge fixes + ranked search) wired into the council context + every JARVIS prompt;
  `council.answer()` + bridge `GET /ask` and `GET /knowledge` (read-only); `docs/TROUBLESHOOTING.md`
  generated from that single source. 5 new tests (ranking, well-formed entries, council carries
  knowledge, ask returns the right grounded fix). 136/136 green.
- **C (Conclusion):** Monty can read the whole system or just ask JARVIS "how do I fix X?" and get a
  system-correct, file-specific answer that always points at the next step toward a consistent pass.

## [2026-06-26] JARVIS live bridge + grounded, progressive multi-agent COUNCIL
- **I (Issue):** Wire the JARVIS cockpit to the real bot (read-only), and make the LLM agents
  (OMEGA/JUSTICE/JARVIS) reason from the live system + chat history, talk to each other, and always
  advise the next improvement toward passing CONSISTENTLY — grounded in the system's logic.
- **R (Rule):** HANDOFF data contract + "never fabricate / safe-default + flag" + read-only; operator
  2026-06-26 emphasis on the LLMs' info, chat history, agent-to-agent reasoning, and a always-progressive view.
- **A (Application):** Pure `src/jarvis/state_contract.build_state` (the exact /state, CLOSE folded into
  HOLD, gaps flagged, directional-only `net_signal` + basis so the HUD never divides by a hardcoded 15);
  `src/jarvis/state_provider` (headless env+policy snapshot, honest no-model alpha fallback, defensive
  directional mask, age/day_history tracking); `src/jarvis/consistency.analyze_consistency` (the
  system-logic the agents cite: pace, breach headroom, binding constraint, p(pass), and ALWAYS a
  progressive next step); `src/jarvis/council.deliberate` (OMEGA→JUSTICE→JARVIS each see the full
  grounded context + chat history + the prior speakers; deterministic core + optional Anthropic LLM,
  always progressive); `jarvis_bridge.py` (lazy-FastAPI, GET /state + /council + /health, structurally
  read-only — POST /order → 405). `docs/JARVIS_LIVE_WIRING.md` patches the HUD (pullLive + councilLive +
  the net-signal/gate fix). 24 deep tests + a live HTTP run; obs/FTMO/contract untouched.
- **C (Conclusion):** Monty gets a live cockpit and a council that reasons from the real system and the
  conversation, never fabricates, and always points at the next gain toward a CONSISTENT FTMO pass —
  while being structurally unable to place a trade.

## [2026-06-26] Made VERSION PAIRING the governing rule for CPU/GPU/TPU
- **I (Issue):** With three implementations (CPU, GPU, TPU) we could get confused about which
  produced a policy and whether they're comparable.
- **R (Rule):** Operator decision — they are ONE bot written three ways: same contract version,
  same fingerprint, same behaviour, same policy format; only the code differs. One shared version
  number; any behaviour change bumps all three together in the same PR.
- **A (Application):** Made version pairing the governing rule of §2 in
  `docs/JAX_GPU_TPU_TRAINER_BLUEPRINT.md` and added it to §4 of `docs/ENVIRONMENT_STATE.md`.
  Docs only.
- **C (Conclusion):** A policy is identified by version+fingerprint, never by machine — so all
  three are ranked in one ledger with zero confusion, on the road to a consistent FTMO pass.

## [2026-06-26] Added the full-rewrite JAX GPU/TPU trainer blueprint
- **I (Issue):** We want a future path to run vast data through thousands of parallel sims until
  the bot passes FTMO consistently, with runtime-changeable target/risk and a pass-likelihood
  readout — without losing the locked obs contract / FTMO numbers / fingerprint parity.
- **R (Rule):** A from-scratch on-device JAX/Flax rewrite (co-location) unlocks GPU/TPU, but it
  must be a *second implementation of the same env* — same observation (v1.5.0/479), same FTMO
  numbers, same fingerprint, step-parity vs the CPU reference, same policy format.
- **A (Application):** Wrote `docs/JAX_GPU_TPU_TRAINER_BLUEPRINT.md` (goal, honest cost,
  non-negotiable invariants, co-location architecture, the 5 rebuild rules, runtime target/risk
  via % features + domain randomization, pass-likelihood grid, training loop, build order,
  when-not-to). Cross-linked from `ENVIRONMENT_STATE.md` §4. Docs only; no code/obs change.
- **C (Conclusion):** Captures the high-throughput path (play thousands of trading lifetimes at
  once, dial risk live, read the odds of passing) while guaranteeing it can never drift from the
  CPU reference — so scaling speed never costs us a consistently-passing policy.

## [2026-06-26] Recorded the GPU-trainer learning principle (data-parallel RL)
- **I (Issue):** When we build the GPU trainer, a future agent must understand WHY the GPU runs
  thousands of sims in lockstep — and that "all do the same thing" does not defeat learning.
- **R (Rule):** Operator-confirmed principle — ONE shared policy learns from thousands of sims
  running the same math on DIFFERENT market data (different experiences pooled into one update);
  the rewrite's hard part is turning branchy FTMO logic into lockstep mask/array math; output is
  the same policy file as the CPU trainer.
- **A (Application):** Added the principle to §4 of `docs/ENVIRONMENT_STATE.md` and to the
  "RULES FOR THE FUTURE GPU TRAINER" header in `src/training/env_fingerprint.py`. Comments only.
- **C (Conclusion):** Locks the intended GPU design so whoever builds it gets 10–100× more varied
  experience per unit time into one bot — faster path to a consistently FTMO-passing policy.

## [2026-06-26] Documented the alpha-scaling logic so the obs stays stable forever
- **I (Issue):** The plan is to grow toward ~1000 alphas. Future agents must not
  destabilise the locked observation (or wrongly "fix" empty slots by reshaping it
  or aggregating away per-alpha weighting).
- **R (Rule):** Operator decision — keep per-slot (policy learns a weight per alpha);
  empty slots don't hurt learning, only memory; beat memory with int8 + a shared
  precomputed table; raising `MAX_STRATEGIES` is a deliberate contract bump.
- **A (Application):** Wrote the logic where an agent will read it — a comment block at
  `MAX_STRATEGIES` in `config/constants.py`, a new "Scaling alphas" rule in `CLAUDE.md`,
  and §3 in `docs/ENVIRONMENT_STATE.md`. Docs only; no code/obs change.
- **C (Conclusion):** Locks the design intent so the observation contract survives the
  road to 1000 alphas — one policy keeps training as the library grows, the prerequisite
  for repeatedly passing FTMO.

## [2026-06-21] Phase 0 — bare-bones framework initialized
- **I (Issue):** Need a fresh, modular RL framework where strategies are alphas the
  agent learns to combine, with a fixed observation that never breaks when strategies
  are added, plus FTMO logic carried over from Quantra.
- **R (Rule):** Build spec (Camillion Phase 0) + Quantra FTMO numbers (2.5% / 4% /
  two-phase) + Monty's corrections (SMA p1–p4 shifts; CCI/RSI raw+shifted) + hybrid
  observation (raw indicators **and** alphas).
- **A (Application):** Frozen contract (357 float32) in `constants.py`; indicator
  registry (190); 64-slot `StrategyRegistry` + occupancy mask; signal summary/memory/
  accuracy (no leakage); account/risk scaffolds; observation builder; 28 tests (all
  passing); importable Jarvis + Barbershop stubs; 7 docs.
- **C (Conclusion):** A locked, scale-stable observation lets one policy keep training
  as the alpha library grows — the prerequisite for repeatedly passing FTMO.

## [2026-06-21] Risk knobs made runtime-editable (no retrain)
- **I:** Monty must change target / trailing-DD / trailing on-off in BOTH modes
  without retraining.
- **R:** Operator note + the percentage-feature design.
- **A:** Moved FTMO target/trailing/toggle into `variables.py`; `ftmo_config.py`
  builds configs from them; added `update_risk_settings(...)` live mutator; account
  features divide by the active config's (editable) limits.
- **C:** One trained model is reusable across many challenge configurations → far
  faster iteration toward a stable pass rate.

## [2026-06-21] Phase 1 start — real indicators + ATR added (contract v1.1.0)
- **I:** Stubs returned NaN; Monty added ATR-14 (raw + SMA2-shift4) per timeframe.
- **R:** Phase-1 spec (real indicators) + operator request 2026-06-21 + CCI/RSI raw+shifted pattern.
- **A:** Real RSI/CCI/Bollinger/ATR (pandas, TA-Lib optional); NaN-aware SMA; ATR adds
  +2 cols/TF -> indicator block 190->200, observation **357->367**, contract **v1.1.0**.
  Added 3 example alphas (SMA-trend, RSI-reversion, Bollinger-breakout).
- **C:** The policy now sees real multi-timeframe indicators incl. volatility and its
  4-bar slope — richer, leak-free context for trading within FTMO drawdown walls.

## [2026-06-21] Locked strategy/alpha/policy contract
- **I:** 'strategy' and 'alpha' risked being conflated; observation must expose
  only alpha OUTPUTS, never strategy internals.
- **R:** Operator contract (strategy=logic, alpha=exposed output, policy=RL).
- **A:** Renamed StrategyRegistry->AlphaRegistry, collect_signals->collect_alphas
  (aliases kept), example classes ->...Strategy; locked semantics in docstrings +
  OBSERVATION_CONTRACT.md. Confirmed 6 SMA obs lines/TF (1/s0,2/s1,3/s2,4/s3,50/s0,
  200/s0) and that the observation exposes only alpha_values/mask/summary.
- **C:** The policy can't 'see the strategy'; clean meta-learning over alphas.

## [2026-06-21] Phase 1 — interpretability + alpha/policy diagnostics
- **I:** Need to see what the PPO is thinking, track alpha vs policy reliability,
  and detect leader-chasing — without observation bloat, leakage, or reward shortcut.
- **R:** Operator diagnostics spec 2026-06-21.
- **A:** Leak-free per-alpha 1/3/10 accuracy + counts; aggregate reliability
  (mean/best/dispersion); policy directional accuracy (same primitive);
  PolicyIntrospector (action dist + value + entropy + block-ablation saliency);
  Policy Doctor (scoreboard, explicit leader-chasing test, best-alpha comparison,
  block importance). All diagnostics-only; reward untouched. +7 tests (35/35).
- **C:** We can prove whether the policy is a real meta-learner over alphas or a
  wrapper around the best recent signal — the core risk in this architecture.

## [2026-06-21] Phase 1 complete — cache, env, trainer, eval
- **I:** Need leakage-safe cache, FTMO env with clean reward, trainer, and
  read-only diagnostics in eval.
- **R:** Operator guardrails 2026-06-21 (reward objective-only, no leakage,
  eval separation) + Phase-1 spec.
- **A:** Leak-free multi-TF cache (last-closed-bar alignment); TradingEnv
  (reward = equity change only, proven alpha-independent; per-day FTMO reset;
  breach terminate; two-phase); PPO trainer (Colab); read-only evaluate harness
  wiring introspection + Policy Doctor. 43/43 tests. Docs: PHASE1_REPORT.md.
- **C:** A fast, honest, FTMO-aligned training loop whose policy we can actually
  interpret and audit for shortcut/leader-chasing.

## [2026-06-21] Phase 2 — Barbershop suite + walk-forward + cockpit
- **I:** Need the rest of the Barbershop diagnostics, a real FTMO pass-rate, and
  the Jarvis cockpit in the repo.
- **R:** Phase-2 spec (Jarvis UI + Barbershop) + 'pass-rate first'.
- **A:** Day Replay, Trade Autopsy, Signal Doctor (real, tested); walk-forward
  validation harness (rolling windows -> per-window pass/breach -> pass-rate,
  leak-safe + read-only); standalone 0_JARVIS_COCKPIT.html (voice + mic-reactive
  + clap + live brief). 48/48 tests.
- **C:** The Barbershop can fully audit a run, and we can put an honest pass-rate
  number on the policy over unseen walk-forward windows.

## [2026-06-25] Fix — realized-PnL double-count + walk-forward pass threshold (on 451-obs v1.2.0)
- **I (Issue):** Two correctness bugs survived into the 14-alpha v1.2.0 (451-obs) base:
  1. **Realized PnL was double-counted.** In `env.step()` the realize block added
     `realized` to balance/daily/episode AND then called `record_close()`, which adds the
     same three again; the two-phase auto-flat block double-added balance. Every closed
     trade moved the account by **2x** its true PnL — corrupting equity, reward, the daily
     accounting, and EVERY FTMO breach/target check.
  2. **walk_forward measured the wrong pass threshold** — it scored a window "passed" at
     **+2.5%** (the DAILY target) instead of the **+10%** challenge target.
- **R (Rule):** CLAUDE.md FTMO numbers untouched; obs shape (451, v1.2.0) untouched;
  reward = equity-change only; "pass-rate first". Entry/exit transaction costs preserved.
- **A (Application):**
  - `record_close()` is now the SINGLE source of truth for balance/daily/episode realized
    PnL + equity + tallies. Removed the manual `+=` lines in BOTH the realize block and the
    two-phase auto-flat block. The one-time entry-cost `-= ecost` and the exit-cost baked
    into `realized` are unchanged (a round trip still pays both sides).
  - `walk_forward.run()` now scores a pass as "env set `episode_passed` (+10% reached) OR
    final return >= target, with NO breach"; default threshold resolves to
    `cfg.profit_target_total_pct` (+10%). `target_pct` still overridable. Detail gains `hit_target`.
  - +2 tests (`tests/test_no_double_count.py`). NOTE: per-step reward is now ~2x smaller than
    before the fix — that is correct (the old reward was inflated by the double-count); do
    NOT re-inflate it via position_size.
- **C (Conclusion):** Equity, reward, and every FTMO check now run on arithmetically-correct
  money, and the walk-forward scoreboard measures passing at the real +10% challenge target.

## [2026-06-25] PPO wiring hardening — eval callback, random-window control, real learn-check
- **I (Issue):** PPO/MLP training wiring had three reliability gaps: (1) `train()` accepted
  `eval_env` but never used it, so learning regressions were invisible during training;
  (2) vec-env factory hardcoded `random_window=True` and ignored
  `training_speed_config.RANDOM_WINDOW_TRAINING`; (3) the overfit test only checked finite
  outputs after `learn()`, which could pass even if the policy learned nothing.
- **R (Rule):** Keep reward objective-only, keep the observation contract unchanged, and make
  training diagnostics verify *actual* learning behavior.
- **A (Application):**
  - `src/training/trainer.py`: wired optional SB3 `EvalCallback` into both `train()` and
    `resume()` when `eval_env` is provided; added `eval_freq` override (defaults to one PPO
    rollout horizon).
  - `src/training/vector_env_factory.py`: random-window flag now defaults from
    `RANDOM_WINDOW_TRAINING`, with caller override via `env_kwargs`, and duplicate-keyword
    collisions are prevented by popping `random_window` before env construction.
  - `tests/test_single_batch_overfit.py`: upgraded from "finite output" to a deterministic
    trend overfit harness that asserts post-train deterministic episode return is higher than
    pre-train return.
- **C (Conclusion):** PPO setup is now better wired for detectable improvement and easier
  to control/reproduce, reducing the risk of shipping a policy that only appears to train.

## [2026-06-25] Feature — configurable 5m CCI open-gate threshold
- **I:** The open-gate (block new opens unless BOTH 5m CCIs are beyond +/-threshold) had
  the threshold hardcoded at 50. Operator wants to set it (e.g. +/-100 = only open on
  stronger momentum) without editing code.
- **R:** Operator request + existing runtime-tunable-knob pattern. Obs shape (451) and FTMO
  numbers untouched; gate still off by default; still computed in _precompute (never in step).
- **A:** Added `OPEN_GATE_CCI_THRESHOLD=50.0` to variables.py and an `open_gate_threshold`
  param on TradingEnv (defaults to the variable, mirrors the `cost_frac` pattern). _precompute
  uses it. +1 test (`test_open_gate_threshold_is_configurable`). 71/71 green.
- **C:** The momentum entry filter is now a dial (50 = original, 100 = stricter) with no retrain.

## [2026-06-25] Feature — two-phase DAILY engine (+2.5%/day of initial -> +10% over ~4 days)
- **I:** Operator's strategy: each day make **+2.5% of the INITIAL balance**, bank it (close
  ALL), and either STOP for the day (default) or optionally CONTINUE under a tight 1% trail.
  ~4 such days ladder to the +10% pass. Phase-1 risk wall = 4% trailing. Current main had
  two-phase + trailing OFF (chase-10%), which is the opposite.
- **R:** Operator directive (matches CLAUDE.md rule #2 "+2.5% -> 1% trailing"). Obs shape
  (451) unchanged. EXPLICIT FTMO-behaviour change (re-enables trailing + two-phase).
- **A:**
  - `daily_target_hit` now = the DAY's gain on **EQUITY** (open profit incl.) >= 2.5% of the
    **INITIAL** balance (was realized PnL vs day-start). FREE mode + obs target-progress matched.
  - `variables.py`: `FTMO_TRAILING_ENABLED` & `FTMO_TWO_PHASE_ENABLED` -> **True**; new
    `FTMO_PHASE2_CONTINUE=False`. `ftmo_config` carries `phase2_continue`.
  - `TradingEnv`: per-day two-phase state (reset each midnight). Hit +2.5% -> `_flatten()`
    (close all, bank, single source of truth). Default -> `_day_locked` (no new opens till
    tomorrow). If `phase2_continue` -> keep trading under a fresh 1% trailing wall from the
    banked peak; give it back -> bank & lock (NOT a breach). Phase-1 4% trailing stays a breach.
  - +5 tests (`tests/test_two_phase_daily.py`); verified a 5-day run banks ~+2.5%/day and
    PASSES at +10% with no breach. **75/75 green.**
- **C:** The bot now trains under the real daily engine: grind +2.5%/day of initial, protect
  it, ladder to +10% — the disciplined, low-drawdown path to the challenge pass.
## [2026-06-25] Feature — per-asset lot-size calibration (config/asset_specs.py)
- **I:** PnL = position * price_move * position_size, so a single fixed position_size is
  sane for FX (~1.1) but absurd for gold (~2000) / US30 (~40000). And at 1 lot EURUSD you'd
  need ~250 pips for +2.5%/day (impossible). The challenge math was not well-posed.
- **R:** Operator "per-asset conversion + reachable 2.5%/day, safe under 4%"; leverage 1:100.
- **A:** `config/asset_specs.py`: per-asset contract_size + typical_daily_range; helpers
  `value_per_point`, `lots_for_daily_target`, `calibrated_position_size`, `leverage_used`.
  Calibrates each asset so capturing one typical daily range ~= +2.5% and a full adverse day
  stays inside 4%. Table: EURUSD 3.12 lots (3.4x), GBPUSD 2.27 (2.9x), XAUUSD 1.25 (2.5x),
  US30 6.25 (2.5x) -- all << 1:100. +4 tests. 74/74 green.
- **C:** The challenge math is now WELL-POSED per asset; training on real data can actually
  reach the target without instant breaches. Prereq for both real-data training and portfolio.

## [2026-06-25] Contract v1.2.0 -> v1.3.0 — SIZING observation block (461 float32)
- **I:** The bot couldn't see (a) the per-asset $-per-move conversion, (b) how much it still
  needs today, or (c) what different lot sizes would do -- it only learned that from reward.
  Operator wants these as OBSERVATIONS now (sizing still NOT an action yet), relative to the
  INITIAL balance, so the policy learns the size<->risk/reward relationship before it can size.
- **R:** CLAUDE.md rule #1 (deliberate shape bump: version + docs + shape tests). No trained
  model exists yet, so this is the right time. Appended (indices 0..450 unchanged).
- **A:** New 10-float `sizing` block (all fractions of INITIAL balance): 6-rung what-if lot
  ladder (0.01/0.1/0.5/1/2/4 -> account-% a typical move is worth), `daily_target_remaining`,
  `dd_room`, `active_lots_norm`, `active_move_value`. `WL.sizing_features()`; env resolves
  `value_per_point` per asset (asset spec, else position_size=1 lot) + a leak-free `ref_move`
  (recent realized range, pandas in precompute only). Contract -> v1.3.0 / 461; updated
  constants, observation_contract, builder order, OBSERVATION_CONTRACT.md (also corrected the
  stale v1.1.0 doc) and all shape tests (451->461). +6 tests. 80/80 green.
- **C:** The bot now SEES sizing in account terms -- groundwork for the future sizing action
  and for portfolio risk allocation, with the challenge math made well-posed by asset_specs.

## [2026-06-25] Contract v1.3.0 -> v1.4.0 — CROSS-ASSET perception block (471 float32)
- **I:** Toward the PORTFOLIO goal (real challenge trades the FULL FTMO broker -- forex, indices,
  metals, energies, crypto, 130+ instruments). One policy must compare opportunity/risk across a
  1.1 pair, a 40000 index and a 2000 metal -- raw price/ATR are not comparable. Operator's ideas:
  ATR-relative movement comparable across symbol types + session/overlap awareness.
- **R:** CLAUDE.md rule #1 (deliberate shape bump). Append-only (indices 0..460 unchanged).
- **A:** New 10-float `cross_asset` block: asset-class one-hot (`pair/index/metal/energy/crypto`
  with a name CLASSIFIER that covers the full broker, unknown -> safe zeros) + ATR-NORMALIZED,
  SCALE-FREE movement (`move_in_atr`, `atr_pct_price`, `atr_regime`) + sessions (`asian`,
  `london_ny_overlap`). Verified scale-free: EURUSD/US30/XAUUSD (36000x price gap) all read
  atr_pct ~0.54. ATR falls back to the realized range where the cache lacks ATR. Leak-free
  (precompute only). `ASSET_CLASSES` lives in constants (contract). +5 tests; constants/contract/
  builder/env + shape tests (461->471) + OBSERVATION_CONTRACT.md updated. **90/90 green.**
- **Movement logic (the 4 we trade):** per-asset `typical_atr` = typical_daily_range/sqrt(1440)
  (EURUSD 0.00021 .. US30 10.54) anchors the vol REGIME to how each asset NORMALLY moves, and is
  the ATR fallback. Profiles documented in asset_specs (EUR low-vol/mean-revert, GBP livelier, gold
  trends/risk-off, US30 trends/NY). +1 test. 91/91 green.
- **C:** One policy can now perceive ANY FTMO instrument in COMMON units (type + volatility +
  session) -- the perception bridge from single-asset to a mixed portfolio.

## [2026-06-25] Contract v1.4.0 -> v1.5.0 — RECENT-CONTEXT block (479 float32)
- **I:** Operator: the (one) bot should see recent DAILY movement (prior days + last-week avg)
  RELATIVE to the symbol's average, and understand what it needs to PASS in the context of TIME.
- **R:** CLAUDE.md rule #1 (deliberate shape bump). Append-only (indices 0..470 unchanged). One
  policy trades everything, so all features are scale-free / relative.
- **A:** New 8-float `recent_context` block: recent daily ranges expressed RELATIVE to the
  symbol's own average (`week_avg_range_vs_typical`, `prev_day/prev2/today_range_vs_week`) +
  TIME-to-pass pace (`days_elapsed_norm`, `episode_return_so_far`, `pace_vs_2_5pct_plan` where
  0.5 = exactly on the +2.5%/day plan, `challenge_target_remaining`). Daily ranges precomputed
  leak-free (prior days complete; today expanding; week-avg uses prior days only). `days_elapsed`
  tracked per episode. +5 tests; constants/contract/builder/env + shape tests (471->479) + doc.
  **96/96 green.**
- **C:** The one bot now perceives each symbol's recent movement vs its own norm AND whether it
  is on pace (in time) to ladder +2.5%/day to the +10% pass -- pacing awareness for the challenge.

## [2026-06-25] Feature — multi-symbol training ("one bot trades everything")
- **I:** The portfolio goal needs ONE policy trained across ALL assets (the 4 in Drive now), not
  one model per symbol -- only one brain can manage the shared equity/drawdown pot.
- **R:** Operator "one bot that trades everything". No obs/contract change (training-side only).
- **A:** `vector_env_factory.make_multi_symbol_vec_env(symbol_data, ...)` spreads N workers
  ROUND-ROBIN over `{symbol: (ind,close,time)}`, each tagged with its symbol + per-asset
  calibrated size (so the cross-asset features are correct and rewards are comparable -- each
  asset sized to ~2.5%/day). `trainer.train_multi_symbol(...)` mirrors `train()` over it. +1 test
  (97/97). Empirical: ONE bot trained across pair+index+metal -> judgment 0.90-0.99; at 120k it
  breached all 3, at 240k it was SAFE on 2/3 (EURUSD,US30) -- safety-first learning, as expected;
  full profitability needs real training scale (Colab GPU, millions of steps).
- **C:** The one-bot-trades-everything training path is wired and proven to learn; it generalises
  across asset types via the cross-asset perception, improving with training. The portfolio bridge.

## [2026-06-25] Alpha 16 — ORB NY-open breakout (INDICES only) + NY-session reward bonus
- **I:** Operator wants an Opening-Range Breakout alpha for INDICES at the New York open (the most
  liquid part of the day), plus a reward bonus for BANKING profit then via indices.
- **R:** Operator ORB spec, adapted to the repo (no 15m TF; env carries close only; reward was
  equity-only). Obs SHAPE unchanged (fills alpha slot 15 -> still 479). Operator explicitly opted
  into reward shaping (overrides the equity-only convention).
- **A:**
  - `orb_ny_breakout_indices_alpha.py` (+register, +alpha_pack slot 15): INDEX-only (asset_specs
    classifier, covers all FTMO indices). Opening range = 09:30-13:30 UTC (4h pre-open; high/low
    approximated by close); breakout in 13:30-15:30 UTC, filtered by the 30m BB200 middle (=SMA200,
    no 15m TF). Stateful per UTC day; reset() clears. Wired `symbol` + `minute_of_day` into
    MarketContext + the env precompute.
  - NY reward bonus (vars `FTMO_NY_HALF/FULL_TARGET_BONUS` 0.15/0.45): on indices, QUALIFIES when
    the session's CLOSED-in-profit P&L hits >=50% (within 2h) / >=100% (within 3h) of the daily
    target; PAID at day-end ONLY if the day passed (closed >= +2.5% of initial); erased if the day
    fails or breaches. Single-symbol index share = 1.0 (portfolio later computes the real share).
  - +5 tests (`tests/test_orb_ny_breakout.py`); end-to-end verified the +0.60 bonus pays at the day
    boundary on a passed day. 102/102 green.
- **C:** A high-liquidity index entry signal the policy can weight, plus an explicit reward that
  pays only for banking the day via indices in the NY session -- the operator's intended behaviour.
## [2026-06-25] Governance — living env record, env fingerprint (CPU/GPU parity), training ledger
- **I:** As runs multiply (CPU/GPU, seeds, evolving env) we risk losing track of WHAT the env
  includes and WHICH policy to trust. Need a living record + update rules + run records + a way to
  keep CPU and GPU versions identical.
- **R:** Operator request. No behaviour change (records/tooling only); reads the LIVE config.
- **A:**
  - `src/training/env_fingerprint.py`: `env_spec()` + `env_fingerprint()` -- a 12-char hash of
    everything that defines the env (obs contract+size, alpha roster, FTMO rules, reward). SAME
    fingerprint = same environment = comparable policies (CPU or GPU). Header carries the RULES
    for building the future GPU trainer (match fingerprint + step-parity + same policy format).
  - `src/training/run_log.py`: append-only JSONL ledger -- `log_run/load_runs/best_run`.
    `best_run(fingerprint=...)` = which policy to follow (top walk-forward pass-rate, same env).
  - `docs/ENVIRONMENT_STATE.md`: living single-source-of-truth + UPDATE RULES + GPU-build rules.
  - `docs/TRAINING_LEDGER.md` + `records/`: how every run is recorded vs FTMO pass-rate.
  - +4 tests (`tests/test_env_governance.py`). 101/101 green. Live fingerprint: 83d880a5f3bf.
- **C:** The environment can never get lost (it's recorded + hashed), every run is tracked vs the
  FTMO pass-rate, and CPU/GPU runs stay version-locked by a shared fingerprint -> no confusion.

## [2026-06-26] Brutal full-system audit harness (one command) + 2 real fixes it surfaced
- **I:** Mark (a non-programmer) needs ONE command to know whether the bot is safe to run an FTMO
  challenge, in plain English, covering PPO/MLP math, FTMO rule enforcement, env integrity, JARVIS,
  stability, code quality and future risk — and it must FIX whatever it finds broken.
- **R:** Operator-supplied audit spec. Diagnostic/tooling only (no behaviour change to the bot);
  tests the REAL repo, marks delegated/missing items honestly (no fake passes).
- **A:**
  - `tools/run_full_audit.py`: 44 checks across 7 categories -> `audit_results/audit_report.{json,md,html}`
    + a GO/NO-GO verdict (exit 0/1). Tests the live SB3 PPO instance (entropy=ln4, MLP 4+1 heads,
    gradient flow, determinism, 100-step train), the real 479-obs contract, `breach_detector`
    (4% trailing fires before FTMO's 5%/10%; isolates the hard lines with trailing off), env
    reset/step/leak-freedom, JARVIS diagnosis of 5 seeded bug categories, and code/contract health.
  - `tests/test_full_audit.py`: dual-mode — SKIPS under the fast stdlib runner (heavy: spins a real
    PPO), parametrized + severity-marked under pytest (`pytest -m critical`).
  - `audit_results/ASSUMPTIONS.md`: STEP-0 discovery (real module map + the 367->479 / SB3 / 5-TF /
    missing-LIVE-controls assumptions Mark should verify).
  - **Fix 1 (doc bug):** `CLAUDE.md` rule #1 + the obs breakdown said **367 / v1.1.0** — stale. The
    locked contract is **479 / v1.5.0**; updated the headline number and the full 14-block breakdown.
  - **Fix 2 (JARVIS coverage):** added two grounded knowledge entries — `entropy-collapse` (ent_coef=0
    -> deterministic HOLD) and `alpha-vs-hold` (alpha-space 0 vs ACTION_HOLD) — so JARVIS now diagnoses
    all 5 audit bug categories (JARVIS 5/5).
  - Verdict: **GO 38/42**, zero critical failures; 4 honest LIVE-readiness warnings (weekend
    auto-close, regime-coverage-depends-on-data, checkpoint contract-version guard, reconnect layer).
    Fast suite **151/151** green.
- **C:** Mark can now run `python tools/run_full_audit.py` (or ask JARVIS "is my bot safe to run?")
  and get a plain-English, color-coded GO/NO-GO he can trust — and the two real issues it surfaced
  (the 479 doc drift and JARVIS's two blind spots) are fixed, not just reported.

## [2026-06-26] Audit now runs the repo's own unit suite (one "big test")
- **I:** Mark wanted the repo's ~150 unit tests folded INTO the big audit, so one command checks
  everything and the GO/NO-GO accounts for the unit tests too.
- **R:** Operator request ("add those tests to the big test"). Tooling only; no bot behaviour change.
- **A:** `tools/run_full_audit.py` gains a STEP-0 check `0.0 Repo unit-test suite` that runs
  `tools/run_tests.py` in a subprocess (with `RUN_FULL_AUDIT` stripped so it can never recurse into the
  heavy audit), parses the `X/Y passed, Z failed` summary, and treats ANY failure (or nonzero exit) as a
  CRITICAL gate -> NO-GO. Reported as its own prominent line in the console + .md + .html. Kept scored=False
  so the 7-category /42 structure is unchanged; it gates GO/NO-GO via the critical-failure path. Updated the
  JARVIS `run-the-audit` entry to say the audit runs the unit suite too.
- **Verified:** injecting one failing unit test flips the audit GO->NO-GO with exit 1 and names the test;
  removing it restores GO 38/42 exit 0. No recursion (subprocess unit run skips the audit's own test).
- **C:** `python tools/run_full_audit.py` is now the single "big test": ~150 unit tests + 44 system checks
  + GO/NO-GO. A broken unit test can no longer hide behind a green audit.

## [2026-06-26] Hardened the audit harness — adversarial review found + fixed 5 real bugs
- **I:** The audit is the flagship "is it safe?" gate; a silent false GO (or a false NO-GO) is dangerous.
  Ran a 4-lens adversarial review (false-GO paths, scoring math, the new unit-suite integration, report
  robustness) with each finding independently verified. 5 confirmed real; fixed all.
- **R:** Tooling-only hardening (no bot behaviour change). Verified each fix empirically.
- **A:**
  - **(HIGH) Colab false NO-GO:** `tools/run_tests.py` bare-called the pytest-parametrized `test_audit`
    (defined only when pytest is installed, e.g. Colab) -> TypeError -> the audit's unit gate FAILed ->
    spurious CRITICAL NO-GO. Fixed: the stdlib runner now SKIPS pytest-parametrized / arg-requiring tests
    (`_needs_args`). Verified zero-arg run, arg-taking skipped.
  - **(HIGH) Dead-code CRITICAL check:** `t_1_5`'s dead-neuron probe was structurally unreachable (it
    tested ReLU `>=0` + `requires_grad is False`, but SB3 uses Tanh and the probe ran with grad) -> it
    ALWAYS reported 0% dead and PASSed, even for a fully collapsed net. Rewrote it: capture post-activation
    outputs under `no_grad`, detect dead (std~=0) OR saturated (|a|>0.99) units, and assert it captured
    something. Verified: healthy net PASS, weight-zeroed net -> 100% degenerate -> FAIL.
  - **(MED) Vacuous unit gate:** `t_0_0` returned PASS on `0/0 passed` (e.g. tests/ glob breaks). Added an
    `EXPECTED_MIN_TESTS=100` floor -> a collapsed/empty suite is now a NO-GO. Verified the 0/0 path FAILs.
  - **(MED) HTML report injection:** crash/traceback messages with `<`, `>`, `</td>` corrupted the report
    table (exactly when a test crashes). Added `html.escape` on every dynamic value. Verified `<script>`
    is escaped.
  - **(MED) C/POSIX-locale crash:** the md/html writers used bare `open()`; the ✅/🚫 glyphs raised
    UnicodeEncodeError under `LC_ALL=C`, killing the audit before its verdict. Added `encoding="utf-8"`.
    Verified the report writes cleanly under `LC_ALL=C`.
  - (Two CRITICAL "false GO" candidates were investigated and verified NOT real: crit_fail/WARN gating and
    the SB3-delegated ppo_math checks — left as-is with rationale.)
- **C:** The audit can no longer fake-pass a collapsed network, silently lose its unit coverage, self-fail
  in Colab, or corrupt or hide its own report. Fast suite 151/151; audit still GO 38/42 exit 0.

## [2026-06-26] JARVIS opens at the root URL + reusable one-click Colab notebook
- **I:** Opening JARVIS didn't "just work": go_live printed a stale cockpit filename
  (`/JARVIS%20Cockpit.dc.html`) while the real file is `0_JARVIS_COCKPIT.html`, and the root URL `/`
  served nothing (no index). A non-programmer would hit a 404.
- **R:** Tooling/UX only; read-only cockpit unchanged (still GET-only).
- **A:** `jarvis_bridge.create_app` now adds `@app.get("/")` -> RedirectResponse to the cockpit file
  (auto-detected: `0_JARVIS_COCKPIT.html`, URL-quoted), mounted BEFORE the catch-all StaticFiles so
  `/state` etc. still win. `go_live` now prints the correct `http://host:port/` to open. Added
  `notebooks/Camillion_One_Click_Train.ipynb` (13 cells: mount Drive -> clone -> install -> audit ->
  train -> open JARVIS via Colab `proxyPort`), robust to re-runs.
- **Verified:** TestClient — `GET /` 307 -> `/0_JARVIS_COCKPIT.html` -> 200 serves the HUD; `/health`
  and `/ask` 200. Fast suite 151/151.
- **C:** Mark opens JARVIS by clicking one link (or browsing to the server root) and trains from a
  single saved notebook — no remembering filenames or commands.

## [2026-06-26] Loader understands MetaTrader 5 exports + JARVIS link slash fix
- **I:** Real run blew up at cache-build: `no datetime column found (cols=['<DATE>\t<TIME>\t<OPEN>...'])`.
  Mark's data is an MT5 history export — TAB-separated, angle-bracket headers, SPLIT <DATE>/<TIME>,
  dotted dates (2021.01.13), real <VOL>=0. The loader assumed a comma file with one datetime column
  (and would have matched <DATE> as a full timestamp, collapsing 1440 bars/day -> 1). Also the Colab
  JARVIS link was missing a '/' -> DNS NXDOMAIN.
- **R:** Bug fix; no contract/FTMO change. Regression-tested.
- **A:** `load_ohlcv_csv` now sniffs the delimiter (comma/TAB/semicolon/pipe) from the header, strips
  `<...>` from column names, COMBINES separate date+time before any single-datetime fallback (preserving
  1-minute resolution), parses dotted MT5 dates, and prefers TICK volume. +2 tests (MT5 export +
  semicolon/no-volume) in tests/test_csv_loader.py. Notebook JARVIS cell now health-checks the server
  and builds the URL with exactly one slash. Added JARVIS knowledge `data-mt5-format`.
- **Verified:** MT5 sample -> 120 bars at 1-min spacing, tickvol used, close correct, 220-indicator cache
  builds; comma + semicolon files still load. Fast suite 153/153; audit GO 38/42.
- **C:** Mark's MetaTrader CSVs load as-is — no manual reformatting — and the JARVIS link opens.

## [2026-06-26] JARVIS link bug -> tested helper + audit coverage (why no test caught it)
- **I:** The "open JARVIS" link was built by string-concat in a NOTEBOOK cell, so it shipped a missing
  slash (`...colab.dev0_jarvis_cockpit.html` -> the browser read it as a hostname -> DNS NXDOMAIN). No
  test caught it because notebook cells are never run by the suite.
- **R:** Move the logic into real, tested code; cover it in the big test. No behaviour/contract change.
- **A:** Added `jarvis_bridge.COCKPIT_FILE` + `cockpit_url(base)` (slash-safe, URL-quoted, single source
  of truth) + `cockpit_path()`; `create_app`'s root redirect now uses them. The notebook imports
  `cockpit_url` instead of hand-joining. Tests (in the unit suite, which the audit runs): 
  `test_cockpit_url_is_wellformed` (asserts exactly one slash, no `dev0_` bug, file exists, empty base
  raises) + `test_root_url_redirects_to_existing_cockpit` (GET / -> redirect -> 200 HTML). New audit
  check `6.6 JARVIS cockpit reachable` (HIGH) verifies the file + URL + live redirect.
- **Verified:** the regression test FAILS on the old no-slash join and PASSES on the fix; suite 155/155;
  audit GO 38/42 with 6.6 ✅.
- **C:** A malformed JARVIS link (or a 404 cockpit) now fails the big test instead of reaching Mark. The
  link is built by one tested function used by both the notebook and the server.

## [2026-06-26] JARVIS on Colab: render inline (serve_kernel_port_as_iframe), drop URL strings
- **I:** The link stayed broken because the URL-building lived in a NOTEBOOK CELL — `git pull` updates
  repo files but NOT the cell already loaded in the user's browser tab, so re-running Step 6 ran the old
  code. The whole "build a proxyPort URL string" approach is fragile.
- **R:** Robustness fix; no contract/behaviour change.
- **A:** Notebook Step 6 now uses Colab's native `output.serve_kernel_port_as_iframe(8000,
  path='/'+COCKPIT_FILE)` (renders JARVIS INLINE in the notebook — no link, no DNS) plus
  `serve_kernel_port_as_window(...)` for a pop-out tab. Both take `path=`, so no hand-built URLs.
  Strengthened the regression test + audit 6.6 to also assert the DIRECT path `/0_JARVIS_COCKPIT.html`
  serves 200 HTML (the exact path the iframe loads), verified via FastAPI TestClient.
- **Verified:** `GET /0_JARVIS_COCKPIT.html` -> 200 HTML; suite 155/155; audit GO 38/42, 6.6 ✅.
- **C:** JARVIS shows up inside the Colab notebook regardless of browser/DNS/auth, and the served paths
  are guaranteed 200 by the big test.

## [2026-06-26] run_training: --from/--to date range (fast first run on huge histories)
- **I:** Mark's data is 2021->2026 x 4 symbols (~2M bars each); a blind full run could churn for a very
  long time / risk Colab OOM before he's confirmed the pipeline works end-to-end on his real files.
- **R:** Operator-friendly; no behaviour change to a full run (omit the flags).
- **A:** `run_training.py` gains `--from`/`--to` (e.g. `--from 2024-01-01 --to 2024-03-31`) -> slices each
  symbol's DataFrame before building the cache. [2/5] prints the range + per-symbol bar counts. +1 test
  (`test_prepare_caches_date_range_filters`).
- **Verified:** 6-month sample -> full 8,736 bars vs Q1-only 4,368; suite green.
- **C:** First run = a quick few-month slice to confirm the day-by-day report works on his real data,
  then the same command WITHOUT the flags does the full multi-year train.

## [2026-06-26] FIX: portfolio training hung for an hour (SubprocVecEnv pickled gigabytes) + heartbeat
- **I:** A real Colab run sat silent for an hour at "training for 2,000,000 steps" then was Ctrl-C'd.
  Two failures: (1) `make_portfolio_vec_env` used SubprocVecEnv with N_ENVS=8 — it PICKLES the full
  aligned dataset (4 symbols x ~1.8M bars ~= 6GB) to EACH of 8 workers (~50GB) -> Colab OOM/thrash,
  hangs before training starts. (2) Zero progress output, so it looked frozen even when it wasn't.
- **R:** Bug fix; obs/FTMO unchanged. Verified end-to-end.
- **A:** `make_portfolio_vec_env` now uses **DummyVecEnv** (one process, arrays SHARED by reference =
  one copy, no pickling) and defaults to fewer envs (min(4, N_ENVS)). `train_portfolio` prints
  "building the environment..." / "training now..." and runs a **heartbeat callback** that prints
  `steps done / total (steps/s, ~ETA min)` after every rollout. `run_training.py` already supports
  `--from/--to` + `--steps` for a fast first run.
- **Verified:** end-to-end portfolio train prints the build lines + heartbeat, uses DummyVecEnv, no OOM;
  suite 156/156.
- **C:** The portfolio trainer starts immediately instead of thrashing 50GB, and shows live progress +
  ETA so a long run is never mistaken for a hang.

## [2026-06-26] Portfolio training review: 4 fixes (HOLD-collapse, identical workers, missing two-phase, broken report)
A review of TRAINING on real data found four real issues in the PORTFOLIO path (the env we actually train);
all four are now fixed. Obs stays 479; FTMO numbers unchanged; nothing heavy added to step(). Suite 162/162;
audit ✅ GO 38/42. Verified end-to-end on synthetic data (real PPO trains, heartbeat shows the action mix,
report covers all days, model save/load works).

- **(1) Day-by-day report was broken for the portfolio.**
  - **I:** `daily_report` used a loop guard built for the 1-step-per-bar single-symbol env. PortfolioEnv
    takes `len(symbols)` steps per bar, so the guard tripped after ~1/len(symbols) of the data — a 3-day,
    3-symbol run reported **0 days** and reached only 30% of the bars. This is the exact `[4/5]` table the
    operator reads to judge a run.
  - **R:** Bug fix; no contract/FTMO change. **A:** guard now `env.T * steps_per_bar + 16`, where
    `steps_per_bar = len(getattr(env,'symbols',[None]))` (TradingEnv -> 1, unaffected). **C:** the report
    now traverses the WHOLE range. +test `test_portfolio_daily_report_covers_all_days_not_just_a_quarter`.

- **(2) HOLD-collapse risk: zero entropy bonus + identical parallel workers.**
  - **I:** `ent_coef=0.0` removed all exploration pressure; every trade pays a cost (immediate negative
    reward) while HOLD is exactly 0 -> always-HOLD is a stable trap. Worse, all DummyVecEnv workers were
    IDENTICAL (PortfolioEnv.reset ignored seed; no random window) -> the N "parallel" envs replayed ONE
    trajectory = no exploration diversity. **R:** anti-collapse; obs/FTMO unchanged.
  - **A:** `ent_coef` 0.0 -> 0.01; PortfolioEnv gains `random_window/window/seed`; `make_portfolio_vec_env`
    gives each worker a different seed + a random window so they explore DIFFERENT stretches. Episode/window
    end is now `truncated` (breach/pass stay `terminated`) — RL-correct time-limit semantics.
  - **C:** the policy keeps exploring and the parallel envs actually diversify. +test
    `test_portfolio_random_window_gives_diverse_starts`; verified live: action mix ~25% each at init, not collapsed.

- **(3) Two-phase +2.5% bank-and-stop was MISSING from the portfolio bot.**
  - **I:** the documented FTMO two-phase engine (bank at +2.5% of initial, then stop / 1% trail) lived ONLY
    in single-symbol `TradingEnv`. `PortfolioEnv` — the only env `run_training.py` trains — had none of it,
    silently ignoring `cfg.two_phase_enabled=True`. The existing `test_two_phase_daily` tested the WRONG env,
    giving false confidence. **R:** restore documented FTMO behaviour on the pot; numbers unchanged.
  - **A:** added a POT-LEVEL two-phase to `PortfolioEnv.step` mirroring TradingEnv: at +2.5% of initial (on
    pot equity) `_flatten_all()` banks the whole book via `record_close` (single P&L truth), then locks the
    day (no new opens on any symbol; HOLD/CLOSE still pass) or, with `phase2_continue`, keeps trading under a
    1% trail; midnight clears the lock. `_flatten_all` is array/dict only (no heavy ops in step).
  - **C:** the trained portfolio bot now banks +2.5% and stops, per the plan. +new file
    `tests/test_portfolio_two_phase.py` (bank-and-stop, phase2 trail, midnight clear, two-phase-off).

- **(4) Live visibility while training.**
  - **I:** progress only showed steps/s; the operator wanted to SEE it learn (and a HOLD-collapse). **R/A:**
    the heartbeat now also prints the ACTION MIX (% HOLD/BUY/SELL/CLOSE) + mean reward from the PPO
    rollout buffer each update (fully guarded so it can never crash training). **C:** one glance shows
    whether the bot is trading and making money — and a collapse to HOLD 100% is immediately visible.

- **(follow-up, from an adversarial multi-agent review of the above)**
  - **I:** the random-window start could SILENTLY collapse to zero diversity on a SHORT history: with the
    default 5000-bar window, any aligned slice under ~5,202 bars made every worker pin to `warmup` again —
    re-creating the identical-copies bug on exactly the fast `--from/--to` first-run path. **R/A:** clamp the
    effective window to at most half the usable span, so there is always room to sample a varied start;
    verified the 8 workers now get 8 distinct starts at T=3000 and T=5201 (was 1). +regression test
    `test_portfolio_random_window_diversifies_even_on_short_history`. Also: guarded the heartbeat against an
    empty rollout buffer (no `nan`), and refreshed the JARVIS `entropy-collapse` knowledge entry to say
    ent_coef is now 0.01 (was stale at 0.0). Suite 163/163; audit ✅ GO.

## [2026-06-27] Training perf #1a: build per-symbol features ONCE and SHARE across workers (+ build progress bar)
- **I:** A real Colab run sat on "building the training environment..." for ~5 hours and never started
  training. Root cause (measured): `make_portfolio_vec_env` built a fresh `PortfolioEnv` per worker, and
  each `PortfolioEnv` precomputed one `TradingEnv` per symbol -> **4 workers × 4 symbols = 16 redundant
  precomputes** over the full 1.8M-bar history (~7,400 bars/s here -> ~65 min on a fast box, more on Colab),
  plus ~15 GB of redundant alpha+streak arrays (vs Colab free ~12.7 GB). And no progress output -> looked
  frozen. **R:** perf only; obs(479)/FTMO/`step()` unchanged. Tracked in `TRAINING_REQUIREMENTS.md` (#1).
- **A:** New `build_portfolio_subs()` builds the per-symbol `TradingEnv`s ONCE; `make_portfolio_vec_env`
  builds them once and passes the SAME dict to every worker via `PortfolioEnv(subs=...)`. The sub-envs are
  read-only after precompute (PortfolioEnv only reads them; never calls `sub.step`), so sharing one copy
  across the DummyVecEnv workers (one process) is safe. Cuts build time AND memory ~N-fold (16 builds -> 4).
  Added a `progress=` flag to `TradingEnv` -> a per-symbol "[SYM] 30% (…bars)" build bar so it is never a
  silent multi-hour mystery again.
- **Verified:** factory build prints once for 4 symbols (not 16); all 4 workers share ONE set of sub-envs
  (`is` identity) yet keep DIFFERENT random-window starts; suite 163/163.
- **C:** the full-history build drops from ~16 redundant builds/~15 GB to 4 builds/~3.7 GB with a live
  progress bar -> the run actually reaches training on a reasonable Colab tier. (Still TODO in #1:
  save-features-to-disk for instant re-runs, and auto-calibrate workers/threads/device to ~70–80%.)
