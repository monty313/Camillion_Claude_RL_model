# UPDATE LOG (IRAC)

Every change appends a dated IRAC entry. **Conclusion** states why it helps the bot
pass FTMO-style challenges more consistently.

## [2026-06-26] Added 2 non-directional movement alphas (ADX as an ALPHA-PRIVATE indicator)
- **I (Issue):** Add a "is the market moving?" filter (STRAT-006): on both TFs, ADX rising AND
  ATR rising → 1, else 0 (never −1). It needs ADX, which the repo didn't have — and ADX columns
  in the obs would resize the locked observation.
- **R (Rule):** Operator 2026-06-26 — "if we don't have to add the extra indicator to the obs,
  don't; we just need the signal." Keep the obs frozen at 479/v1.5.0; expose the movement only
  through the alpha's 1/0 slot.
- **A (Application):** New `src/indicators/adx.py` (Wilder ADX, TA-Lib fast-path). New
  **alpha-private** indicator path (`constants.py` ADX_PERIODS/ALPHA_PRIVATE_SHIFT;
  `base.py` per_tf_alpha_private_columns/compute_timeframe_alpha_private, SEPARATE from the 220
  obs indicators; `cache_builder.build_aligned_alpha_private` + `load_alpha_private`). Env takes
  optional `alpha_indicators=` and merges them into `ctx` ONLY (obs untouched). Two alphas
  `dual_movement_filter_5m_30m` (slot 16) and `dual_movement_filter_30m_4h` (slot 17), registered
  in `alpha_pack`. Multi-symbol factory accepts an optional per-symbol 4th element. Streak is the
  existing alpha_streak block (1,1,1→1,2,3 confirmed). Tests: `test_dual_movement_filter.py`
  (logic, obs-unchanged, env wiring, absent-is-safe, cache alignment); `test_alpha_pack` 16→18.
- **C (Conclusion):** The policy gains a movement gate (and a reusable "add an alpha that needs a
  new indicator WITHOUT changing the obs" pattern) — more selective entries toward a consistent
  FTMO pass, with the observation contract still frozen on the road to 1000 alphas. Obs **479 /
  v1.5.0 unchanged**; alpha roster 16→18 (fingerprint rolls = new experiment line).

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
