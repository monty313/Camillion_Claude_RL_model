# UPDATE LOG (IRAC)

Every change appends a dated IRAC entry. **Conclusion** states why it helps the bot
pass FTMO-style challenges more consistently.

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
