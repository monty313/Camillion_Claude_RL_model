# UPDATE LOG (IRAC)

Every change appends a dated IRAC entry. **Conclusion** states why it helps the bot
pass FTMO-style challenges more consistently.

## [2026-06-29] "Don't trade the chop": 5m CCI open-gate OR→AND + wired into the portfolio (CPU + TPU)
- **I (Issue):** Operator: don't allow a new trade when, on the 5m, BOTH CCI(30) AND CCI(100) sit in [-50,50]
  (a flat, no-momentum market). The existing `open_gate` used OR (block if EITHER is neutral) and was applied
  ONLY in the single-symbol env — the shared-pot PortfolioEnv (the product) ignored it. This is lesson #1
  (`PSYCHOLOGY.md`): take a protected zero on a dead market instead of forcing a trade.
- **R (Rule):** Reward/behaviour shaping only — obs unchanged (517/v1.8.0). CPU ↔ JAX bar-for-bar. `open_gate`
  default OFF (existing trajectories unchanged); training turns it on.
- **A (Application):**
  - `TradingEnv._precompute`: `open_gate_blocked` is now **AND** — `(|5m cci30|<=50) & (|5m cci100|<=50)`
    (was OR). Threshold stays 50. Blocks a new directional open only on a genuinely flat 5m.
  - Wired into the **portfolio**: `PortfolioEnv(open_gate=…)` blocks a new open when `sub.open_gate_blocked[t]`;
    `open_gate_blocked` threaded into `PortfolioStaticData` / `PortfolioDeviceStatic`, `open_gate` added to
    `PortfolioParams`, and the JAX `step_portfolio` applies it branchlessly right after the day-lock.
  - `test_open_gate.py` flipped to AND semantics; notebook Step 8b trains with `open_gate=1.0`.
- **C (Conclusion → consistency):** The bot stops forcing entries into chop — fewer low-edge trades, fewer
  whipsaw losses, and a real ability to sit out a dead 5m and protect the day/streak. Verified CPU ↔ JAX
  bar-for-bar with the gate ON; full CPU suite green; portfolio learn-smoke finite with the gate + all
  behaviours on.

## [2026-06-29] Streak psychology: stretch the horizon + SEE the multi-day standing + selectivity (contract v1.8.0)
- **I (Issue):** A brutally-honest audit found the two pieces that make the operator's vision (a streak-guarding
  sniper, see `PSYCHOLOGY.md`) were missing: (1) the bot was **near-sighted** (gamma 0.9995 ≈ ⅓ day with the
  4-symbol cycle), so the escalating won-day STREAK reward was discounted to ~0 a day out — it couldn't VALUE
  or PROTECT the streak; and (2) the conviction nudge was a flat ≥2-of-3 gate that didn't reward the
  **strongest** consensus, and the daily-quota pressure drowned selectivity.
- **R (Rule):** Operator 2026-06-29. A new obs block is a deliberate **contract bump** (CLAUDE.md rule #1 —
  append-only, obs 513→517, v1.8.0). FTMO numbers unchanged. CPU ↔ JAX bar-for-bar. `gamma` must stay in sync
  across `jax_config` and `src/training/trainer.py`.
- **A (Application):**
  - **Stretched the horizon:** `gamma 0.9995 → 0.9999` (~1/3 day → ~1.7 days) in `jax_config.GAMMA` +
    `trainer.PPO_HPARAMS`. Now a breach TODAY is felt as also forfeiting tomorrow's (bigger) streak reward —
    the streak becomes a psychology, not just a number. (Even 0.9999 can't see 40 days; the obs block below +
    the escalation carry the rest.)
  - **CONSISTENCY obs block (v1.8.0, 4 floats, append → 513..516):** the bot's multi-day FTMO standing so it
    can SEE what it's protecting — `won_day_streak_norm`, `days_won_norm`, `won_day_rate`, `days_into_journey`.
    Shared builder `win_loss_features.consistency_features` (jnp twin in `jax_obs_blocks`); `PortfolioEnv`/JAX
    track a new `days_won` counter; the single-symbol env emits zeros (no streak logic). DYNAMIC block, both
    envs, parity-clean.
  - **Selectivity = strongest one-directional consensus:** the conviction reward now SCALES with the **number
    of firing alphas aligned with the trade** (`(alpha_matrix == target).sum()`, capped at
    `CONVICTION_ALIGN_CAP=8`), and pays ONLY when the bot traded WITH the majority (`entry_agreed`) and won —
    so it prefers the symbol/direction with the greatest signal agreement, never a thin or against-consensus
    trade. Replaces the flat ≥2-of-3 gate; mirrored CPU + JAX.
  - `PSYCHOLOGY.md` (new): the canonical statement of the trader we're building, to judge every future change
    against.
- **C (Conclusion → consistency):** The bot can now both **value** (longer horizon) and **see** (the streak in
  its obs) its multi-day run, so "protect the streak" becomes a real drive — and its conviction reward pulls it
  toward the highest-agreement trend setups it trades *with*, sharpening the sniper over the churner. Verified
  CPU ↔ JAX bar-for-bar at 517 (single + portfolio, x64); full CPU suite green.

## [2026-06-29] 3 strong-setup ALPHAS (slots 18/19/20) + a PnL-capped CONVICTION bonus (PortfolioEnv, CPU + TPU)
- **I (Issue):** Operator wants the bot motivated to trade the day's strong setups: (1) CCI extended beyond
  +/-160 on 5m AND 30m, (2) price beyond BOTH BB(200,dev1) AND BB(20,dev1) on ANY timeframe, (3) price
  above/below the forward-displaced SMA(4) on 2+ timeframes. A code-grounded multi-agent review concluded
  these are **signal generators (alphas)**, not bespoke reward bonuses, and that a literal "BIG incentive to
  TRADE" is the #1 risk to a 40-day streak (it teaches over-trading into the 4% wall).
- **R (Rule):** CLAUDE.md — "each strategy = a signal generator (+1/-1/0); grow to ~1000 alphas; the obs is
  locked." Alphas fill free slots (NO obs-contract change, the obs stays 513/v1.7.0). Keep CPU <-> JAX
  bar-for-bar. Leak-free (no future-shifted data).
- **A (Application):**
  - **ROUTE A — 3 new per-slot alphas** (free slots 18/19/20 of 64; columns already cached, no precompute):
    `cci_x160_align_5m_30m` (|cci30| > 160 on 5m AND 30m), `bb_double_breakout_anytf` (close beyond BB200(dev1)
    AND BB20(dev1) on ANY TF), `fwd_sma4_align_5m_30m` (close vs the forward-displaced `sma_p4_s3` on 5m AND
    30m). The bot SEES them in the alpha obs blocks and the existing (10x'd) alpha-shaping reward motivates
    trading WITH / BEATING them — and it learns a per-alpha WEIGHT. No obs-shape change; alphas precompute into
    the alpha matrix both envs already consume, so ZERO JAX kernel change.
  - **ROUTE B — one PnL-capped CONVICTION bonus** (`cfg.conviction_bonus`, default 0.0 = off; notebook trains
    at 0.1): paid when **>=2 of those 3 alphas confirmed the trade's direction AT ENTRY** and the trade CLOSES
    in profit with the day net up. Reads the 3 alpha slots from the per-symbol alpha matrix (no new precompute,
    no new side arrays). Shares the existing `min(bonus, trade-PnL)` cap with the alpha/band/re-entry terms, so
    it can NEVER pay for a loser or for volume — it only breaks ties toward high-conviction setups. Mirrored in
    `src/env/portfolio_env.py` + `jax_tpu/jax_portfolio_env.py` (PortfolioParams + EnvState `entry_confirms` +
    open-leg compute + close-leg pay); `CONVICTION_SLOTS=(18,19,20)` resolved by name in `alpha_pack` (a test
    asserts it can't drift).
- **C (Conclusion → consistency):** The bot gains explicit perception + signal for the operator's three
  high-quality setups (the designed, scalable mechanism), and a SMALL, loss-proof nudge toward entering when
  multiple setups agree — without the over-trading trap of a literal "pay to trade" bonus. Verified CPU <-> JAX
  bar-for-bar with the conviction bonus ON; obs unchanged at 513.

## [2026-06-29] Reward REBALANCE for 40-won-days-in-a-row (PortfolioEnv, CPU + TPU) — no obs/contract change
- **I (Issue):** A code-grounded "bot psychology" review (multi-agent) found the reward's #1 threat to a 40-day
  streak was its OWN shape: an **every-4th-won-day +1.0 jackpot** (largest scalar in the reward) taught the bot
  to GAMBLE as a streak neared a multiple of 4, while a **−0.2 breach** was only a flinch, and the dense
  intraday shaping (seek 0.10, dd-proximity 0.02) was tiny relative to the day outcomes. Operator chose a new
  scale and asked for an honest review of it.
- **R (Rule):** Operator 2026-06-29. Reward shaping only — the **locked 513-obs contract and the FTMO rule
  numbers (2.5%/4%/5%/10%) are UNCHANGED.** Keep CPU ↔ JAX bar-for-bar.
- **A (Application):** in `config/variables.py` + `config/ftmo_config.py`, mirrored in the day-scoring of
  `src/env/portfolio_env.py` and `jax_tpu/jax_portfolio_env.py`:
  - **WON day** = `+day_pass_reward (10)` **+ an ESCALATING streak bonus**: every ADDITIONAL consecutive won
    day pays `+streak_bonus (1.0)` more, i.e. day N pays `10 + 1.0*min(N-1, streak_bonus_cap=10)`. This
    REPLACES the lumpy every-4th `pass_bonus` jackpot — a smooth per-day reward has no multiple to gamble for;
    the **cap** bounds the per-day value so the PPO value function can't blow up on a long ramp.
  - **FAILED day** = `−day_fail_penalty (5)` and resets the streak; **BREACH** = `−breach_penalty (20)` (2× a
    won day) felt immediately (the agent is near-sighted at γ≈0.9995 with the multi-symbol cycle, so the
    streak-reset alone is too delayed a deterrent).
  - **10× the alpha terms** (`agree 0.001→0.01`, `against 0.001→0.01`, `beat 0.002→0.05`, still PnL-capped +
    day-up gated) so the bot develops real edge (BEAT) instead of merely shadowing the consensus.
  - **SCALED the dense shaping with the day reward** so it stays visible (else the bot would learn only from the
    sparse midnight outcome and lose proactive wall-avoidance): `target_seek_weight 0.10→3.0` (still HWM-capped,
    ~⅓ of a won day) and `dd_proximity_coef 0.02→2.0` (nearing the 4% wall now costs real points every step).
  - `pass_bonus (1.0)` is now ONLY the +10% CHALLENGE-pass terminal (eval; training continues past +10%).
  - New params `streak_bonus`/`streak_bonus_cap` threaded through `FTMOConfig`, `PortfolioParams`,
    `portfolio_params`, the notebook Step 8b kwargs, and the portfolio-parity test.
- **C (Conclusion → consistency):** Removes the single biggest self-sabotage (the gamble-for-the-jackpot
  incentive), makes a blow-up genuinely feared in the moment (−20), rewards SUSTAINED consistency (each extra
  day worth more, bounded for stability), keeps the smooth "climb to +2.5% / ease off the wall" guidance alive
  at the new scale, and pushes the bot to BEAT the alphas. CPU ↔ JAX verified bar-for-bar with the new reward.

## [2026-06-29] TRADE-RISK observation block + BB(10,1) hard stop + risk-based sizing + band-stack/re-entry bonuses (contract v1.7.0, CPU + TPU)
- **I (Issue):** The bot traded with NO awareness of its OPEN-trade risk: no view of unrealized P&L in
  ATR/account terms, no sense of how close it was to a stop, no max favorable/adverse excursion, no re-entry
  context — and no hard stop or risk-aware sizing. Operator wants the policy to (a) SEE and MANAGE each trade,
  (b) be auto-stopped at a real technical level, (c) size each entry to a fixed % risk, (d) learn to RE-ENTER
  a winner, and (e) get a small bonus for entering with a multi-TF band-stack that closes in profit.
- **R (Rule):** Operator 2026-06-29 (confirmed YES to all three design points + the band-stack bonus). A new
  observation block is a **deliberate contract bump** (CLAUDE.md rule #1). FTMO numbers unchanged. No
  TA-Lib/pandas in `step()` — the BB(10,1) bands are PRECOMPUTED.
- **A (Application):**
  - `config/constants.py` — **contract v1.6.0 → v1.7.0**: appended the 14-float `trade_risk` block →
    **OBS_TOTAL_SIZE 499 → 513**. Indices 0..498 are UNCHANGED (append-only). Names in
    `src/observation/observation_contract.py` (`TRADE_RISK_NAMES`).
  - `src/observation/trade_risk.py` (**new**) — ONE shared numpy builder for the 14 floats: `in_trade`,
    `direction`, unrealized P&L in **ATR units** and as **% of the pot**, distance to the **2×ATR(14) SOFT**
    stop and to the **1m BB(10,1) opposite-band HARD** stop (0→1), `bars_held`, **MFE/MAE** in ATR, re-entry
    context (bars since last close, last dir, price-vs-last-exit in ATR), and the **band-stack** flags
    (price above/below BB200(dev1) AND BB10(dev1) on BOTH 1m & 5m). jnp twin: `jax_obs_blocks.trade_risk_features`
    (cancellation-prone subtractions run in the native/x64 dtype to match the CPU — see its docstring).
  - BB(10,1) bands: the 220-indicator cache has BB periods **20 & 200 only**, so `TradingEnv._precompute`
    (`compute_bb10_bands`) builds the **1m + 5m BB(10,1) upper/lower** leak-free from `close`+`time` (no
    cache-format change; cached as 4 new arrays, `feature_cache` bumped `fc-v1 → fc-v2`). BB200(dev1)+ATR(14)
    are read from the existing cache columns. All threaded into the JAX env via `jax_static_features`.
  - **Per-trade state** (entry bar/ATR, the hard-stop band at entry, MFE/MAE, last-close context, band-stack &
    re-entry flags, the actual risk-based size) tracked in BOTH envs — single-symbol (`TradingEnv`/`jax_env`,
    obs-only) and shared-pot (`PortfolioEnv`/`jax_portfolio_env`).
  - **BEHAVIOURS (PortfolioEnv, default OFF — training turns them on):** (1) **BB hard stop** auto-closes a
    long below / short above the 1m BB(10,1) band (protective, bypasses the day-lock); (2) **risk-based
    sizing** sets each entry's size so a stop-out loses ~`risk_per_trade_pct`% of the pot, capped at the
    calibrated size; (3) **band-stack** enter bonus + (4) **re-entry** nudge — small, PnL-capped CLOSE bonuses
    (`cfg.band_stack_bonus`, `cfg.reentry_bonus`) paid when the qualifying trade closes in profit, day net up.
  - JAX/TPU: `jax_config.OBS_SIZE 499→513`; `trade_risk` added to `DYNAMIC_BLOCKS`; per-symbol band/ATR arrays
    threaded; `step_portfolio` mirrors all four behaviours branchlessly. Fixed a latent `jax_trainer`
    `_restart_continue` bug (`s.t` vs single-symbol `s.ptr`) that broke single-symbol on-device training.
- **C (Conclusion → consistency):** The policy can now SEE its open-trade risk (ATR/account/band units),
  TIME exits, and RE-ENTER trends; a real BB(10,1) hard stop caps the downside; risk-based sizing keeps a
  losing day small (one stop ≈ a fixed tiny % of the pot, so several losers can't breach the 4% wall); and the
  band-stack/re-entry bonuses bias it toward high-quality, trend-aligned entries — all of which push toward the
  consistent, low-drawdown +2.5%/day days the 40-won-day streak demands. CPU ↔ JAX verified **bar-for-bar**
  (max|obs| 2.4e-7, max|reward| 3.7e-10) with the behaviours both OFF and ON.

## [2026-06-28] Two ADX-DI alphas (slots 16/17) + RAW OHLC observation block (contract v1.6.0, CPU + TPU)
- **I (Issue):** Operator wants (1) two new alphas on the ADX/DMI directional system — ADX periods **14 & 45**,
  alpha A on **5m & 30m**, alpha B on **30m & 4h**; rule: *all −DI above +DI = SELL, all −DI below +DI = BUY* —
  and (2) **raw OHLC added to the observation**, on **both** the CPU and the JAX/TPU trainers. ADX/+DI/−DI did
  not exist in the repo, and both needs require raw **high/low** the env never carried (it holds only `close`).
- **R (Rule):** Operator 2026-06-28. Adding alphas must not destabilise the obs (fills slots); adding OHLC to the
  obs is a **deliberate contract bump** (CLAUDE.md rule #1 — confirmed with the operator before bumping).
- **A (Application):**
  - `src/indicators/adx.py` — Wilder **+DI / −DI / ADX** (TA-Lib fast-path + pandas fallback, NaN warmup).
  - `src/data/aux_features.py` + `cache_builder.build_aligned_aux` — ONE leak-free `{symbol}_aux.npy` (T, 32):
    **[ OHLC obs block (20 = O/H/L/C × 5 TFs) | ADX-DI side-channel (12 = ±DI@14/45 on 5m/30m/4h) ]**. Built at
    cache time (where OHLC exists), aligned last-closed-bar (no leak), trimmed by `align_symbol_data`, threaded
    into the env as `aux=`. The OHLC half goes INTO the obs (block 15); the DI half feeds only the two alphas.
  - `config/constants.py` — **contract v1.5.0 → v1.6.0**: appended `ohlc` block (20) → **OBS_TOTAL_SIZE 479 → 499**.
    Indices 0..478 are UNCHANGED (append-only); blocks 1–14 keep their positions.
  - Strategies: `adx_di_align_5m_30m_alpha.py`, `adx_di_align_30m_4h_alpha.py` (+ register helpers) → slots **16, 17**.
    All-four-agree (period × TF) sign rule; missing/equal → inactive. Only the sign of (−DI − +DI) is used.
  - Env: `TradingEnv`/`PortfolioEnv` take `aux=`, emit the `ohlc` block, and inject the DI columns into the ctx;
    no aux → OHLC block is zeros and the two alphas stay inactive (old caches still load — degraded, never a crash).
  - JAX/TPU: `jax_static_features.build_static_data` places `ohlc` (static); `jax_config.OBS_SIZE 479→499`,
    `N_STATIC_OBS 439→459`. The TPU trainer lifts `alpha_matrix` from the CPU build, so slots 16/17 work for free.
  - Robustness: the feature-cache fingerprint now folds in an **aux content hash** (a no-OHLC cache can never be
    loaded as an OHLC one) + a `aux_features_code` source hash. Full CPU suite + JAX parity green; new tests added
    (`tests/test_adx_di_alphas.py`: DI logic, slot wiring, DI-needs-aux, OHLC block correctness, aux leak-free).
- **C (Conclusion):** The bot gains a trend-direction alpha family AND first-class raw OHLC perception (High/Low/Open
  on every timeframe) in one leak-free, parity-clean change — more edge and more market context toward a consistent
  FTMO pass — without ever silently moving the observation. A trained v1.5.0 policy retrains (shape changed, as flagged).

## [2026-06-28] Crash-safe checkpointing: save on interrupt + every checkpoint (resume never loses much)
- **I (Issue):** Operator: make sure that if training is INTERRUPTED it saves the progress, and that progress
  is saved when a checkpoint is reached. A long unbounded run to 40-won-days must survive a Colab disconnect
  or a Ctrl-C without losing meaningful work.
- **R (Rule):** Operator 2026-06-28. Progress = the policy params + obs-norm + the run ledger; saving must be
  frequent + on interrupt; `resume=True` continues from the latest.
- **A (Application):** `jax_tpu/jax_trainer.train` now saves progress (a) at EVERY eval (existing — full
  checkpoint + `jax_progress.jsonl` ledger + rolling `best_policy/`), (b) **every `CHECKPOINT_EVERY=25`
  updates** (a lightweight params/norm checkpoint between evals, so a hard kill loses at most ~25 updates),
  and (c) **on INTERRUPT** — a `try/except KeyboardInterrupt` saves on Ctrl-C, and a best-effort `SIGTERM`
  handler (Colab disconnect) sets a flag that the loop catches to save + exit gracefully (the previous
  handler is restored in `finally`). The interrupt save also writes a named `interrupted/` policy. `resume=True`
  reads `latest_step.txt` and continues. TESTED: an interrupt mid-run wrote the checkpoint + `interrupted/` +
  `latest_step.txt`, periodic checkpoints appeared between evals, and a resumed run continued from the saved
  update. Full repo suite green.
- **C (Conclusion):** A disconnect or Ctrl-C now costs at most a handful of updates, not the whole run — the
  unbounded march to 40 winning days in a row is fully crash-safe and resumable from Drive.

## [2026-06-28] Fail = START OVER (not trade dead) + proof the bot learns and the report tracks it
- **I (Issue):** Operator concern (suspected from the CPU bot): is the bot actually LEARNING and do the
  reports accurately SHOW it? And: "if it fails one day the bot stops trading for that day, then restarts
  the following day" — but a single env stepped past a breach was **trading a DEAD/breached account**
  (equity below the wall, `episode_breached` stuck True, never recovering), which makes a report look frozen.
- **R (Rule):** Operator 2026-06-28. A failure must = START OVER and keep going (fresh attempt), and the
  live report must move as the policy learns. Env step-parity unchanged (these are rollout/eval/report changes).
- **A (Application):** Probed the env: observations NEVER stop (one per step), but a continuously-stepped
  single env trades dead after a breach. Fixes: (1) **training rollout** now restarts a FRESH account at the
  current bar on a breach (continue the same timeline — the bot never trains dead) and draws a new random
  window only on a window-end; (2) the **won-day-streak eval** now models "stop trading for the day, restart
  the following day" — on a breach it goes FLAT (forced CLOSE) until the next midnight, then resets fresh
  (which also stops a partial breach-day being mis-counted as a winning day); (3) **CPU `daily_report`** gets
  a new `PortfolioEnv.restart_account()` and calls it on a breach so each day after a fail is a clean attempt
  (no dead-walk). PROOF OF LEARNING: a portfolio training run on a learnable uptrend was logged eval-by-eval —
  `mean_reward -0.0012 -> +0.0003`, `P(pass) 0% -> 100%`, `won-day-streak 0 -> 4`, `BUY-action 11% -> 63%`,
  `beats_alphas False -> True` — every metric moved and was captured live (incl. the noisy dips, so the
  operator can SEE struggle vs progress). 14 daily_report/portfolio/parity tests green; full suite green.
- **C (Conclusion):** A failed day now triggers a clean restart instead of a dead account, in training, the
  eval, and the report — so the bot keeps learning after a fail and the dashboard honestly tracks it
  (reward, P(pass), winning-day streak, BUY-mix, beat-the-alphas), demonstrated to actually move as it learns.

## [2026-06-28] JAX/TPU: stop at 40 WINNING DAYS in a row + reward/horizon tune + beat/all-symbol diagnostics
- **I (Issue):** Operator clarified the goal: keep training **until the bot strings 40 WINNING DAYS in a
  row on held-out data** — a winning day ends >= +2.5%, and a BREACH (or a losing day) **resets the streak
  (start over)**, training never stops short of 40. Plus a 6-point upgrade: survive breaches in the
  walk-forward, measure more than one slice, add a drawdown-proximity penalty, drop the breach cliff,
  lengthen the planning horizon, and instrument the action mix; and confirm it trades ALL symbols, not one.
- **R (Rule):** Operator 2026-06-28. Parity stays sacred: every reward change is mirrored in BOTH the CPU
  PortfolioEnv and the JAX env and re-verified bar-for-bar; FTMO numbers (2.5%/4%/5%/10%/+10%) unchanged.
- **A (Application):** (STOP) `jax_eval.evaluate_won_day_streak` runs continuous held-out walk(s) that
  AUTO-RESET the account on a breach (start over) and keep walking, tracking the longest run of winning days
  (`daily_pass_streak`); the trainer now STOPS at `TARGET_WON_DAY_STREAK=40` (challenge pass-rate kept as a
  HEALTH metric), and training is now **UNBOUNDED** (`total_updates=None`) so only 40-in-a-row stops it
  (entropy anneal decoupled via `ANNEAL_UPDATES`). (REWARD, both envs) a **drawdown-proximity** penalty
  (`dd_proximity_coef=0.02`, dense `coef*(dd/wall)^2` so the bot plans AWAY from the wall) + a smaller
  **breach cliff** (`breach_penalty 1.0 -> 0.2`; the streak reset is the real deterrent now); `pass_bonus`
  kept at 1.0. (HORIZON) `gamma 0.997 -> 0.9995` (~a full trading day) so it plans toward the midnight
  target. (DIAGNOSTICS) the dashboard now shows the **winning-days-in-a-row** bar, the existing
  HIDING/OVER-TRADING/LEARNING diagnosis, **VS ALPHAS** (bot vs a follow-the-alpha-consensus baseline on the
  same windows — is it BEATING them?), the **ACTION MIX** (HOLD/BUY/SELL/CLOSE — kills the HOLD-collapse
  question), and **per-symbol EXPOSURE** with a "⚠ CONCENTRATED" flag (confirms it trades ALL symbols, not
  one). (CPU) `daily_report` now keeps walking past a breach AND past a +10% pass (only end-of-data/max_days
  stops it) so the day-by-day report shows every day + the longest won-day run. CPU↔JAX portfolio parity
  re-verified to max|reward|≈3e-9; 28 CPU portfolio tests + jax_tpu parity + deep checks green.
- **C (Conclusion):** The bot now trains, unbounded, toward the operator's real bar — **40 winning days in a
  row** — with a reward that pulls toward the daily target while planning away from the wall, a longer
  horizon, and a live dashboard that shows whether it's winning, hiding, over-trading, beating the alphas,
  and trading the whole book. That is the most direct path to — and the clearest read on — a CONSISTENT pass.

## [2026-06-28] docs: `docs/ALPHAS.md` — the complete alpha (strategy) catalogue
- **I (Issue):** The 16 production alphas (+3 example strategies) were only documented
  inside each source file's header; there was no single reference explaining what every
  alpha is and the exact rule it fires on. Hard for the operator (non-programmer) to see
  the whole signal library at a glance.
- **R (Rule):** Operator 2026-06-28: "create a .md file that has all of the alphas in it
  and give a detailed description of how they work." Must stay faithful to the code
  (CLAUDE.md: code is source of truth; keep the alpha-vs-action-HOLD and per-slot
  contract intact in the explanation).
- **A (Application):** New **`docs/ALPHAS.md`**, built by reading every file in
  `src/strategies/` (cross-checked against `config/constants.py` and `src/indicators/`).
  It documents: the `BaseStrategy` +1/-1/0 contract and the alpha≠action-HOLD≠empty-slot
  distinction; the 64-fixed-slot registry and why it never resizes; the canonical slot
  map (0–15); how alpha outputs reach the policy (`collect_alphas` + `occupancy_mask` +
  the 4 summary %); the full indicator vocabulary (5 TFs × 44 columns, column naming,
  `bbN_devX_middle == SMA(N)`, the SMA fan, `sma4_sh4_high/low`); and a per-alpha
  writeup for all 6 families (Gravity, Regime Pulse, CCI Surge, SMA Stack, SMA Reversion
  Rally, ORB) with exact BUY/SELL/INACTIVE conditions + thresholds, the 3 examples, and
  a "how to add a new alpha" section. Docs-only; no code changed.
- **C (Conclusion):** The operator can now see and reason about the entire signal library
  in one place — which alphas exist, what each looks at, and exactly when it fires — so
  tuning, adding, or auditing alphas (toward the ~1000-alpha plan) is faster and less
  error-prone, without touching the locked observation contract.

## [2026-06-28] JAX/TPU trainer (`jax_tpu/`): on-device PPO, step-parity to the CPU env, train-to-40-in-a-row
- **I (Issue):** CPU training can't reach the SCALE (thousands of parallel lifetimes) or the
  CONSISTENCY bar we want. Operator 2026-06-28: build a TPU version of the training that (a) uses
  ~70–80% of the Colab TPU, (b) reads the market data already in Colab/Drive, and (c) keeps training
  until the bot passes **40 challenges in a row**, saving progress + every policy's details to Drive.
- **R (Rule):** Operator 2026-06-28 + `docs/JAX_GPU_TPU_TRAINER_BLUEPRINT.md` invariants: it is ONE
  bot written a second way — the TPU env MUST match the CPU env bar-for-bar (same 479 obs `v1.5.0`,
  same reward, same FTMO numbers, same `env_fingerprint`), proven by a step-parity test. CLAUDE.md
  rules hold (obs shape sacred, FTMO numbers fixed, no TA-Lib/pandas in the hot loop).
- **A (Application):** New self-contained folder **`jax_tpu/`** (everything JAX in one place):
  `jax_ftmo.py` (branchless breach + two-phase banking), `jax_obs_blocks.py` (the 40 dynamic obs
  floats as a 1:1 jnp port of `win_loss_features`), `jax_static_features.py` (builds the shared
  `(T,479)` STATIC obs tensor + per-bar scalars straight from a precomputed CPU `TradingEnv` — so 439
  of 479 obs floats are byte-identical and the hot loop indexes a shared table, the "build once +
  share" scaling plan), `jax_env.py` (an `EnvState` pytree + branchless `step_env` reproducing
  `trading_env.step` incl. day rollover + NY-index bonus + breach/pass + two-phase banking),
  `jax_ppo.py` (Flax 3×256 tanh + GAE + clipped PPO + a `VecNormalize`-style obs normalizer, all
  hyperparams mirroring `trainer.py`), `jax_trainer.py` (pmap rollout+update at scale, domain-randomized
  risk, **stop at 40 consecutive held-out challenge passes**, checkpoints + a `jax_progress.jsonl`
  ledger + `best_policy/` to Drive every eval, resumable), `jax_eval.py` (held-out walk-forward
  pass-rate = P(pass) + the consecutive-pass streak), `jax_checkpoint.py` (Drive persistence),
  `jax_indicators.py` (OPTIONAL on-device indicators in jnp, parity-tested vs `src/indicators/*`),
  `export_to_pytorch.py` (JAX→PyTorch 3×256→**ONNX** for MT5, bit-verified), the **step-parity GATE
  test** (`tests/test_jax_parity.py` + `run_parity.py`), and the Colab **`notebooks/Camillion_JAX_TPU_
  Train.ipynb`** (mirrors `run_training.py`'s data path + a TPU-utilization probe for the 70–80%
  target). `pyproject.toml` gains `jax-gpu`/`jax-tpu` extras and adds `jax_tpu/tests` to `testpaths`
  (self-skips when jax is absent). CPU `src/` is UNCHANGED (it's the reference). Verified locally with
  `jax[cpu]`: step-parity CPU↔JAX max|obs|≈1.2e-7 / max|reward|≈1e-20; indicator parity green; full
  pipeline (pmap PPO → held-out eval → Drive checkpoint+resume → ONNX) runs; 9/9 JAX tests + 22/22
  sampled CPU tests green.
- **C (Conclusion):** The same FTMO bot can now train at TPU scale — thousands of lifetimes at once —
  and only stops once it passes **40 challenges in a row on unseen data**, with every policy + its
  details saved to Drive. Matching the CPU env bar-for-bar means those TPU policies are ranked head-to-
  head with CPU policies by pass-rate, so the TPU buys consistency, not a different bot.

## [2026-06-28] JAX/TPU phase 2: shared-pot PortfolioEnv parity + LIVE FTMO-consistency progress
- **I (Issue):** Two operator follow-ups on the TPU trainer: (1) "I need to SEE progress relative to
  passing the FTMO challenge consistently AS training is going" — not just at the end; and (2) finish
  the whole project — the real product is the PORTFOLIO bot (one shared pot, the whole book), not the
  single-symbol foundation.
- **R (Rule):** Operator 2026-06-28. Parity is still the prime directive: the portfolio JAX env MUST
  match the CPU `src/env/portfolio_env.py` bar-for-bar (same 479 obs, same alpha-shaping/day-scoring/
  two-phase reward, same fingerprint). Live progress must be FTMO-consistency-focused (P(pass) + the
  40-in-a-row streak), not just loss curves.
- **A (Application):** (1) `jax_tpu/jax_progress.py` — a per-eval readout (`P(pass) @ 2.5%/4%` with
  trend arrows, held-out return, breach rate, and a `N/40 in a row` consistency bar) + a Colab
  `LiveDashboard` that redraws P(pass)/streak/breach curves every eval; wired into `jax_trainer.train`
  (a light heartbeat between evals, the rich readout + an `on_eval` callback at each eval). (2)
  `jax_tpu/jax_portfolio_env.py` — the shared-pot, symbol-cycling env as branchless jnp: per-symbol
  decisions cycling symbol-by-symbol over one pot, the alpha-shaping reward (USE/AGAINST/BEAT the
  firing-alpha consensus, PnL-capped), midnight day-scoring (won/failed day + the 4-in-a-row bonus),
  pot-level breach/+10% pass, and two-phase banking that flattens the WHOLE book (flatten unrolled over
  the static symbol count for exact tally/peak ordering parity). `jax_static_features.build_portfolio_static`
  stacks per-symbol static data + raw alpha tables; `jax_obs_blocks.portfolio_features_agg` is the
  pot-aggregate portfolio block. The trainer + eval were made env-agnostic (one `(init_state, reset_obs,
  step)` interface) so `train_portfolio()` reuses the whole pmap/40-in-a-row/Drive-checkpoint machinery.
  New gate `tests/test_jax_portfolio_parity.py` steps the CPU and JAX portfolio envs in lockstep across
  the symbol cycle and asserts obs+reward match — verified incl. FORCED two-phase banking, won days +
  4-in-a-row, and breach (max|obs|≈2e-7, max|reward|≈1e-10). Notebook gains a live-dashboard training
  cell + a "core goal" portfolio-training cell. 13/13 JAX tests green; CPU `src/` still UNCHANGED.
- **C (Conclusion):** The operator now WATCHES the bot get more consistent in real time (P(pass) and the
  40-in-a-row streak climbing), and the TPU trains the ACTUAL product — one bot balancing the whole FTMO
  book from one pot — proven identical to the CPU portfolio bot, so its policies rank head-to-head by
  pass-rate on the march to a consistent portfolio pass.

## [2026-06-28] Reward rebalance (seek the target, don't hide) + the live dashboard as a DIAGNOSIS
- **I (Issue):** Honest read of the design: the breach penalty (1.0) so dominates the tiny per-day rewards
  (0.025) that the easiest local optimum is to BARELY TRADE — a bot that "hides" (never breaches, but never
  makes +10% either). And the live readout was a scoreboard, not a diagnosis of WHICH failure mode is happening.
- **R (Rule):** Operator 2026-06-28: "reward it for SEEKING the target, not just avoiding breach — or you'll
  get a bot that hides," and "read the dashboard as a DIAGNOSIS (breach↑ = over-trading; breach≈0 + return≈0
  = hiding; P(pass) rising with controlled breach = learning)." Parity is still sacred: any reward change is
  mirrored in BOTH the CPU PortfolioEnv and the JAX env and re-verified bar-for-bar.
- **A (Application):** Two new reward terms in PortfolioEnv ONLY (single-symbol TradingEnv stays equity-only
  by design), config-gated + ON by default (`config/variables.py` + `FTMOConfig`): (1) **TARGET-SEEK** — a
  DENSE reward for NEW progress toward the +2.5%/day target (high-water-mark so it can't be farmed by
  churning), `target_seek_weight=0.10`, so "move toward the day's target" becomes the gradient and the bot
  actively SEEKS profit; (2) **ANTI-HIDE** — a penalty for a day the bot was FLAT the whole day (exposure-based,
  NOT close-count, so opening + holding across midnight is correctly NOT "hiding"), `idle_day_penalty=0.02`,
  so sitting out is no longer free. Implemented identically in `src/env/portfolio_env.py` and
  `jax_tpu/jax_portfolio_env.py` (new state: `day_progress_hwm`, `day_had_exposure`); CPU↔JAX portfolio parity
  re-verified to max|reward|≈2e-19. New `tests/test_portfolio_seek_idle.py` (seek rewards partial progress even
  on a failed day; idle penalises a flat day; a traded day avoids it); all 28 CPU portfolio tests green.
  DASHBOARD-AS-DIAGNOSIS: `jax_progress.diagnose()` classifies each eval as OVER-TRADING / HIDING / LEARNING /
  STRONG / DEVELOPING with actionable advice (which knob to turn), fed by a new `eval_mean_trades` activity
  metric (`jax_eval`), shown as a "DIAGNOSIS:" line in the readout + the LiveDashboard title.
- **C (Conclusion):** The reward now pulls the bot toward MAKING the target instead of merely surviving — the
  most likely "learns something useless (hides)" failure mode is directly countered — and the operator can read,
  live, whether it's hiding (raise seek/idle), over-trading (cut size / raise breach cost), or actually learning
  (P(pass) rising with controlled breach), and which knob to turn. This is the lever most likely to move the bot
  toward a consistent FTMO pass.

## [2026-06-28] JAX/TPU deep verification: stress-test driver + adversarial review found & fixed 3 bugs
- **I (Issue):** Before trusting the TPU trainer, do a DEEP test that everything works as intended (the
  float32/TPU path, real multi-device pmap, training stability, the 40-in-a-row eval gate, export).
- **R (Rule):** Operator 2026-06-28 ("deep test to make sure everything works"). Verify, don't assume.
- **A (Application):** Added `jax_tpu/tests/deep_test.py` — 14 checks, each in a clean subprocess with the
  right flags: float32 (TPU-path) parity single+portfolio, exact x64 parity over 6 seeds + 3 symbol
  combos, constant-action edge cases, vmap==scalar, determinism, finiteness (no NaN), **real 8-device
  pmap** (params + obs-norm synced across cores), training-LEARNS on a learnable task, resume continuity,
  eval/streak logic, ONNX-on-real-rollout, and fingerprint/obs-contract. Ran an adversarial CPU-vs-JAX
  bug-hunt (6 reviewers). Triaged: most "findings" were non-issues (per-minibatch advantage norm MATCHES
  SB3; mask-ones is correct under auto-reset; portfolio LOWs self-concluded "consistent"). THREE were real
  and fixed: (1) **eval could count a window as PASSED even if it breached after touching +10%** — would
  inflate the 40-in-a-row gate; fixed so a window passes only if it reached +10% AND never breached, and
  eval treats the challenge as ending at +10% (`jax_eval.py`). (2) **the obs-normalizer was not synced
  across TPU cores** (grads were pmean'd, norm wasn't) → on a real v2-8 each core would normalize
  differently and destabilize learning; fixed by pmean-ing the norm's batch stats (`jax_ppo.norm_update`
  axis_name), with a var>=0 clamp to avoid a sqrt-NaN on constant obs features. (3) **the single-symbol
  env didn't implement `open_gate`** (default off, but a silent divergence if enabled); implemented it
  (gate mask in the static tensor + `EnvParams.open_gate`) + a parity test. Plus hardening: an export
  shape-assertion and eval window-bounds clamping. Re-verified: 14/14 jax_tpu parity tests + 14/14 deep
  checks green. CPU `src/` still UNCHANGED.
- **C (Conclusion):** The deep test (and an adversarial review) turned up three real issues — including one
  that would have made the 40-passes-in-a-row consistency gate read falsely safe, and one that would have
  destabilized training on the actual 8-core TPU — all fixed and regression-tested. The TPU trainer is now
  verified to behave as intended end-to-end, so the march to a consistent portfolio pass rests on a
  trustworthy foundation.

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

## [2026-06-27] Training perf #1b: save features to disk (Google Drive) with a no-mismatch fingerprint
- **I:** Even after build-once+share, every run re-precomputes the per-symbol features (~minutes on the
  full history). The owner wants them SAVED (on Google Drive, which persists across Colab sessions, unlike
  local disk) and reused -- but "specify exactly what it's used for ... no mismatches": a naive
  (symbol+dates) key would silently load WRONG features after a code/data change = training on stale inputs.
- **R:** perf + safety; obs(479)/FTMO/`step()` unchanged. A 4-agent dependency audit first mapped EVERY
  input the precompute depends on so the fingerprint is complete (key finding: the existing
  `env_fingerprint` is INSUFFICIENT -- it hashes only SORTED alpha NAMES, missing slot order AND
  threshold/logic edits).
- **A:** New `src/data/feature_cache.py`. The cache KEY folds in: a CONTENT hash of the input arrays
  (close+indicators+time -> captures the data slice AND all indicator math), the resolved obs-contract
  values (contract version, MAX_STRATEGIES, asset classes, block sizes), the indicator-columns hash, the
  slot-ORDERED alpha roster, SOURCE hashes of the code that defines the features (strategies + signals +
  the env precompute + asset_specs -- catches logic/threshold edits names miss), the per-symbol asset spec,
  and the resolved open-gate / signal-accuracy values. Load ONLY on exact match, else rebuild. TradingEnv
  gains `export_precomputed()`/`_load_precomputed()` + a `precomputed=` fast path; `build_portfolio_subs`
  does load-or-build-and-save; `make_portfolio_vec_env`/`train_portfolio`/`run_training` thread a
  `feature_cache_dir` (default auto -> `MyDrive/Camillion/feature_cache` on Colab, else local; `--feature-cache off`
  to disable). Each cache folder is human-named (`SYMBOL__from_to__contract__key8/`) with a plain-English
  `manifest.json` ("exactly what this is").
- **Verified:** save→load round-trips to byte-identical obs; changed data / different symbol = MISS (never
  stale); cache keys stay in sync with the env's exported arrays; build_portfolio_subs reuses the cache on
  the 2nd call. +5 tests (`tests/test_feature_cache.py`); suite 168/168; audit ✅ GO 38/42.
- **C:** re-runs skip the slow build (load from Drive) with ZERO risk of stale features, and every cache is
  self-describing for the future. (Still TODO in #1: auto-calibrate workers/threads/device to ~70–80%.)

## [2026-06-27] Training perf #1c (part 1): auto-calibrate resources + honest utilisation report
- **I:** Owner wants training to use ~70–80% of a (possibly paid) Colab tier without freezing or wasting it.
- **R:** perf/UX; obs/FTMO/`step()` unchanged. **A:** new `src/training/autotune.py` detects CPU cores /
  RAM / GPU and picks a MEMORY-SAFE number of parallel copies (never over-subscribe RAM → never the freeze),
  a sensible compute-thread count (the tiny 3x256 MLP doesn't benefit from many), and the device (CPU by
  default — fastest for this net; GPU only if `prefer_cpu=False`). `train_portfolio` calls it, uses its
  `n_envs` when none is given, sets the PPO `device`, and applies the thread count. It prints a clear report.
- **HONEST LIMITATION (in the report + TRAINING_TASKS #1c part 2):** with the single-process DummyVecEnv path
  the market is stepped one copy at a time, so on a big multi-core box CPU use stays modest — true ~70–80%
  saturation needs a MULTI-WORKER (subprocess) upgrade where each worker LOADS its data+features from disk
  (now feasible thanks to the #1b cache, without the old pickle blowup). Scoped as the next step, owner's call.
- **Verified:** autotune returns sane, memory-safe settings (collapses to 1 copy on tiny RAM; CPU for the
  tiny model); end-to-end `train_portfolio` runs with autotune + device wired AND reuses the saved feature
  cache on a 2nd run ("loaded saved features ✓"). +4 tests; suite 172/172; audit ✅ GO 38/42.
- **C:** sensible, memory-safe resource use out of the box with a truthful utilisation picture; Phase 1
  (build-once+share, Drive cache, autocalibrate part 1) complete.

## [2026-06-27] Training perf #1c (part 2): TRUE multi-core training (workers load data+features from disk)
- **I:** Owner wants ~70–80% of CPU AND RAM used (don't waste paid Colab time). The single-process
  DummyVecEnv path steps the market on ~1 core. Going multi-process naively pickles the gigabyte dataset
  to every worker -> the original OOM hang.
- **R:** perf; obs/FTMO/`step()` unchanged. **A:** new top-level `make_portfolio_gym_env_from_disk()` builds
  a PortfolioEnv by LOADING its data (indicator cache) + features (the #1b feature cache) FROM DISK, so a
  SubprocVecEnv worker pickles only small strings -- no gigabyte pickling. `make_portfolio_vec_env` now: builds
  the features ONCE in the parent (populates the cache), then if multi-core is chosen starts N **worker
  processes** (`SubprocVecEnv`, `start_method="fork"` = Colab-notebook-safe) that each load from disk; else
  falls back to single-process DummyVecEnv (shared subs). `autotune` sizes the worker count to ~`target_util`
  of cores AND a memory-safe cap (reserves one env for the parent; collapses to single process when RAM is
  tight or <2 cores). `train_portfolio`/`run_training` thread `data_cache_dir`+`symbols`+`use_subproc`.
- **Verified:** forced 2-worker run spawns processes, each loads from disk (no pickle blowup), resets to
  (2,479) and steps cleanly with finite rewards; the from-disk worker builder is a cache HIT and yields a
  valid 479 obs (+test); single-process fallback intact. Suite 173/173; audit ✅ GO 38/42.
- **C:** on a big paid tier training now auto-scales to multiple cores (~70–80%) without the pickle-OOM,
  and safely drops to one process on small machines. Phase 1 fully complete.

## [2026-06-27] Training Phase 2: clean output — day-by-day pass metrics + balance, streamed (no spam)
- **I:** The training output was noisy (TensorFlow/CUDA/Gym banners + a per-rollout `...steps … HOLD 25%
  BUY 26%…` line) and the day-by-day table only printed at the very END. Owner wants ONLY the metrics that
  matter for passing -- each day as produced -- plus the account balance, and the action % removed (it's a
  readout, not a control; he found it confusing/alarming).
- **R:** output/UX; obs/FTMO/`step()` unchanged. **A:** (1) `run_training` silences the third-party banners
  before heavy imports (`TF_CPP_MIN_LOG_LEVEL=3`, gym/deprecation/future warning filters) and the
  `utcnow()` deprecation is fixed. (2) The heartbeat is now a SPARSE "training… X% done (~ETA)" tick (only on
  ~10% milestones — never a silent freeze, never spam) and the action-mix line is gone. (3) A new
  progress-check callback runs the day-by-day FTMO report on a FIXED test stretch with the CURRENT policy a
  few times during training and prints each day's `bal $… +P&L% +2.5%? DD% ok/BREACH breach` — so you watch
  the SAME test improve as it learns (loads features from the #1b cache → fast; best-effort, never breaks
  training). (4) `ent_coef` now ANNEALS 0.01 → ~0 over training, so the finished policy is fully decisive
  (nothing forces its action mix).
- **Verified:** end-to-end run shows the sparse tick + the live per-day table with balances (e.g. "Day 2
  2026-03-03 bal $97,740 -0.00% +2.5%? no DD 0.6% ok breach no"); `run_training` imports cleanly with the
  suppression; suite 173/173; audit ✅ GO 38/42.
- **C:** the screen now shows exactly what the owner asked for — day-by-day pass metrics + balance as they
  are produced — and the finished bot explores early but ends fully dynamic.

## [2026-06-27] Training Phase 4: fix the day-report trailing-drawdown math (chronological, engine-agreeing)
- **I:** A real run showed Day 1 `TRAIL_DD 5.14% → BREACH` in the `<WALL?` column while the actual engine
  said `breach: no` and kept trading -- a contradiction. Cause: the report computed trailing drawdown as
  `(max − min)` over the day, which pairs a LATER peak with an EARLIER trough and OVERSTATES it; real
  trailing drawdown is the drawdown from the RUNNING peak (chronological), which is how the engine breaches.
- **R:** report correctness; obs/FTMO/`step()` unchanged. **A:** `daily_report` now tracks a RUNNING peak
  (persists across days, like the engine's episode peak) and records each day's MAX drawdown-from-running-peak
  (`day_max_trail`) and max daily-loss-from-day-start (`day_max_loss`). New pure helper `running_drawdown_pct()`
  encodes + tests the chronological rule. So the `<WALL?` column now agrees with the real `breached` column.
- **Verified:** `running_drawdown_pct([100,95,110,104]) == 5.45%` (not ~15% under the old bug); simple cases
  (4% drop, only-rising=0) pass; +2 tests; suite 175/175; audit ✅ GO 38/42.
- **C:** the day-by-day table no longer flags phantom breaches — a `<WALL? BREACH` now means the engine
  actually breached.

## [2026-06-27] Training Phase 3: behavior to match the rules (DELIBERATE FTMO changes, owner-approved)
- **I/R:** Mark confirmed the trained portfolio bot's rules (see `TRAINING_REQUIREMENTS.md`). These change
  FTMO behaviour ON PURPOSE (CLAUDE.md rule #2 — stated explicitly here). Obs(479) + `step()` hot-loop rules
  unchanged.
- **A (PortfolioEnv + config + run_training):**
  1. **Bank +2.5% NET OF FEES** — the two-phase bank now triggers on `equity - pending_exit_cost` (what you
     actually KEEP after closing), so a banked day is genuinely ≥ +2.5% of initial.
  2. **Keep trading after banking** — `FTMO_PHASE2_CONTINUE` default flipped to **True**: bank +2.5% → close
     ALL → continue under the 1% trailing leash (was: stop). Updated the 4 stop/lock tests to force
     `phase2_continue=False` explicitly.
  3. **+10% / 4-in-a-row = BIG BONUS, keep training** — a per-day `_day_passed` flag + `_daily_pass_streak`;
     4 consecutive banked days adds a big `pass_bonus` reward AND training does NOT stop (new
     `continue_after_pass` flag, set True by `run_training`). EVAL / a real challenge still ends at +10%
     (`continue_after_pass=False`, the default, preserves the report/walk-forward behaviour).
  4. **Account size scales** — `run_training --balance` sets the starting balance; `build_portfolio_subs`
     calibrates each symbol's position size for THAT account (`calibrated_position_size(account=balance)`), so
     behaviour is identical at $10k or $200k.
  5. **Trailing wall is a dial** — `run_training --trailing-dd` sets the 4% wall (taper it over time).
- **Verified:** +5 tests (`tests/test_portfolio_behavior.py`): default phase2-continue ON · banked ≥ +2.5%
  net of fees · continue-after-pass doesn't terminate at +10% (while eval does) · a 4-day pass streak earns
  the big bonus · halving the account ~halves the position size. Suite 180/180; audit ✅ GO 38/42.
- **C:** the trained bot now behaves the way Mark trades it — bank the day net of fees, keep pushing under a
  tight leash, reward consistent 4-in-a-row passing, scale to any FTMO account, with a tunable risk wall.

## [2026-06-27] FIX: the daily-DD gauge the bot SEES now matches the actual breach (was blind to open losses)
- **I:** In `win_loss_features.daily_features`, `dd_used`/`risk_remaining` were built from `daily_realized_pnl`
  (CLOSED trades only): `daily_loss_frac = max(0, -pnl_frac)`. But `ftmo_rules.daily_drawdown_breached`
  fires on LIVE equity: `bal0 - acc.equity`. So while a losing trade was OPEN, the bot's "daily room left"
  gauge read SAFER than reality and the 5%-daily breach could fire with no warning the policy could see --
  blinding it to the #1 cause of FTMO failure. (Independently flagged by today's honesty-critic AND an
  external review.) **R:** observation-correctness; obs SHAPE unchanged (479), only this block's *meaning*
  is corrected to match its documented intent; no trained model exists yet.
- **A:** the daily DD now uses `max(0, bal0 - acc.equity)/bal0` (live equity, same base as the breach), so
  the gauge agrees with the engine. The daily *target* side already used live equity, so the block is now
  fully equity-consistent. Closed-trade `pnl_frac` is still shown separately.
- **Verified:** +3 tests (`tests/test_daily_dd_gauge.py`): a -3% OPEN loss now shows ~0.6 used (was 0.0); at
  -5% the gauge reads ≥1.0 exactly when `daily_drawdown_breached` is True; no phantom DD when flat/up.
  Suite 183/183; audit ✅ GO 38/42.
- **C:** the bot can finally SEE the daily wall approaching while a trade is open -> it can actually learn to
  cut losses before the daily breach, the most important behaviour for passing CONSISTENTLY.

## [2026-06-27] ALPHA-SHAPING in the portfolio reward (ON by default — operator decision; DELIBERATE departure)
- **I:** Operator wants the portfolio bot to both (a) USE the alphas and (b) BEAT them, expressed in the reward.
  **This is a conscious departure from the locked "reward = equity only / NEVER alpha" design rule** — which
  STILL holds for the single-symbol `TradingEnv` and its `test_reward_independent_of_alphas` (left untouched,
  still green). Recorded loudly here so it is not mistaken for drift. (NOTE: `CLAUDE.md`'s "reward = equity
  only" line is now superseded for `PortfolioEnv`; flag to update CLAUDE.md if desired.)
- **R:** owner-approved after the tradeoffs were laid out (leader-chasing risk; relative-vs-absolute; that the
  profit conditions are already rewarded by equity). Obs SHAPE unchanged (479); `step()` stays light (reads
  cached alpha arrays only); breach/FTMO numbers unchanged.
- **A:** `PortfolioEnv.step` adds three SMALL alpha-conditioned terms, gated by `cfg.alpha_reward_enabled`
  (True by default): (1) **USE** — bonus when a trade that AGREED with >=50% of the FIRING, unmasked alphas
  closes in profit with the day net up; (2) **BEAT** — bonus when a closed trade OUT-EARNED what following the
  alpha consensus would have made; (3) **AGAINST** — penalty for OPENING a trade against >=50% of firing
  alphas. **Every bonus is CAPPED at the trade's own PnL** (operator rule: "reward can never exceed the PnL")
  and only pays when the day is net up. Coefs in `config/variables.py` (FTMO_ALPHA_*), all 0.001 default; set
  `FTMO_ALPHA_REWARD_ENABLED=False` to restore the alpha-free reward.
- **Verified:** +5 white-box tests (`tests/test_portfolio_alpha_reward.py`) forcing a known consensus — agree
  bonus fires, against penalty fires, beat bonus fires, the bonus is CAPPED at the trade PnL (not the coef),
  and with the toggle OFF the reward is alpha-INDEPENDENT. `test_reward_independent_of_alphas` (TradingEnv)
  still green. Suite 188/188; audit ✅ GO 38/42.
- **C:** the portfolio bot is now trained to lean on the alphas when they pay AND to be rewarded for beating
  them — a deliberate, reversible, PnL-capped design choice. (Honest caveat: this trains toward consensus-use,
  which the `policy_doctor` leader-chasing detector exists to watch; judge it on a real run.)
- **[2026-06-28] BEAT bonus DOUBLED (0.001 → 0.002, operator).** Fixes the cancellation flagged earlier: with
  equal coefs a divergent win netted 0 (BEAT +0.001 − AGAINST 0.001) while an agreeing win netted +0.001 → a
  pure follower bias (the leader-chasing the doctor flags). At 2x, a (large) divergent win nets +0.001 =
  the agreeing win → the follower bias is REMOVED (now neutral). Conditions unchanged (profit + day-up +
  capped at PnL). To make it actively PREFER beating, set beat > 2x. Suite 188/188; audit GO.

## [2026-06-28] PER-DAY consistency scoring: "won day" = ENDS ≥ +2.5%, reward a won day / penalise a failed day
- **I:** Operator: "a won day is only at the end of the day it has 2.5% of initial or more" + "there needs to
  be a reward for passing the day AND failing the day." Before, a day counted as passed the moment it BANKED
  +2.5% intraday (`_day_passed` set at the bank) — so banking +2.5% then leaking it back to +1.5% under the
  phase-2 1% leash still counted as a "pass." That's the give-back hole the external review flagged.
- **R:** consistency shaping; obs(479)/FTMO numbers unchanged; deliberate (per operator). **A:** removed the
  intraday `_day_passed` flag; the day is now SCORED AT MIDNIGHT on its ENDING equity: `won = (equity −
  day_start) ≥ 2.5% of initial`. A won day adds `day_pass_reward` (0.025) and advances the streak (4-in-a-row
  still adds the big `pass_bonus`); a failed day subtracts `day_fail_penalty` (0.025) and resets the streak.
  Coefs in `config/variables.py` (FTMO_DAY_PASS_REWARD / FTMO_DAY_FAIL_PENALTY), tunable.
- **Why this is good:** (1) it makes the streak count REAL end-of-day wins, not transient banks; (2) it turns
  giving back a banked +2.5% into a FAIL → a real cost → directly addresses the phase2_continue give-back
  concern via the reward; (3) the fail penalty applies whether you trade or HOLD, so it pressures REACHING
  +2.5% rather than sitting — low freeze risk. (Honest caveat: too large a fail penalty could push reckless
  risk-taking to avoid failing → tune it; default 0.025 = the daily target magnitude.)
- **Verified:** +3 tests (`tests/test_portfolio_day_score.py`): a +3% day → +pass at midnight; a +0.5% day and
  a flat day → −penalty; the 4-in-a-row test still passes on the end-of-day logic. Suite 191/191; audit ✅ GO.
- **C:** the consistency signal now means what the operator intends — the bot is rewarded for FINISHING days at
  ≥ +2.5% and penalised for not, with give-backs correctly counted as fails.

## [2026-06-28] JARVIS cockpit reskinned to the dense "Stark Industries" HUD + added to repo root
- **I:** Operator: "update jarvis to look like this" (the classic Stark/JARVIS arc-reactor HUD reference) and
  "you can even use playhtml to make it more interactive." The `JARVIS Cockpit.dc.html` cockpit lived only in
  the DC editor; `docs/JARVIS_LIVE_WIRING.md` asks for it (+ `support.js`) at the repo root.
- **R:** UI artifact only — does NOT touch obs(479)/FTMO numbers/`env.step()`. The whole `Component` logic
  (state, methods, `renderVals` keys, the `/state` data contract the bridge wires to) is byte-for-byte
  behaviour-identical; only the presentation layer (CSS + decorative template markup + glyphs) changed.
- **A:** Wrote `JARVIS Cockpit.dc.html` + the matching `support.js` (DC runtime) to the repo root. Restyle:
  (1) shared "instrument-rose" ambient layer on all 3 views — dense radial tick-rings (conic-gradient) + a
  rotating radar sweep + concentric reticles, sitting behind every panel; (2) the central JARVIS core and the
  cockpit arc-reactor each gained Stark tick-rose rings; (3) faint edge telemetry rails (segment meters +
  oscilloscope strips) + a "CAMILLION INDUSTRIES · ARC-CLASS QUANT REACTOR" watermark; (4) tighter cyan/gold
  palette + denser blueprint grid; (5) all mojibake glyphs replaced with a clean monochrome geometric icon set
  (no colour-emoji — truer to the Stark look and renders everywhere); (6) command top-bar tightened so all 8
  controls fit. playhtml: two draggable, position-persisting "pin" tabs docked in the screen margins, added as
  body-level siblings of `<x-dc>` so the React reconciler never fights them (offline-safe — they degrade to
  static if unpkg is unreachable).
- **Verified:** embedded `Component` JS passes `node --check`; every `{{binding}}` resolves to a `renderVals`
  key (no "never resolved" warnings); zero mojibake; headless-Chromium render of all three views
  (command / cockpit / evolve) — no `pageerror`, all panels present, glyphs render, controls reachable.
- **C:** JARVIS now reads as the dense Stark arc-reactor HUD the operator asked for, the bot's live-wiring
  contract is untouched, and the cockpit + runtime are in the repo root ready for `uvicorn jarvis_bridge:app`.

### [2026-06-28] follow-up: fixed SETUP "GOT IT" not closing + full interaction proof
- **Bug found + fixed:** the SETUP modal's backdrop and its **GOT IT** button BOTH called `toggleInstr`; a
  click on GOT IT bubbled to the backdrop → `toggleInstr` ran twice (open→close→open) → the modal stayed
  open and its z-82 backdrop then swallowed every other click. Fix: a dedicated idempotent
  `closeInstr=()=>setState({instrOpen:false})` on the backdrop **and** GOT IT (double-call still nets closed);
  the top-bar SETUP button keeps `toggleInstr` to open. (Pre-existing bug, inherited from the source.)
- **Verified working (Playwright click-through, 29/29, 0 pageerrors):** all 3 nav paths + back; RUN DIAGNOSTIC
  IRAC cycle; module-card inspect; HELP / CLEAR / BRIEFING / DEBRIEF / SPEAK feed actions; mic + voice toggles;
  SETUP modal (open + GOT IT + backdrop close); ElevenLabs modal (open + cancel); daily-loss & max-DD sliders
  update their labels; MODE FTMO↔FREE toggle (resets to 5.0% on →FTMO); TRAILING + CCI gates; chat send
  (echo + canned reply); DREAM/REPLAY; EXPORT BRIEF downloads an .html; playhtml pins drag. (Note: assert via
  DOM `textContent`, not Playwright `inner_text`, which under-reports text inside the feed's scroll container.)
