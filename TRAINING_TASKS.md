# 🗂️ Camillion — Training Build Tasks (tracker)

> **The checklist so we never forget anything.** Companion to **`TRAINING_REQUIREMENTS.md`** (the *what*);
> this file is the *how + status*. **Owner:** Mark (monty313) · **Updated:** 2026-06-27
>
> **Legend:** ✅ done & pushed · 🔄 in progress · ⬜ to do · 💬 needs Mark's decision

---

## ✅ Already done earlier — the learning/quality fixes (commit `0aeafe6`)
- [x] ✅ Day-by-day report no longer truncates the portfolio run (counted sub-steps)
- [x] ✅ Anti HOLD-collapse: entropy nudge back on (`ent_coef` 0 → 0.01)
- [x] ✅ Parallel copies are now diverse (random window + per-worker seed; + short-history fix)
- [x] ✅ The "+2.5% bank-and-stop" two-phase rule added to the **portfolio** bot (with tests)
- [x] ✅ Live action-mix readout added (later being replaced — see Phase 2)
- [x] ✅ Adversarial multi-agent review (1 MEDIUM found + fixed) · 163/163 tests · audit GO
- [x] ✅ `TRAINING_REQUIREMENTS.md` written & pushed (commit `8b67f25`)

---

## 🏗️ PHASE 1 — Speed, memory, never-freeze, no waste

### 1a. Build markets ONCE + share across workers + build progress bar ✅ (commit `5124f6b`)
- [x] ✅ Build per-symbol features once, share read-only across all workers (16 builds → 4)
- [x] ✅ Per-symbol build progress bar so it's never a silent hang
- [x] ✅ Verified: builds once not 16×, workers share one copy yet stay diverse; tests green

### 1b. Save features to Google Drive — with NO mismatches ✅ (commit pending)
- [x] ✅ Dependency audit (4 agents) mapped every input the features depend on
- [x] ✅ Complete **fingerprint** (data content-hash · contract version · indicator-columns hash · slot-ORDERED alpha roster · **source-hashes of strategy/signal/precompute/asset-spec code** · per-symbol spec · resolved thresholds)
- [x] ✅ Save prepared features to **Google Drive** (`feature_cache.default_cache_dir()` auto-detects Drive; persists across sessions)
- [x] ✅ Plain-English **`manifest.json`** ("exactly what this is") in each cache folder
- [x] ✅ **Mismatch guard:** load ONLY on exact fingerprint match, else rebuild (never stale) — tested
- [x] ✅ **Organized layout:** `feature_cache/SYMBOL__from_to__contract__key8/`
- [x] ✅ Drive path is a setting: `run_training --feature-cache <dir|off>` (default auto)
- [x] ✅ Tests: round-trip · stale/different-symbol rejected · keys in sync · build_portfolio_subs reuse (+5 tests, 168/168)
- [x] ✅ **DECISION:** Drive base folder = **`MyDrive/Camillion/`** (Mark chose recommended)
- [ ] ⬜ *Carry-over:* each saved **model** gets a manifest of which features+rules it trained on (do with Phase 3 / model-save)

### 1c. Auto-calibrate resources to ~70–80% CPU/GPU (no freeze, no waste)
**Part 1 ✅ (commit pending)**
- [x] ✅ Detect CPU cores, RAM, and GPU (`src/training/autotune.py`)
- [x] ✅ Memory-safe worker count (collapses to 1 copy on tiny RAM → never over-subscribes/freezes) + sensible thread count
- [x] ✅ Pick CPU vs GPU automatically (CPU for the tiny model; GPU only if `prefer_cpu=False`)
- [x] ✅ Clear "machine: X cores / Y GB / GPU … → using …" report; wired into `train_portfolio` (n_envs + device + threads)
- [x] ✅ Honest utilization note printed; +4 tests (172/172); verified end-to-end with cache reuse

**Part 2 ✅ (commit pending) — true multi-core (the actual ~70–80% on a big paid tier)**
- [x] ✅ Multi-worker (subprocess) training where each worker **LOADS its data + features from disk** (no pickle blowup), with a single-process DummyVecEnv fallback
- [x] ✅ autotune chooses processes vs single based on cores + RAM (memory-safe; reserves one env for the parent; collapses to single process on small machines)
- [x] ✅ Verified: 2 workers spawn + load from disk + step cleanly; from-disk worker builder tested (173/173); audit GO
- [ ] ⬜ *Profile on Mark's actual paid tier* to confirm the real % (can't measure true many-core here — 2-core box)

---

## 🧹 PHASE 2 — Clean live output ✅ (commit pending)
- [x] ✅ Removed the TensorFlow / CUDA / Gym warning noise (`run_training` silences before heavy imports) + fixed the `utcnow` deprecation
- [x] ✅ Removed the `...steps … HOLD 25% BUY 26%…` spam
- [x] ✅ Stream the **day-by-day pass metrics + account balance** during training (progress check on a fixed test stretch — watch the same test improve)
- [x] ✅ One tiny "training… X% done (~ETA)" tick on ~10% milestones (never a silent freeze, never spam)
- [x] ✅ Exploration nudge **anneals 0.01 → ~0** so the finished bot is fully decisive/dynamic
- [x] ✅ Verified end-to-end (live per-day table with balances); suite 173/173; audit GO

---

## ⚙️ PHASE 3 — Behavior to match the rules ⬜
- [ ] ⬜ **Fee-net banking:** bank the +2.5% only when *true* equity (after all fees) hits it; close ALL trades
- [ ] ⬜ **Continue after banking** under the 1% trailing leash (turn `phase2_continue` ON by default)
- [ ] ⬜ **+10% / 4 daily-passes-in-a-row = BIG BONUS** reward — and **keep training past it** (don't end the episode)
- [ ] ⬜ **Account size is an input**; position sizes **scale to the starting balance** (works at any FTMO size)
- [ ] ⬜ **4% trailing wall is a tunable dial** we can taper smaller over time
- [ ] ⬜ Tests for each of the above
- [ ] 🔮 *Future:* after banking, optional "keep going with the trailing stop cut in half" to protect profit

---

## 📊 PHASE 4 — Fix the day-by-day report ⬜
- [ ] ⬜ Fix the trailing-drawdown math to **peak-then-drop** (chronological), not (max − min) over the day
- [ ] ⬜ Make the report's breach column **agree with the actual engine** (no more "5.14% BREACH" that didn't breach)
- [ ] ⬜ Test the corrected drawdown on a known up-then-down sequence

---

## ✅ Done on EVERY change (guardrails)
- [ ] 🔒 Observation stays **479** (never silently change the shape)
- [ ] 🔒 FTMO numbers change **only** where Mark explicitly decided (the rules above)
- [ ] ⚡ Nothing slow (TA-Lib / MT5 / pandas) inside the trading loop
- [ ] 🧪 `python tools/run_tests.py` stays green · `python tools/run_full_audit.py` stays ✅ GO
- [ ] 📝 Every change logged in `docs/UPDATE_LOG.md`

---

## 💬 Open decisions (waiting on Mark)
- [ ] Drive base folder: **`MyDrive/Camillion/`** (recommended) vs `MyDrive/Camillion_data/`
