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

### 1c. Auto-calibrate resources to ~70–80% CPU/GPU (no freeze, no waste) ⬜
- [ ] ⬜ Detect CPU cores, RAM, and GPU
- [ ] ⬜ Set thread count + a memory-safe number of workers (never over-subscribe RAM → never freeze)
- [ ] ⬜ Pick CPU vs GPU automatically (tiny brain usually prefers CPU — benchmark, don't pay for a GPU that does nothing)
- [ ] ⬜ Print a clear "Detected X cores / Y GB / GPU? → using …" report
- [ ] ⬜ **Profile** the real achievable utilization and tell Mark the honest number
- [ ] 💬 **Possible follow-up:** if the single-core market-stepping caps us below ~70–80%, do the deeper shared-memory multi-process upgrade to truly use a big paid tier

---

## 🧹 PHASE 2 — Clean live output ⬜
- [ ] ⬜ Remove the TensorFlow / CUDA / Gym warning noise
- [ ] ⬜ Remove the `...steps … HOLD 25% BUY 26%…` spam
- [ ] ⬜ Stream the **day-by-day pass metrics + account balance** as each day is produced (test on a fixed held-out stretch, so you watch the same test improve)
- [ ] ⬜ Keep one tiny "still working / % done" line so it's never a silent freeze
- [ ] ⬜ Make the exploration nudge **anneal to ~0** by the end so the finished bot is fully decisive/dynamic

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
