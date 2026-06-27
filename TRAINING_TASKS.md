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

### 1b. Save features to Google Drive — with NO mismatches 🔄
- [🔄] Dependency audit: find EVERY input the features depend on (so the fingerprint is complete) — *running now*
- [ ] ⬜ Build the **fingerprint** (symbols · date range · bar count · **obs contract version** · **strategy/indicator version** · **content hash of the data** · build date)
- [ ] ⬜ Save the prepared features to **Google Drive** (persists across Colab sessions; local disk gets wiped)
- [ ] ⬜ Write a plain-English **`manifest.json`** in each cache folder ("exactly what this is + which policy/rules it pairs with")
- [ ] ⬜ **Mismatch guard:** on load, recompute the fingerprint from current code+data; load only if it matches **exactly**, else **rebuild** (never load stale)
- [ ] ⬜ **Organized Drive layout** (`feature_cache/`, `models/<policy>/`) — see `TRAINING_REQUIREMENTS.md`
- [ ] ⬜ Each saved **model** gets a manifest recording which features+rules it was trained on (model never paired with wrong features)
- [ ] ⬜ Drive path is a **setting** (default chosen below) you can change
- [ ] ⬜ Tests for: fingerprint changes on any real change · stale cache is rejected · save→load round-trips
- [ ] 💬 **DECISION (Mark):** Drive base folder → new **`MyDrive/Camillion/`** *(recommended)* or next to data under **`MyDrive/Camillion_data/`**?

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
