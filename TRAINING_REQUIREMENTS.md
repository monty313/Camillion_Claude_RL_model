# 🎯 Camillion — Training Requirements (what we expect)

> **The one-page agreement for how the bot should train.** Plain language, no jargon.
> **Owner:** Mark (monty313) · **Date:** 2026-06-27 · **Status:** ✅ Agreed — build against this.
> If we change our minds, we edit THIS file first, then the code. (Mirrors memory: `training-spec-refinements`.)

---

## 🧠 The mission
- 🤖 **ONE bot** trading the **whole portfolio** (EURUSD, GBPUSD, XAUUSD, US30 to start) from **ONE shared pot of money** — not four separate bots.
- 🌍 Built to **scale to every asset the broker offers** later, without a rebuild.

## 📈 The daily job
- 🎯 Aim for **+2.5% of the starting balance, every day.**
- 💵 **Fees count.** The +2.5% must be **real, after all transaction costs.** The instant true (post-fee) equity hits +2.5% → **close ALL trades and bank it.**
- 🏃 **Then keep pushing** (don't sit out the rest of the day): trade on under a **tight 1% trailing leash** — give back 1% of the banked peak → close & done for the day.

## 🔥 The real goal: consistency (not one lucky run)
- 🏆 **4 daily passes in a row (~+10%) = a BIG BONUS reward.**
- 🔁 **Training does NOT stop there — it keeps going.** We want the bot to repeat that streak **over and over.** Consistency is the whole point.
- *(A real, live FTMO challenge still ends at +10% — the "keep going" is only while training, to teach consistency.)*

## 🛡️ The hard rules it must NEVER break (FTMO)
- 🚧 Never lose **more than 4% from a peak** (the trailing wall).
- 🚧 Never lose **more than 5% in a single day.**
- 🚧 Never lose **more than 10% total.**
- 🎚️ The **4% trailing wall is a dial** — we want to **tighten it over time** (4% → smaller) as the bot gets good.

## 💵 Account size
- 🔢 **Starting balance is an input** (FTMO has ~5 sizes — e.g. $10k, $25k, $50k, $100k, $200k).
- ⚖️ The bot **scales its trade sizes to that balance**, so it behaves the **same at any size.**

## ✅ What "good enough to trust" means
- 🔒 Judged on **periods it has NEVER seen** (held-out data), not the data it practiced on.
- 🎯 The bar: it **reliably hits the +10% / 4-in-a-row pass — repeatedly**, with **zero rule breaches.**

## 👀 What you see while it trains (clean output)
- 🧹 **Removed:** the TensorFlow / CUDA / Gym warning noise **and** the `...steps … HOLD 25% BUY 26%…` spam.
- 📊 **Shown instead:** the bot run on a **fixed held-out test stretch**, printed **day-by-day as produced** — only the stuff that matters for passing + the **account balance**:

```
[training… 45% done]                         ← one tiny line so it's never a silent freeze

── test on held-out Jan 2024  (after 50,000 steps) ──
 Day 1  2024-01-02   bal $102,480   +2.48%   +2.5%? no    DD 2.6% ok   breach no
 Day 2  2024-01-03   bal $104,990   +2.45%   +2.5%? YES   DD 1.1% ok   breach no
 Day 3  2024-01-04   bal $107,560   +2.45%   +2.5%? YES   DD 0.9% ok   breach no
```

- 🔎 **About the action %:** that was just a *readout* of what the bot chose — **nothing forces it.** The bot is fully dynamic, and the small "keep exploring" nudge **fades to zero** by the end, so the finished bot is **decisive on its own.**

## ⚙️ Under the hood (so it can run on the FULL data, not just a slice)
- ♻️ **Build the markets once and share** them across the practice copies (today it rebuilds them 16× — that's the 5-hour "stuck building" hang).
- 📊 **Progress bar while building**, so it's never a silent mystery.
- 💾 **Save the built features to disk**, so future runs **skip the slow part** and start fast.

## 🧮 Use the machine smartly — no freeze, no wasted money 💸
- 🩺 **Never freeze silently again:** a watchdog/progress so a stall is *reported*, not a 5-hour mystery.
- 🎛️ **Auto-calibrate to the hardware:** detect the CPU cores, RAM (and GPU if present) and tune the number of parallel copies + threads to run at **~70–80% usage** — high enough to not waste paid Colab time, low enough to leave headroom so it never chokes or swaps.
- 🧠 **Memory-safe by design:** never spawn more copies than RAM can safely hold (over-filling RAM is what caused the freeze).
- ⚡ **GPU only if it actually helps:** the bot's brain is tiny, so on most tiers CPU is faster — benchmark and pick the faster device automatically (don't pay for a GPU that does nothing).
- 💳 Mark may pay for higher Colab tiers, so the bot must **make full use of whatever it's given** — without wasting it.

## 🔮 Future ideas (noted — not now)
- 🪙 After banking +2.5%, option to keep going with the **trailing stop cut in half** to protect profit — but the instant equity (after fees) hits the target, **close everything.**
- 🎚️ Gradually **tighten the 4% trailing wall** as consistency improves.

## 🚫 Invariants we must NEVER break (from `CLAUDE.md`)
- 🔒 The observation is **locked at 479 numbers** — never silently change its shape.
- 🔒 **FTMO numbers** only change when **Mark explicitly says so** (like the decisions above).
- ⚡ **Nothing slow** (TA-Lib / MT5 / pandas) inside the trading loop — speed is priority #1.

---

## 🗺️ Build order (we'll check these off together)
- [ ] **1. Speed & memory** — build-once + share + progress bar + **save-to-disk** + **auto-calibrate to ~70–80% CPU/GPU** (kills the hang; no waste; enables full data)
- [ ] **2. Clean output** — drop the noise; stream **day-by-day pass metrics + balance** (the mock above)
- [ ] **3. Behavior to match these rules** — fee-net banking · 1% leash continue · +10%/4-in-a-row bonus & keep training · account-size input + scaling · 4% trailing as a dial
- [ ] **4. Report fix** — correct the drawdown math so only a real >4% peak-to-drop counts as a breach

> 📌 Every change keeps the tests green, runs the audit (`python tools/run_full_audit.py`), and is logged in `docs/UPDATE_LOG.md`.
