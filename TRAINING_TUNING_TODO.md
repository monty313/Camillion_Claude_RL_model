# 🧪 Camillion — Consistency-Tuning To-Do (with the *why / why-not*)

> **Purpose:** the backlog of reward/training changes aimed at making the bot pass FTMO **consistently**,
> each with the honest reason to do it — or NOT do it (yet). Companion to `TRAINING_REQUIREMENTS.md` (the
> what) and `TRAINING_TASKS.md` (the build log). **Owner:** Mark · **Updated:** 2026-06-27
>
> **Legend:** ✅ do · 🕒 do-but-later (sequenced) · 🟡 defer · 🔴 skip · 🚪 gate (must happen first)

---

## 🚪 GATE #0 — Run the REAL baseline on Colab BEFORE tuning anything
**Status: do this first.** **Why:** we just A/B-tested the tuning ideas on synthetic data and **neither config
learned to make a single +2.5% day** (see evidence below). You cannot tune windows/gamma/penalties on a bot
that hasn't been shown to learn to trade profitably *at all*. The first question is **"does it learn to trade
profitably on REAL data?"** — and only real market data has a real edge to learn (my toy didn't carry one
the bot could exploit in the budget). **Why it matters:** every item below is premature until the baseline
either (a) trades and makes *some* +2.5% days but lacks *consistency* → then the tuning is justified, or
(b) doesn't trade/learn at all → then the fix is the learning setup, not these knobs.
- [ ] Run `run_training.py` on a real `--from/--to` slice, enough steps; read the live day-by-day + action mix.
- [ ] Decide from what we SEE: is the problem *consistency* (tune below) or *learning at all* (different fix)?

---

## 🧾 EVIDENCE — the A/B we ran (2026-06-27) and why it's inconclusive
Same toy market (a clean, learnable momentum edge), same 120k-step budget, same seed; only window/gamma/n_steps differ:

| config | days | hit +2.5% | inside 4% wall | breaches | max streak | final % | still trading? |
|---|---|---|---|---|---|---|---|
| BASELINE (5k window, γ0.997, n2048) | 9 | **0** | 9 | 0 | 0 | +0.33% | yes |
| BATCH_A (14.4k window, γ0.999, n4096) | 5 | **0** | 5 | 1 | 0 | −3.03% | yes |

**What it shows (honest):**
- **Inconclusive on "does Batch A help."** *Neither* bot learned to make a +2.5% day, so there's no working
  baseline to beat. The −3% for BATCH_A is **one seed, within RL noise — NOT evidence it's worse.**
- **Fair caveat for BATCH_A:** longer episodes + higher gamma are **harder to train and need MORE steps to
  converge**, so an equal *small* (120k) budget is *unfair* to it — it wasn't given enough to show its benefit.
- **Both still traded** (balanced action mix) — neither change caused a HOLD-collapse. Good.
- **Limits:** toy data, **1 seed, 1 symbol, 120k steps** → tests mechanics, **not** real-FTMO passing. A toy
  win wouldn't prove real improvement, and this toy null-result doesn't condemn the ideas — it condemns
  *validating them on a toy*. Hence GATE #0.

---

## ✅ / 🕒 The backlog (sequenced — ONE lever per training run)

### 1. 🕒 Batch A — longer random windows + higher gamma  *(approved; do after the baseline)*
- **Change:** `WINDOW_LENGTH_BARS` 5,000 → **~14,400** (≈2 weeks); `gamma` 0.997 → **0.999**; `n_steps` 2048 → **4096**.
- **Why:** the 4-in-a-row streak bonus is **structurally unreachable** in a 3.47-day window (it needs 4 full
  days; the streak resets each episode). And gamma 0.997 over 4 symbols = ~**1.4h** of foresight, so a wall a
  day away is discounted to ~3e-8 — the consistency reward **can't propagate across days**. Longer windows let
  it *experience* multi-day consistency; higher gamma lets that reward *reach back*. (n_steps↑ keeps PPO stable
  at the higher gamma.)
- **Why not yet / risk:** no freeze risk (A/B confirmed both still trade), but it **needs a bigger step budget**
  to converge, and tuning it before the baseline learns is premature. Cost: ~3× longer episodes → lower throughput.

### 2. 🕒 DD-proximity penalty — put risk INTO the gradient  *(highest-value reward change; do ALONE)*
- **Change:** add a small, **graduated** penalty as live equity approaches the **4% trailing** and **5% daily**
  walls (using *live* equity, matching the engine + the Bug #1 gauge fix).
- **Why:** today **drawdown is free until the cliff** — the only risk signal is the −1.0 terminal breach. So the
  gradient literally rewards riding *toward* the wall. This makes "respect margin" something it *learns*, not
  just *sees*. This is the real cure for the #1 FTMO failure (daily DD).
- **Why not stacked / risk:** a reward penalty **can freeze the bot** (penalize drawdown too hard → it stops
  trading). So it goes in its **own** run, small magnitude, tuned, with an action-mix check that it still trades.
- **Note:** target the **4% trailing** wall (the one that bites first), *not* the 10% total — trailing is tighter
  whenever the account isn't already deep underwater.

### ✅ DONE (2026-06-27) — ALPHA-SHAPING in the portfolio reward (ON by default, operator decision)
- **Built:** USE bonus (profitable close that agreed with ≥50% firing alphas) + BEAT bonus (closed trade
  out-earned following the consensus) + AGAINST penalty (opening against ≥50%); **every bonus CAPPED at the
  trade's PnL**, only pays when the day is net up. Toggle `FTMO_ALPHA_REWARD_ENABLED` (default True).
- **Deliberate departure** from "reward = equity only" (still holds for single-symbol TradingEnv). +5 tests; 188/188; audit GO.
- **Honest watch:** this trains toward *consensus-use* — the very thing `policy_doctor`'s leader-chasing
  detector flags. Judge it on a REAL run; if it's just following the alpha majority without beating it, reconsider.

### 3. 🟡 SMA-cross-while-losing penalty  *(defer; prefer observation over reward)*
- **Idea:** penalize holding a losing trade that **crosses the 1m 30-SMA against its direction**.
- **Why NOT (as a reward):** (a) it **bakes a tactic into the reward** instead of the goal — RL should *discover*
  exits; (b) it **fights mean-reversion alphas** (a reversion entry is *supposed* to be underwater and on the
  wrong side of the SMA before it reverts); (c) it **overlaps the DD penalty** (both target "stop bleeding in
  losers") → stacking risks double-punishing and freezing the bot; (d) risks teaching it to **cut winners early**.
- **If wanted:** add the SMA-cross state as an **observation feature** (free — lets it *learn* to use it) rather
  than a reward term; or trial it as a *separate, later* lever only if the DD penalty alone doesn't fix loser-holding.

### 4. 🟡 NY-session bonus in the portfolio env  *(defer)*
- **Why NOT (now):** it's a **session-timing nudge, not a consistency lever**; in the shared pot it needs fiddly
  **P&L attribution** to the index symbol (the single-symbol version assumed the index = 100% of session P&L);
  the bot **already sees the NY-session flag** in its time block and can learn to prefer it if it pays; and adding
  it now just **muddies** whether the real consistency fixes (1 & 2) worked. Do later if you specifically want
  the bias, on its own.

### 5. 🔴 Surface `day_locked` in the observation  *(skip)*
- **Why SKIP:** low value — with `phase2_continue=True` the bot is rarely *fully* locked, and it can already infer
  "day's won" from `target_progress ≈ 1.0`. And a clean flag means a **479→480 contract bump** (rebuilds every
  feature cache, updates the contract docs + shape tests). Not worth it. (Do **NOT** do the rejected hack of
  overloading `risk_remaining=0` when locked — it conflates "safe, done" with "about to die" and would corrupt
  the wall-awareness we just fixed.) Revisit only if a *trained* model visibly mis-behaves in locked bars.

---

## 🧭 Recommended sequence
1. 🚪 **Colab baseline** (real data) → see if it learns to trade + where it fails.
2. 🕒 **Batch A** (windows + gamma + n_steps) — safe, no freeze risk; needs a real step budget.
3. 🕒 **DD-proximity penalty** — *alone*, small, watch the action mix.
4. 🟡 then, only if still needed: SMA-cross (as observation first) and/or NY bonus — each alone.

## ⚖️ The discipline rule (why we don't big-bang)
**One lever per training run, then read the day-by-day + action mix before the next.** Reward shaping is where RL
projects quietly die: a too-strong penalty → the bot stops trading; two penalties at once → you can't tell which
broke it. Change one thing, retrain, confirm it still trades and improved, *then* move on.

> 📌 Bottom line: the toy A/B proved we **can't** validate these on synthetic data — the gate is a real Colab
> run. Until then, items 1–2 are *staged and ready*, 3–4 are *deferred*, 5 is *skipped*.
