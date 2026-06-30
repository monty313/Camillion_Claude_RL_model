# BUILD PLAN — teaching the bot to trade like Jordan (the principles, not the rules)

> Plain-English tracker so you always know what's done and what's next. Updated 2026-06-30 (v2: added the
> proof stage + honest "evidence-gated" labels after the external principle-learning critique).

## The goal (one sentence)
Teach the bot the **principles** of momentum/trading — so it generalizes to any instrument/session — instead
of hard-coding Jordan's exact CCI/SMA rules (which break when the market shifts).

## The honest trap we're avoiding
- Hard-coded **rules** → brittle. (We left this behind.)
- Hard-coded **concepts** (my 9 momentum features) → better, but still *my* theory of momentum. The bot can
  learn to trust my formulas instead of the market.
- **The fix is not more machinery — it's PROOF.** We build a measuring stick (Stage 6) and let it decide how
  much of the heavy stuff (auxiliary heads, preference model) is actually worth building. **Evidence first.**

## How we teach a principle (the levers)
1. **PERCEIVE** — give the bot the senses (features). 2. **PREFER** — reward the behavior (shape reward).
3. **RAILS** — a few hard "never" limits. 4. **CURRICULUM** — teach in order. 5. **PROVE** — test that it
generalized, not memorized. Master spec: **`JORDAN_PRINCIPLES.md`**.

---

## The staged plan (stricter version)

### ✅ Stage 0 — Define the principle (DONE — in `JORDAN_PRINCIPLES.md`)
Momentum = a state where directional pressure is **active**, **broad** across timeframes, **strong** enough to
persist, **cheap** enough to join without chasing, in **favorable** execution conditions, with **risk
controlled**. Split into: tradeability, bias, alignment, strength/exhaustion, location/structure, persistence,
decay. (These are the *questions*, not the answers.)

### ✅ Stage 1 — PERCEIVE: the momentum sense  ← BUILT + TESTED (uncommitted)
9 observation features, one per node of your momentum tree.
- [x] All 9 scores; obs 517 → **526** (v1.9.0); CPU + JAX byte-identical
- [x] 266 CPU + 3 momentum + 10 JAX parity + 2 trainer smokes (obs=526, diff 0.00) — all green
- [ ] **Commit + push** (waiting on you)

### ⬜ Stage 2 — PREFER: reward the principle (keep shaping WEAK)
- [ ] Reward cheap *continuation* entry (with-trend, pulled-back); mild penalty for *chasing* a late move
- [ ] More selective right after a loss (raise the bar, never the size)
- [ ] Small penalty for trading a dead/low-quality regime
- [ ] **Guardrail:** shaping stays small vs the real win-the-day/streak reward, or the bot games the proxy

### ⬜ Stage 3 — RAILS: only the structural "nevers" (keep MINIMAL)
- [ ] −4% wall + tiny fixed risk (have) · block trades that fight the higher-TF trend · chop gate (have)
- [ ] **Guardrail:** entry-quality / how-much-alignment / how-long-to-hold stay SOFT (learned), not hard masks
      — too many style rails = a boxed-in script that never learns

### ⬜ Stage 4 — CURRICULUM: teach in order (pass each before the next)
survive → bias → entry quality → persistence/decay → adaptive selectivity.

### ⬜ Stage 5 — TRAIN: RL fine-tune on the full stack
- [ ] Train across multiple instruments / sessions / volatility regimes
- [ ] **Randomize feature windows/thresholds slightly during training** so the bot can't anchor on one recipe

### ⬜ Stage 6 — PROVE it learned a principle (THE MISSING PIECE — build the measuring stick EARLY)
This is what tells us "Jordan" vs "refined heuristic machine." Cheap to build; it **drives every later spend**.
- [ ] **Ablation:** zero-out the 9 momentum features at eval — graceful degradation = good; collapse = it only
      learned dependence on my formulas
- [ ] **Parameter perturbation:** shift CCI/MA lengths, band widths — does it still seek the same *kind* of
      state? (principle) or fall apart? (recipe matching)
- [ ] **Instrument / session / regime holdout:** train on some, test on others — does the style transfer?
- [ ] **Counterfactual:** widen spread / weaken alignment / make the entry late / flip the HTF — do the
      action preferences shift the right way?
- [ ] **(Heavy, only if needed)** style-fidelity ranking against scripted-Jordan demonstrations

### 🔭 EVIDENCE-GATED upgrades (build ONLY if Stage 6 shows the simple version fails)
- [ ] **Outcome-based auxiliary heads** — predict the *future* (did the move persist? was the excursion good?),
      NOT my own scores (that's circular). Forces the encoder to learn real abstraction.
- [ ] **Preference / reward model** — rank trajectory *snippets* for "Jordan-likeness," train a style model,
      use it as reward. (Heavy: needs ranking labels — bootstrap from scripted-Jordan demos.) See
      `JORDAN_PRINCIPLES.md` "Learning method."
- [ ] Add 15m/1h timeframes.

---

## Decisions already made (override any)
- **Size stays fixed** (a rail). Conviction changes take/skip, never bet size.
- **Priority:** survive > don't-trade-chop / no-counter-trend > protect-the-streak > selectivity > let-it-run.
- **+2.5% auto-bank:** soft protection ON (helps the 40-streak), not a hard stop.
- **Shaping weak, rails minimal** (per the critique — avoid reward-hacking and a boxed-in style).

## The recommended order from here (evidence-driven)
1. **Commit Stage 1** (it's a strictly-better representation; no reason to sit on it).
2. **Build Stage 6's measuring stick** (ablation + holdout eval) — *before* long training, so we can read it.
3. **Baseline train** (Stages 2–4 light) and **read Stage 6**.
4. **Only then** decide on auxiliary heads / preference model — driven by the numbers, not by speculation.

## Where we are RIGHT NOW
Stage 1 built + green, **uncommitted**. The real next investment isn't more features — it's the **proof
harness** (Stage 6), so we never again confuse "the plumbing works" with "it learned the principle."
