# JORDAN_PRINCIPLES.md — teach the agent the PRINCIPLES, not the rules

> **The supervision document.** Source: `OPERATOR_PROFILE.md` (the interview) + `PSYCHOLOGY.md` (the vision).
> This file is what the engineer builds from. It converts Jordan from a brittle checklist into an RL agent that
> **generalizes his style** to new instruments, sessions, and regimes.
>
> **The mental model (from the operator's own framing):**
> 1. **Rules are examples.**  2. **Principles are invariants.**  3. **Reward + observations teach the
> invariants.**  4. **The policy learns how Jordan expresses them in new states.**
>
> So the 15m SMA, 1h highs/lows, BB pulls and CCI ladders in the profile are **feature generators and
> demonstration scaffolding** — NOT final hard-coded law. We do not script "buy when 5m closes above the 1h
> high." We teach: *"when the higher-timeframe direction is favorable, lower-timeframe momentum confirms, and
> the entry is cheap, continuation is desirable" — and let the policy decide where that's true.*

---

## The principles (invariants — these are what we engrain)

- **I1 — Survive first.** Tiny per-trade risk; the daily loss limit is sacred. Profit comes from many good
  states, never from hero trades.
- **I2 — Don't trade chop.** No momentum / no edge → stand aside. A protected zero beats a forced loss.
- **I3 — Bias first, execution second.** Only trade *with* the higher-timeframe direction; never counter-trend.
- **I4 — Alignment over prediction.** When more timeframes agree, continuation is more likely than reversal —
  prefer aligned momentum over isolated signals.
- **I5 — Cheap continuation, not late extension.** Enter where momentum *resumes* cheaply (pullback into the
  trend-side band), not where the move is already exhausted.
- **I6 — Selectivity, and tighten after failure.** Prefer the strongest available setup; after a loss, raise
  the bar (more confirmation) — never raise the size.
- **I7 — Manage, then let it breathe.** Bank some, protect the rest (breakeven), let a real winner run while
  momentum lives; exit when momentum dies.
- **I8 — Trade only quality regimes.** High liquidity / thin spread / active session — stand down otherwise.
- **I9 — Consistency over heroics.** Win a little, clean, every day; protect the streak; no behavioral drift.

---

## How each invariant is taught (the four levers)

| Invariant | PERCEIVE (state feature the policy needs) | PREFER (reward shaping) | RAIL (hard limit) | Status |
|---|---|---|---|---|
| I1 Survive | dd-to-wall (have) + **loss-streak / give-back rate** | breach −20, dd-proximity (have) | **−4% daily stop; fixed tiny risk** (have) | rail ✓ + add early-warning perception |
| I2 No chop | 5m CCI(30/100) (have) + **momentum-quality/slope** | idle vs seek (have) | **open-gate: both 5m CCI in ±50 → no open** (have) | rail ✓ + add perception |
| I3 Bias-first | multi-TF SMA (have) | reward trend-aligned entries | **no counter-trend to the trend filter** (build mask) | add reward + rail |
| I4 Alignment | multi-TF CCI (have); **a clean "alignment" scalar** | conviction scales with aligned signals (have) | — | mostly ✓; sharpen |
| I5 Cheap continuation | band location (partial, trade_risk) + **extension-vs-pullback** | **reward pullback-entry; penalize late-extension chase** | — | **build** |
| I6 Selectivity / tighten | conviction count (have, absolute) + **since-loss "tighten" state** | conviction reward (have); **post-loss caution** | "never size up after a loss" | **build perception + reward** |
| I7 Manage / let run | trade_risk MFE/MAE/band (have) | let-winners-run vs +2.5% bank (have) | — | tune |
| I8 Quality regime | session one-hot (have); **spread / liquidity** | reward trading good regimes, stand down in bad | **spread veto** (if data) | **build / needs data** |
| I9 Consistency | won-day streak + rate (have, v1.8.0) | escalating streak (have); stretched gamma (have) | — | ✓ |

> **Most of the perception + reward is already in the bot** (we built it over the last sessions). The genuine
> *new* work is a short list of **perception features Jordan reacts to that the policy still can't see** — and
> a few reward terms that turn his preferences into gradients.

---

## Perception layer — what Jordan NOTICES (state features)

Already present: multi-TF SMA (trend), multi-TF CCI 30/100 (momentum), BB(20/200/10) + band-stack + band
distance (extension), OHLC (candle thrust), session one-hots, won-day streak/consistency, re-entry context.

**To ADD (the real gap — a contract bump, append-only, v1.8.0 → v1.9.0):**
1. **Structure distance** — normalized distance from price to the nearest recent swing high / swing low (so the
   *breakout/continuation* principle is learnable as "near/through structure," not a 1h-high formula).
2. **Momentum quality / slope** — is momentum *building or fading* (the change in CCI, candle-size trend) — so
   the policy can tell "wait, it's coming" from "fading, stand down," and detect momentum *death*.
3. **Extension-vs-pullback** — where in the move are we (riding the band = extended vs pulled-back-to-mean =
   cheap entry) — turns I5 into perception.
4. **Since-loss / tighten state** — bars/trades since the last *loss* + a "tighten-active" flag (I6's trigger
   AND its reset) — so "raise the bar after a loss" is learnable.
5. **Spread / liquidity** — *if the data carries it*, real spread + a liquidity read (I8). If the CSVs have no
   bid/ask, use a volatility/range proxy and lean on the session one-hots — flagged as an approximation.

---

## Preference layer — what Jordan FAVORS (reward shaping, not rules)

- **Continuation > reversal; cheap entry > late chase** → reward an entry taken on a pullback into the
  trend-side band; mildly penalize chasing extension. *(new — uses the extension/pullback perception.)*
- **Aligned momentum > isolated signal** → already the conviction reward (scales with aligned signals).
- **Reduce activity in chop / wide spread** → open-gate (have) + a small "traded a low-quality regime" penalty.
- **Tighten after a loss** → a small post-loss caution shaping (and/or a learned response to the since-loss
  feature). *(new.)*
- **Behave like Jordan even before profit** → the reward already pays for *good behavior* (seek the target,
  trade with conviction, don't hide, protect the streak), not only realized P&L — exactly the principle-shaping
  this calls for.

---

## Risk / rail layer — the few HARD invariants (action masks / clamps)

These are the only things hard-coded, because Jordan treats them as absolute:
- **−4% daily equity stop** — the wall (have, as the 4% trailing limit).
- **Tiny FIXED risk per trade** — size is a **rail, NOT conviction-scaled** (Jordan *never* varies size;
  conviction changes *take/skip*, not bet size — decision below).
- **No counter-trend to the higher-TF trend filter** — *(build: a directional action mask.)*
- **No trading the chop** — the 5m-CCI open-gate (have).
- **Spread/session veto** — stand down in bad regimes *(build, if data).*

---

## Curriculum — the order we engrain (so survival is learned before aggression)

1. **Survive + don't-trade-chop** (rails + dd-proximity) — first, non-negotiable.
2. **Bias-first + aligned momentum** (trend mask + conviction).
3. **Cheap continuation entries + structure** (the new perception + the continuation reward).
4. **Selectivity + management + the streak** (conviction, tighten-after-loss, escalating streak, stretched
   gamma).

(Mechanically: keep the strong survival rails on from step 1; phase in the shaping weights; the stretched
gamma already lets late-curriculum streak-protection take hold.)

---

## Decisions I made (you said "you choose" — these are overridable)

1. **Size stays a fixed rail.** Jordan never varies size, so conviction changes **take vs skip vs wait**, not
   bet size. (Keeps survival clean; avoids teaching the bot to over-bet on "conviction.")
2. **Priority hierarchy (sets reward magnitudes):** survive (−4%) > don't-trade-chop / no-counter-trend >
   protect-the-streak / win-the-day > selectivity > let-it-run / press. (Matches the current magnitudes:
   breach 20 > day 10 > streak/seek > conviction.)
3. **+2.5% auto-bank:** keep it **ON as a soft protective rail** (it helps the 40-streak), not a hard "stop for
   the day" — the bot may keep trading clean momentum under the −4% wall, which is closer to "pure Jordan."
   One knob; revisit from the dashboard.
4. **News:** default **soft avoidance** (no hard blackout) until/unless we wire a news calendar; revisit.
5. **Timeframes:** teach the *principle* across the engine's 1m/5m/30m/4h/1d + the new structure/momentum-slope
   features; **do not** add 15m/1h yet — earn that big build only if training shows the bot can't express the
   edge without them.

---

## The build (staged) — what we actually do

- **Stage 1 — PERCEPTION (contract bump v1.9.0, append-only):** structure-distance, momentum-slope/quality,
  extension-vs-pullback, since-loss/tighten state, spread-or-proxy. *Without these senses the policy literally
  cannot learn Jordan's continuation/structure/tighten principles.* Verified CPU↔JAX bar-for-bar, like every
  prior block.
- **Stage 2 — REWARD:** continuation/cheap-entry preference, tighten-after-loss caution, low-quality-regime
  penalty. (Magnitudes per the hierarchy above.)
- **Stage 3 — RAILS:** counter-trend action mask; spread/session veto (if data).
- **Stage 4 — CURRICULUM:** stage the shaping weights per the order above; train.

> Each stage is a normal, parity-verified change to obs / reward / rails — the same loop we've run for the
> trade-risk, consistency, conviction, and open-gate work. The difference now is the *intent*: every change
> teaches an **invariant**, so the agent becomes "Jordan in a new state," not a replay of Jordan's last setup.

---

## Learning method (research-grounded — and honest about THIS codebase)

The literature (imitation/behavior-cloning, max-entropy **IRL**, **preference-based RL / RLHF**, reward
modeling, reward shaping) all converges on one thing: **move structure into features + preferences + reward,
not if/then laws.** We agree. Honest mapping of each method to what we can actually do here:

- **Feature design (perception)** — *feasible now, and the prerequisite for EVERY method.* You can't learn a
  preference over a state you can't see. → **Stage 1.**
- **Principle reward shaping** — *feasible now; already in the bot* (conviction, open-gate, escalating streak,
  dd-proximity, continuation). → **Stage 2.**
- **PPO fine-tuning under hard rails** — *our trainer.* → **Stage 4.**
- **Behavior cloning / IRL from demonstrations** — needs a dataset of Jordan's **real** trades (state→action).
  We don't have one. **Bridge: SYNTHESIZE the demonstrations** — script Jordan's rules as a deterministic
  policy, run it in the sim to produce demonstration trajectories, then use them to (a) **warm-start** the
  policy (a style prior to fine-tune *beyond*, not copy) and (b) auto-generate **preference pairs**
  (scripted-Jordan trajectory > random/early-PPO) to shape the reward toward "Jordan-like." This gives the
  demonstrations the research wants — from the profile, with **no logged trades and no manual labeling**.
- **Preference-based reward MODEL (full RLHF)** — powerful, but needs Jordan to label *thousands* of
  trajectory-pair comparisons + a separate reward-model pipeline. A real upgrade but a **bigger program**;
  defer until Stages 1–4 are in and we see where they fall short. (The synthetic-demo preferences above
  bootstrap it cheaply when we get there.)

**What we AVOID (per the research):** pure behavior cloning (overfits Jordan's exact actions, no intent), pure
P&L maximization (reward hacking on a shallow reward), and hard if/then rules (brittle, single-ontology).

**The decided teaching stack:** Stage 1 perception → Stage 2 principle reward shaping → **Stage 2.5 (optional)
scripted-Jordan demonstrations: warm-start + preference shaping** → Stage 4 PPO under hard rails + curriculum.
Heavy IRL / full RLHF reward model: **deferred** (needs data or labeling we don't yet have).
