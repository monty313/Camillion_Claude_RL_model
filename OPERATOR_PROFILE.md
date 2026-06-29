# OPERATOR PROFILE — how the trader thinks (the bot's calibration source)

> **Purpose.** Capture *how the operator trades and thinks* — in concrete, encodable detail — so we can
> calibrate the bot (its **alphas/signals**, **reward shaping**, **observation**, **risk rails**, and
> **exits**) to trade like *him*, not like a generic system. Pairs with `PSYCHOLOGY.md` (the vision). This is
> a living doc: **re-read or re-fill it anytime to re-calibrate the bot.**
>
> **The loop:**
> 1. Run the **Perplexity prompt (Part A)** — it interviews you and produces a thorough profile.
> 2. Paste Perplexity's result into **Part C**, and/or answer **Part B** directly.
> 3. Hand it back to the engineer (Claude). Each answer maps to a bot lever (the `calibrates →` tags show how),
>    and I translate it into reward/obs/alpha/rail changes — then we verify (CPU↔JAX parity) and train.

---

## Part A — the Perplexity prompt (copy-paste this whole block into Perplexity)

> Paste everything between the lines. Answer its questions as they come; when it's done it gives you a
> structured profile — copy that into **Part C** below and bring it back to me.

```
You are my trading-psychology analyst and systems interviewer. Your job: deeply understand HOW I trade and
think, then write a thorough, structured profile an engineer can turn into rules.

CONTEXT — what we're doing and why:
I (with an engineer) am building an AI reinforcement-learning trading bot for FTMO-style prop-firm challenges.
The rules it trades under: make +2.5% of the account per day; never breach a 4% TRAILING drawdown (or 5% daily
/ 10% total); and the real goal is 40 WINNING DAYS IN A ROW — a winning day ends at >= +2.5%, and a losing day
OR a breach resets the streak to zero. The bot trades several instruments from ONE shared account, one decision
at a time, with actions {hold, buy, sell, close}. I want the bot to trade the way *I* do — my setups, my
entries, my exits, my risk rules, my market read, my discipline. To program that, my engineer needs a precise,
concrete map of how I think — not vague philosophy, but the exact rules and instincts I use.

YOUR JOB:
1. INTERVIEW me, a few questions at a time (never dump a huge list — ask 2-4, wait for my answers, then follow
   up). Make it a real conversation.
2. DRILL for concrete, encodable detail. If I say "I wait for confirmation," ask "confirmation of WHAT, on
   WHICH timeframe, at WHAT value?" Push past vague answers; ask me to walk through 2-3 real trades (a winner,
   a loser, and one I correctly skipped) step by step.
3. COVER all of these areas, in roughly this order:
   (a) GOAL & RISK — daily target, max risk per trade and per day, what a "good day" and a "bad day" are, how
       big a single loss can be.
   (b) WHAT & WHEN — instruments, sessions/times of day, the market CONDITIONS I trade vs the ones I avoid, and
       how I decide a market is even tradeable today.
   (c) SETUPS / ENTRIES — my exact A+ setup(s): what MUST be true to pull the trigger, which indicators /
       levels / timeframes (be specific with values), how many confirmations I need, whether long and short
       are symmetric, and what makes me NOT take a trade (chop, news, no momentum).
   (d) MANAGEMENT & EXITS — where my stop goes and whether I move it (breakeven / trail), where I take profit
       (fixed target / trail / scale-out), do I let winners RUN or take quick profit, my target reward:risk, do
       I add to winners or re-enter a runner.
   (e) SIZING — fixed % or conviction/volatility-based; do I bet BIGGER on the best setups; how I handle
       several positions that are really the same bet (correlation).
   (f) PSYCHOLOGY & DISCIPLINE — what I do after a loss, after a win, on a winning streak, and on a dead /
       no-trade day; my single biggest leak; the rules I NEVER break.
   (g) EDGE & BELIEFS — WHY my setups work, who's on the other side of my trade, and how I'd know my edge has
       stopped working.
   (h) STREAK MINDSET — would I take a small or a flat/zero day to PROTECT a long streak, and when (if ever)
       would I break my own rules?
4. Keep going until you have a COMPLETE, unambiguous picture. Where my answers contradict each other, point it
   out and make me resolve it. Tell me when you believe you have enough.

FINAL OUTPUT — when the interview is done, produce this as Markdown:
- TRADER SUMMARY — one paragraph: who I am as a trader.
- DECISION PROCESS — a numbered, step-by-step checklist / flowchart, from "is the market tradeable today?" ->
  "is this an A+ setup?" -> "I'm in" -> "manage it" -> "I'm out."
- HARD RULES — the things I never break (these become the bot's hard limits).
- SETUPS — each setup with EXACT entry conditions (indicator, timeframe, value, direction, # confirmations).
- EXITS & RISK — stop logic, profit logic, sizing logic, daily-stop / "I'm done for the day" logic.
- PSYCHOLOGY — my strengths and my leaks (so the bot can amplify the strengths and the engineer can guard the
  leaks).
- DO / DON'T — two explicit lists of instructions for an automated version of me.
Be specific, concrete, and honest.

Start by confirming in one or two sentences that you understand the goal, then ask me your first 2-4 questions.
```

---

## Part B — the calibration questionnaire (answer directly, or let Perplexity fill it)

> Concrete answers beat philosophy. "I enter when 5m CCI(30) > 160 and price is above the BB200 upper on 1m+5m"
> is something I can encode; "I trade momentum" is not. Each block notes what it `calibrates →` in the bot.

### 1. Goal & risk  · `calibrates → daily target, breach penalty, risk-per-trade sizing, daily-stop`
1. What is a *good* day, a *normal* day, and a *bad* day for you, in %?
2. Max % you'll risk on a single trade? Max % you'll lose in a day before you stop?
3. Is hitting +2.5% the whole job, or do you push for more on good days? When do you *stop* for the day?
4. How many trades is a normal day — a handful, or many?

_Answers:_

### 2. What & when  · `calibrates → which symbols, session/time obs, the "is it tradeable" gate`
1. Which instruments do you trade, and which do you avoid?
2. Which sessions / hours do you trade — and which do you sit out?
3. How do you decide *today* is tradeable vs a day to stand aside?
4. Trending market vs ranging vs news — which do you trade, which do you avoid, and how do you tell them apart?

_Answers:_

### 3. Setups & entries  · `calibrates → the ALPHAS (signals) + the conviction/confirmation reward`
1. Walk me through your **A+ setup** in exact detail — what must be true to enter (indicator, timeframe, value)?
2. How many confirmations do you need, and which ones (CCI levels, BB bands, SMAs, structure…)?
3. Are long and short mirror images, or do you favor one? Any bias?
4. What single thing makes you *cancel* a trade you were about to take?

_Answers:_

### 4. Management & exits  · `calibrates → hard stop, let-winners-run reward, re-entry, two-phase bank`
1. Where exactly does your stop go on entry? Do you move it (to breakeven / trail)? When?
2. Where do you take profit — fixed target, trail, or scale out in pieces?
3. Do you let winners run, or take the money quickly? What reward:risk are you aiming for?
4. If you exit and price keeps going your way, do you get back in? Under what condition?

_Answers:_

### 5. Sizing  · `calibrates → risk-based sizing, conviction-scaled size, correlation/exposure cap`
1. Same size every trade, or bigger on the best setups? If bigger — by how much, and on what basis?
2. Do you size by volatility (wider stop → smaller size)?
3. If you're in 3-4 positions that are really the *same* bet (e.g., all "long the dollar"), how do you handle
   the combined risk?

_Answers:_

### 6. Psychology & discipline  · `calibrates → reward balance (anti-tilt), idle/seek, streak protection`
1. After a loss, do you change anything (size, aggression)? After a *winning streak*, do you press or protect?
2. On a dead day with no setup — can you take a flat/zero day, or do you feel you *must* trade?
3. What's your single biggest leak / the mistake you make over and over?
4. What rules do you **never** break, no matter what?

_Answers:_

### 7. Edge & beliefs  · `calibrates → which signals to trust, regime/edge-decay awareness`
1. *Why* do your setups work — what's actually happening in the market when they fire? Who's on the other side?
2. What market conditions *kill* your strategy?
3. How would you know your edge has stopped working — and what would you do about it?

_Answers:_

### 8. The streak mindset  · `calibrates → escalating streak reward, "protected zero" day, discount horizon`
1. What does "40 winning days in a row" mean to you — and what would you sacrifice to keep a long streak alive?
2. Would you deliberately take a *small* day (e.g., +0.5%) or a *zero* day to avoid risking the streak? When?
3. Is there ever a situation where you'd knowingly break your own rules? Describe it.

_Answers:_

---

## Part C — synthesized profile  *(paste Perplexity's output here, or your own write-up)*

_(to fill)_

---

## Part D — bot-calibration mapping  *(the engineer fills this from Parts B/C)*

| Your rule / tendency | Bot lever it maps to | Setting |
|---|---|---|
| _(e.g., "only trade strong 5m momentum")_ | open-gate (5m CCI) | _AND, ±50_ |
| | | |

> When this table is filled, each row is a concrete change to a reward weight, an alpha, an obs block, or a
> rail — applied, verified bar-for-bar (CPU↔JAX), and trained. That's how we keep the bot calibrated to you.
