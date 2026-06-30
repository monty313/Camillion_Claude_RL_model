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

## Part C — synthesized profile  *(round 1, via Perplexity — Setup 2 was truncated; momentum logic pending Round 2)*

**Jordan — high-frequency intraday momentum scalper.** Targets ~+2.5–3.5% good days; hard −4% daily *equity*
drawdown stop (open P&L counts). 0.1–0.2% risk/trade off a stop; 30+ trades/day. Trades only thin-spread,
high-liquidity windows (London, NY, the London–NY overlap, the 9:30 NY cash open). Instrument-agnostic — any
symbol that passes spread + momentum filters.

**Decision process (per session):**
1. **Risk gate** — equity vs prior day + open P&L; if day drawdown ≤ −4% → stop (no new entries). Set
   per-trade risk to 0.1–0.2% of equity (size = risk$ ÷ stop distance).
2. **Time & spread gate** — only thin-spread, high-liquidity sessions; reject symbols whose spread exceeds the
   "thin" threshold.
3. **Symbol scan** — broad universe (FX, indices, maybe crypto CFDs); keep those with tight spreads + clear
   directional momentum on ≥2 timeframes.
4. **Trend filter** — **15m 200 SMA**: price above → longs only; below → shorts only. No counter-trend.
5. **1h structure** — mark nearest 1h swing high/low; longs focus on breakouts above the 1h high, shorts below
   the 1h low.
6. **Momentum alignment** — baseline ≥2 aligned timeframes (15m + 5m, same direction as the 15m SMA filter);
   **after a loss, upgrade to 3** (1h + 15m + 5m). Low-momentum days → shift to lower TFs (5m/30m) + low-spread
   indices, still requiring clear momentum.
7. **A+ setup (NY-open 1h breakout)** — near 9:30 NY: a **5m candle that fully closes beyond the nearest 1h
   level** in the allowed direction, with the 15m 200 SMA confirming. Primary trigger.
8. **Entry refinement — 5m BB(10,1):** uptrend (longs) → buy small pullbacks toward/through the **lower** band;
   downtrend (shorts) → sell pullbacks toward/through the **upper** band; entries must coincide with continued
   multi-TF momentum.
9. **Execution** — stop sized so risk = 0.1–0.2% equity; market/limit per the pullback rule; expect 30+/day.
10. **Management** — keep adding in the trend direction while the 15m SMA bias holds, momentum stays aligned,
    and day drawdown is above −4%; take partials as price reaches the opposite Bollinger envelope / next 1h
    structure; move stop to breakeven after ~1R on fast moves.
11. **Daily shutdown** — at −4% day drawdown, stop new trades, manage/close only. Does **not** cap upside on
    good days — keeps trading momentum while setups exist.

**Hard rules:** 0.1–0.2% risk/trade (equity-based); −4% daily hard stop; 15m-200-SMA direction only (no
counter-trend); ≥2 aligned TFs (3 after a loss); all primary setups reference the nearest 1h high/low;
thin-spread sessions only; **after a loss, do NOT add size — tighten criteria (more TFs)**; after a win, risk
stays fixed (no "house money").

**Setups:** (1) **NY-open 1h breakout trend scalper** — as above (5m close beyond 1h level + 15m SMA bias + 5m
BB(10,1) pullback). (2) **General multi-TF momentum scalps** — London/NY, tight spreads, 15m trending vs 200
SMA, 1h+5m(+30m) momentum aligned with the 15m bias; **[Round-1 text truncated at the 5m/30m BB(10,1) entry —
to be completed in Round 2].**

**Exits/risk:** stop below recent 5m swing low / slightly beyond the breached 1h level (more conservative of the
two), sized to 0.1–0.2%; partials toward the opposite BB envelope / next 1h structure; breakeven after ~1R.

**Psychology:** disciplined, fixed-risk, momentum-only; after losses he *tightens* (more confirmation), never
sizes up; comfortable doing nothing until momentum is clearly there.

> **Open question the profile does NOT yet answer (the important one): what *exactly* is "momentum/thrust" —
> which indicator(s), which periods, what values, per timeframe? → Round 2 below.**

---

## Round 2 — follow-up questions (MOMENTUM is the priority)

> Answer in exact indicators + values per timeframe wherever you can — that's what becomes the alphas.

### A. Momentum — the core (please be precise)
1. When you read "momentum / thrust," what are you literally looking at — an **oscillator** (CCI? RSI? MACD?),
   the **slope** of the 200 SMA, **candle size/body**, **structure** (higher highs/higher lows), or a combo?
2. We know on the **5m** you use **CCI** (you block when 5m CCI(30) & CCI(100) are both inside ±50). Do you read
   CCI the *same way* on 15m / 30m / 1h? Same periods (30 & 100)? Or a different tool on the higher TFs?
3. The **CCI ladder** — map it: |CCI| < 50 = ? (you said: no trade); 50–100 = ?; 100–160 = ?; > 160 = ?. Which
   band is the **minimum to ENTER**, and does it differ by timeframe?
4. Write the **"2 timeframes aligned" long condition as literal ANDs**, e.g.: "price > 15m 200 SMA **AND** 15m
   CCI(30) > ___ **AND** 5m CCI(30) > ___ **AND** 5m close above the 1h high." Give the real numbers.
5. Does the **breakout candle's SIZE** matter — is a 5m close *barely* beyond the 1h level as good as a big
   thrust candle, or do you need a minimum range/body?
6. How do you know momentum is **DYING** (stop adding / exit)? CCI rolling back under a level (which?), price
   closing back inside the 5m band, a timeframe flipping, candles shrinking?
7. After a loss → **"3 timeframes."** Which third TF (1h?), and how long does the stricter rule stay on — the
   next trade only, until your next win, or the rest of the day?

### B. Exits & management
8. **Finish Setup 2** (it cut off): the exact entry trigger for a general momentum scalp with no 1h breakout.
9. **Profit-taking, exactly:** how much do you take off and where (opposite 5m band? a fixed R like 1R/2R? next
   1h level?), and what makes you FULLY close?
10. **Stop:** below the 5m swing low *or* the 1h level — which, and how many bars define that swing low?

### C. Risk / portfolio / day
11. **Good day:** you don't cap upside — but the bot banks +2.5% and protects the day to win 40 in a row. Do
    you keep pressing past +2.5%, or ease off / protect once solidly green? (Decides the +2.5% auto-bank.)
12. **Correlation:** with 30+ trades on a shared account, do you cap how many positions at once, or how many in
    the same direction / correlated instruments (e.g., long four dollar-pairs at the same time)?
13. **News:** do you avoid entering right around high-impact releases (NFP, FOMC, CPI)?

### D. Timeframes (a calibration decision)
14. Your edge leans on the **15-minute** (200 SMA) and **1-hour** (structure), but the bot only carries 1m / 5m
    / 30m / 4h / 1d. How essential are the *exact* 15m and 1h? Can 5m + 30m stand in — or do we **add 15m and
    1h** as real timeframes (a bigger build, worth it if they're core to you)?

---

## Part D — bot-calibration mapping  *(the engineer fills this from Parts B/C)*

| Your rule / tendency | Bot lever it maps to | Setting |
|---|---|---|
| _(e.g., "only trade strong 5m momentum")_ | open-gate (5m CCI) | _AND, ±50_ |
| | | |

> When this table is filled, each row is a concrete change to a reward weight, an alpha, an obs block, or a
> rail — applied, verified bar-for-bar (CPU↔JAX), and trained. That's how we keep the bot calibrated to you.
