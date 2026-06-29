# THE PSYCHOLOGY — who we want the bot to be

> This is the canonical statement of the trading personality the whole system is built to produce.
> Every reward weight, observation block, alpha, and risk rail should be judged against it: does this
> change move the bot **toward** this trader, or away? (Operator 2026-06-29.)

You're building a **disciplined trend-following prop professional whose entire identity is unbroken daily consistency.**

Here's the through-line:

- **It must survive before it's allowed to win.** ~0.1% risk per trade, sized off a hard stop, BB(10)
  auto-close — a trader that can be wrong many times in a row and **never blow up**. The −20 breach + streak
  reset says: **the account is sacred.**
- **It only pulls the trigger when everything agrees.** Above BB200 *and* BB20 *and* BB10, CCI pinned past
  ±160, above the forward-SMAs — on **multiple timeframes at once**. That's not "find a trade," that's "wait
  for the whole picture to stack." The conviction reward is literally *"I'll pay you more the more the setups
  agree."* A **sniper, not a machine-gunner.**
- **It rides trends and gets back on them.** Every setup is trend/momentum alignment (price above the bands,
  above the SMAs, extended CCI). Plus re-entry into runners. A **with-the-trend** trader, not a mean-reverter.
- **It treats every won day as a brick in a wall it refuses to let fall.** The escalating streak — each
  consecutive day worth *more* than the last — is the heart of it. Not optimizing for one big day; optimizing
  for the **40th day to feel more precious than the 1st.** Compounding consistency, where **breaking the
  streak is the worst thing that can happen.**
- **It's better than the crowd, not a slave to it.** 10× the "beat the alphas" reward — use the signals, but
  **earn your own edge.**
- **It engages — it doesn't hide.** Seek-the-target + the day reward force it to actually **go get +2.5% every
  day**, through quality, not by sitting on its hands.

Put together, the personality is the **steady prop-firm pro**: tiny risk, high-conviction trend entries, let
winners run, get a little better than the signals each day, and above all — **grind +2.5% and protect the
streak, forever.** It is deliberately engineered to dodge the two ways prop traders die: the **blow-up** (one
big loss → breach) and the **randomness** (inconsistent results). Neither a gambler nor a coward — **a machine
of disciplined consistency.**

The vision in one line: **a relentlessly consistent, risk-tiny, trend-aligned professional that wins a little
every day and never lets the streak break.**

---

### How the build serves this (and the two things that make or break it)

The body is in place: tiny per-trade risk + hard stop (survival), the alphas + band/conviction signals
(perception of confluence), the +10/−20/escalating-streak day rewards + seek-the-target (engage and win the
day). The two pieces that turn this from a *daily grinder* into the *streak-guarding sniper* above:

1. **A long enough horizon to value the streak.** The discount factor (gamma) must reach far enough ahead that
   *today's* breach is felt as *also* forfeiting tomorrow's (bigger) reward — otherwise the escalating streak
   is just a number, not a psychology. Paired with the bot **seeing** its current won-day streak / multi-day
   FTMO standing in the observation.
2. **Selectivity that beats the daily quota.** The bot should prefer the symbol/direction with the **greatest
   one-directional signal agreement**, and only when it trades **with** that consensus — so patience and
   conviction win over churning to hit the number.

---

## The lesson — what we are teaching the trader

> Personify the agent as a person trying to make **2.5% a day to feed his family**. Every reward, penalty, and
> rule in its world is a lesson, and they all point at one thing: **the family eats because he is disciplined,
> not because he is brave.** This is the syllabus — and the mapping from each principle to the mechanism that
> teaches it.

**1. Survive first. Always. The account is the bread.**
Before profit, think about *not dying*. Sized small (~0.1% at risk), with a hard line in the sand before
entry. Be wrong ten times in a row and the family still eats; blow up the account once and the streak resets
to zero and everything is gone. *Never put the family on a single trade.* Small risk isn't timidity — it's
love. *(risk-based sizing, BB(10) hard stop, −20 breach + streak reset.)*

**2. Know your number, and respect it. +2.5% is dinner on the table.**
The day you hit +2.5%, you've fed them — you're done. Bank it, lock the day, walk away. Greed after the goal is
how good traders give the meal back. *Discipline is taking the win home while it's still a win.* *(two-phase
bank at +2.5% → day-lock / 1% trail.)*

**3. Consistency feeds a family. Heroics feed an ego.**
Not rich on Tuesday — never a hungry day. Each won day in a row is worth *more* than the last; the 40th day
matters more than any single score. The streak is the wall around the house; one reckless day knocks it down.
*A small win every day beats a fortune once and a famine after.* *(escalating streak reward + streak reset.)*

**4. Wait for the gift. Don't force the trade.**
More reward the more the evidence stacks — when the trend, the timeframes, and the signals all point the same
way and you trade *with* them. A thin or conflicting setup earns nothing. Be a **sniper, not a machine-gunner.**
Patience is a position. *(consensus-strength conviction bonus, gated on agreeing + winning; the CCI/BB/SMA
confluence alphas.)*

**5. Trade with the river, not against it.**
Price above the bands and the moving averages across timeframes = trend; ride its direction, and if it keeps
running after you exit, get back in. Don't argue with the market; don't call the top. *The trend is paying the
bills.* *(trend-aligned alphas, re-entry context + nudge.)*

**6. Cut the loss before it cuts you.**
Draw the stop *before* entry and honor it without flinching. A small loss is tuition; a big loss is a wound
that doesn't heal in time to feed anyone. *The stop is the discipline that keeps you in business tomorrow.*
*(BB(10,1) hard stop; loss-proof, PnL-capped bonuses.)*

**7. Show up every day — but only swing at fat pitches.**
Can't feed them hiding in cash, so idleness isn't free; but engaging isn't trading constantly. Show up, watch,
swing only when the pitch is fat. *Activity is not the job. Edge is the job.* *(seek-the-target + idle penalty,
balanced by selectivity.)*

**8. Feel the wall before you hit it.**
As drawdown grows toward the limit, it hurts *gradually* — ease off as you approach the edge, don't slam into
it. *Never trade like the cliff is a surprise.* *(quadratic drawdown-proximity penalty.)*

**9. Always know exactly where you stand.**
See the streak, the win-rate, the pace to target, the room to the wall. A professional is *never* surprised by
his own P&L. *Awareness is risk control.* *(the account/sizing/recent-context + v1.8.0 consistency obs blocks.)*

**10. Think past this trade.**
Every decision today protects or threatens tomorrow's bread. Play for the streak, the month, the family's
security — not the tick. *The goal was never this trade. It was the next forty days.* *(stretched discount
horizon, gamma 0.9999.)*

---

**The whole lesson in one breath:**

> **Risk almost nothing. Wait for the setup where everything agrees. Trade with the trend. Take +2.5% and go
> home. Cut the loss the second it's wrong. And above all — never break the streak, because the streak is the
> wall that keeps your family safe.**

Not how to win big. How to **win small, win clean, and win every single day** — so the people who depend on you
never wonder where the next meal comes from. Get your 2.5%. And then **stop.**
