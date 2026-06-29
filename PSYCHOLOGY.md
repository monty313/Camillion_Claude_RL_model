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
