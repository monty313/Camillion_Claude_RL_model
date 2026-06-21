# CLAUDE.md — operating rules for any agent editing this repo

> Read this before changing anything. Monty is NOT a programmer; keep it
> simple, commented, and Colab-friendly.

## The three rules that override everything
1. **NEVER silently change the observation shape.** It is locked in
   `config/constants.py` (357 float32, contract `v1.0.0`). If a change would
   alter it, **STOP and explain first**, then bump the contract version and
   update `docs/OBSERVATION_CONTRACT.md` + the shape tests.
2. **NEVER change FTMO numbers** (2.5% daily target, 4% trailing wall,
   two-phase +2.5% -> 1% trailing) without saying so explicitly.
3. **NEVER put TA-Lib / MT5 / pandas inside `env.step()`.** The hot loop only
   reads cached float32 arrays. Speed is the #1 priority.

## Core design (do not drift)
- Each strategy = a signal generator. Output: `+1` buy, `-1` sell, `0` inactive
  (0 is "no setup", NOT a HOLD action). Empty slot = no strategy assigned.
- The RL action space {HOLD, BUY, SELL, CLOSE} is SEPARATE from alpha outputs.
- Observation = raw indicators (190) + alpha values (64) + alpha occupancy
  mask (64) + alpha summary % (4) + last-5 signal memory (5) + signal
  accuracy (2) + account daily (7) + account episode (7) + time (6) +
  portfolio (8) = **357**. Adding strategies fills slots; shape never changes.
- Percentages, not raw counts, so adding strategies does not confuse the bot.

## File header standard (every important file)
WHEN / WHO / WHY / WHERE / HOW / DEPENDS_ON / USED_BY / CHANGE_NOTES(IRAC).
IRAC = Issue, Rule, Application, Conclusion(why it helps pass FTMO).
Every change appends a dated entry to `docs/UPDATE_LOG.md`.

## How to run the tests
- In Colab / with pytest:  `pytest -q`
- Without pytest (stdlib only):  `python tools/run_tests.py`

## Build phases
- Phase 0 (DONE): skeleton, configs, indicator registry/stubs, strategies,
  signals, observation builder, account/risk scaffolds, shape tests.
- Phase 1: wire TA-Lib indicators (5 TFs), example strategies, signal
  accuracy on real data, fast cached training pipeline.
- Phase 2: Jarvis UI, Barbershop mode, one-click Colab notebook.
