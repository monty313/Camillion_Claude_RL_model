# CLAUDE.md — operating rules for any agent editing this repo

> Read this before changing anything. Monty is NOT a programmer; keep it
> simple, commented, and Colab-friendly.

## The three rules that override everything
1. **NEVER silently change the observation shape.** It is locked in
   `config/constants.py` (367 float32, contract `v1.1.0`). If a change would
   alter it, **STOP and explain first**, then bump the contract version and
   update `docs/OBSERVATION_CONTRACT.md` + the shape tests.
2. **NEVER change FTMO numbers** (2.5% daily target, 4% trailing wall,
   two-phase +2.5% -> 1% trailing) without saying so explicitly.
3. **NEVER put TA-Lib / MT5 / pandas inside `env.step()`.** The hot loop only
   reads cached float32 arrays. Speed is the #1 priority.

## Core design (do not drift)
- Each strategy = a signal generator. Output: `+1` buy, `-1` sell, `0` inactive
  (0 is "no setup", NOT a HOLD action). Empty slot = no strategy assigned.
- **Two kinds of alpha** (both fill slots, both weighted per-slot by the policy):
  *DIRECTIONAL* (default, `+1/-1/0`) vote in the directional consensus
  (`alpha_summary` buy%/sell%/net% + `signal_accuracy`); *NON-DIRECTIONAL gates*
  (`BaseStrategy.DIRECTIONAL = False`, output `1`/`0`, e.g. the movement filters)
  are **excluded** from that consensus via `registry.directional_mask()` — a gate's
  `1` is "condition true", NOT a buy, so it must never be miscounted as a bull vote.
  The policy still sees a gate in its own slot + streak and learns its purpose.
- The RL action space {HOLD, BUY, SELL, CLOSE} is SEPARATE from alpha outputs.
- Observation = raw indicators (200) + alpha values (64) + alpha occupancy
  mask (64) + alpha summary % (4) + last-5 signal memory (5) + signal
  accuracy (2) + account daily (7) + account episode (7) + time (6) +
  portfolio (8) = **367**. Adding strategies fills slots; shape never changes.
- Percentages, not raw counts, so adding strategies does not confuse the bot.

## Scaling alphas toward ~1000 (do not drift from this)
The plan is to grow to ~1000 alphas WITHOUT ever destabilising the observation.
Keep **per-slot** (one slot per alpha = the policy learns a weight per alpha).
Empty slots do NOT hurt learning (the net ignores constant-0 inputs); their
only cost is **memory**. Beat that memory with **int8 alpha/streak tables +
one shared precomputed table across envs** — NOT by switching to aggregate /
consensus features (that throws away per-alpha weighting). Raising
`MAX_STRATEGIES` is a deliberate **contract bump** (resizes 3 obs blocks):
follow the obs-contract protocol; set it once with headroom. Full logic at the
`MAX_STRATEGIES` comment in `config/constants.py` and in
`docs/ENVIRONMENT_STATE.md`.

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

## Alpha-state 0 vs action HOLD (do not conflate)
`alpha = 0` means *an assigned alpha has no setup right now* (alpha-space).
`ACTION_HOLD` means *the policy chose to take no trade action this step*
(action-space). They share the integer 0 but live in DIFFERENT spaces — keep
them distinct by name and context in code, docs, and diagnostics. Empty alpha
slot (mask 0) is a third, separate thing (no alpha assigned).
