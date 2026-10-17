# Camillion_Claude_RL_model — IRAC handoff (2026-10-17)

## Issue
RL forex agent combining 64 alpha slots + raw indicators to repeatedly pass FTMO-style challenges (2.5%/day, 4% trailing).
Central question: Goal not met: no evidence of a trained policy hitting the target; reward still being tuned; PR #12 open.

## Rule
README.md (locked 367-dim obs contract, later v1.6.0); docs/OBSERVATION_CONTRACT.md; FTMO 2.5% target / 4% trailing wall; Doctrine: scalping-only; 5m CCI 30/100 burst entry; 1R=0.25% equity; max 4 concurrent; daily floor -3R; pyramid winners OK, add-to-losers banned.

## Application
73 commits, last 1586a72 2026-06-28 (merged PR #13: JARVIS bridge, PortfolioEnv, CPU+TPU trainer, ADX-DI alphas). 3b1f7b1 jax_tpu/ trainer 'train to 40 winning days'. 0a1313e/9750cf9 reward tweaks same day. README still says 'Phase 0 skeleton' (stale). Tests in tests/ (not run). 5 TODO/FIXME. HANDOFF.md = JARVIS bridge wiring. TRAINING_TUNING_TODO.md present. Risk is FTMO-%, not 0.25% R; no CCI-burst entry rule verified.

## Conclusion
Status: in progress. No pushes since 2026-06-28. Blockers: no validated training result; reward instability (repeated retunes); risk model diverges from doctrine.
Next: 1) Run tests (python tools/run_tests.py). 2) Decide PR #12 merge/close. 3) Run jax_tpu trainer, record OOS pass rate.
