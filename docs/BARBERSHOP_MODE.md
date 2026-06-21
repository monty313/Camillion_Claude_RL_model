# BARBERSHOP MODE

Diagnostics suite (`src/barbershop/`). Phase-0 wires the first three against the live
contract/risk layer; the rest are placeholders filled in Phase 2.

1. **Scoreboard** (`scoreboard.py`, ready) — daily/episode PnL, target progress,
   drawdown, pass/fail.
2. **Day Replay** (`day_replay.py`, Phase 2) — replay a day with signal pressure +
   trades + risk events.
3. **Trade Autopsy** (`trade_autopsy.py`, Phase 2) — why a trade opened, which alphas
   were active, pressure at entry, outcome.
4. **Risk Doctor** (`risk_doctor.py`, ready) — drawdown events, near-breaches, loss
   streaks, budget remaining.
5. **Signal Doctor** (`signal_doctor.py`, Phase 2) — per-strategy stats, accuracy,
   conflict detection.
6. **Feature Doctor** (`feature_doctor.py`, ready) — observation shape, missing/stale
   features, non-finite values, leakage note.
