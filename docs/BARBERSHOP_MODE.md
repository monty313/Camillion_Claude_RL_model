# BARBERSHOP MODE

Diagnostics suite (`src/barbershop/`). Phase-0 wires the first three against the live
contract/risk layer; the rest are placeholders filled in Phase 2.

1. **Scoreboard** (`scoreboard.py`, ready) — daily/episode PnL, target progress,
   drawdown, pass/fail.
2. **Day Replay** (`day_replay.py`, done) — per-bar timeline of pressure, equity,
   position, drawdown and risk events.
3. **Trade Autopsy** (`trade_autopsy.py`, done) — why a trade opened, which alphas
   were active, pressure at entry, alignment with consensus, outcome.
4. **Risk Doctor** (`risk_doctor.py`, ready) — drawdown events, near-breaches, loss
   streaks, budget remaining.
5. **Signal Doctor** (`signal_doctor.py`, done) — per-alpha activity, 1/3/10-bar
   accuracy, agreement with consensus, and conflict-rate detection.
6. **Feature Doctor** (`feature_doctor.py`, ready) — observation shape, missing/stale
   features, non-finite values, leakage note.
