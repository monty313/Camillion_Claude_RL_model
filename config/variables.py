# =====================================================================
# WHEN:   2026-06-21 (created Phase 0; updated same day for runtime risk knobs)
# WHO:    Claude (Camillion build agent) for Monty
# WHY:    User-tunable knobs. Changing these is SAFE and needs NO RETRAIN:
#         the observation exposes target/drawdown ONLY as percentages
#         (progress-to-target, % of DD budget used/remaining), so the same
#         trained policy reads them correctly even when the absolute numbers
#         change. That is why these live here and NOT in constants.py.
# WHERE:  config/variables.py
# HOW:    Plain module-level values; risk knobs grouped per mode (FTMO/FREE).
# DEPENDS_ON: numpy
# USED_BY: config/ftmo_config.py, src/risk/*, src/account/win_loss_features.py,
#          src/observation/builder.py, src/jarvis/* (live edits)
# CHANGE_NOTES (IRAC):
#   I: Monty must change target / trailing-DD / trailing on-off in BOTH modes
#      WITHOUT retraining.
#   R: Spec percentage features + Monty note "change target & trailing dd,
#      trailing on/off, ftmo or free, without having to retrain".
#   A: Expose target/trailing/toggle per mode here; features stay fractional.
#   C: Operator can retune risk live and reuse the same model -> faster
#      iteration toward a consistent FTMO pass rate.
# =====================================================================
"""User-tunable variables (safe to edit at runtime; never changes obs shape)."""
from __future__ import annotations
import numpy as np

# --- Mode: "FTMO" (prop-firm defaults) or "FREE" (your own rules) ---
MODE: str = "FTMO"

# --- Account ---
STARTING_BALANCE: float = 100_000.0

# =====================================================================
# RISK SETTINGS -- editable at RUNTIME, NO RETRAIN NEEDED (see header WHY).
# Use config.ftmo_config.update_risk_settings(...) to change these live.
# =====================================================================

# --- FTMO mode (locked-style defaults you can still tweak) ---
FTMO_DAILY_TARGET_PCT: float = 2.5          # +%/day target = 2.5% of the INITIAL balance
FTMO_DAILY_DRAWDOWN_PCT: float = 5.0        # daily loss limit (FTMO hard line, from day start)
FTMO_MAX_TOTAL_DRAWDOWN_PCT: float = 10.0   # overall loss limit (FTMO hard line)
FTMO_TRAILING_DRAWDOWN_PCT: float = 4.0     # PHASE-1 trailing wall (self-imposed, from peak; trips before FTMO's 5/10%)
FTMO_TRAILING_ENABLED: bool = True          # ON: phase-1 risk wall is a 4% trailing drawdown
FTMO_TWO_PHASE_ENABLED: bool = True         # ON: hit +2.5%/day -> close ALL & bank it (daily engine -> +10% over ~4 days)
FTMO_PHASE2_TRAILING_PCT: float = 1.0       # PHASE-2 trailing wall after banking (tight, protects the day's gain)
FTMO_PHASE2_CONTINUE: bool = True           # after banking +2.5%: keep trading under a tight 1% trail (operator 2026-06-27); False=stop
FTMO_PROFIT_TARGET_PCT: float = 10.0        # FTMO Challenge PASS target (episode +10%)

# --- NY-session reward bonuses (DELIBERATE reward shaping for the ORB index strategy, operator
# decision). The bot earns a bonus for BANKING (closing in profit) during the most-liquid New York
# session on INDEX instruments. NY open = 13:30 UTC. A bonus QUALIFIES when, on indices, the
# session's CLOSED-IN-PROFIT P&L reaches: HALF >=50% of the daily target within 2h (13:30-15:30);
# FULL >=100% within 3h (13:30-16:30); each once/day, index share of closed P&L >=50%. It is only
# PAID at day-end IF THE DAY PASSES (day closed >= +2.5% of initial). If the day fails or breaches,
# the bonus is erased (never paid). Set both to 0.0 to disable. ---
FTMO_NY_HALF_TARGET_BONUS: float = 0.15
FTMO_NY_FULL_TARGET_BONUS: float = 0.45     # 3x the half bonus

# --- Transaction cost (per SIDE, as a fraction of notional; ~0.000035 ~= 0.8 pip round-trip on EURUSD) ---
TRANSACTION_COST_FRAC_PER_SIDE: float = 0.000035

# --- 5m CCI open-gate threshold (only used when open_gate=True) ---
# A NEW position may open only when BOTH 5m CCI(30) and CCI(100) are beyond +/-this
# level (i.e. |cci| > threshold). Bigger = stricter (trades only on stronger moves).
# 50 = original behaviour; 100 = "both CCIs past +/-100" momentum filter.
OPEN_GATE_CCI_THRESHOLD: float = 50.0

# --- FREE mode (your own rules) ---
DAILY_TARGET_PCT: float = 2.5
MAX_DAILY_DRAWDOWN_PCT: float = 5.0
MAX_TOTAL_DRAWDOWN_PCT: float = 10.0
TRAILING_DRAWDOWN_PCT: float = 4.0          # trailing wall amount (FREE)
TRAILING_DRAWDOWN_ENABLED: bool = True      # trailing ON/OFF (FREE)

# --- Universe (Phase 0 default; wired to MT5/cache in Phase 1) ---
SYMBOLS: tuple[str, ...] = ("EURUSD", "XAUUSD", "GBPUSD", "US30")
PRIMARY_SYMBOL: str = "EURUSD"

# --- Signal accuracy rolling window (bars) ---
SIGNAL_ACCURACY_WINDOW: int = 100

# --- "max reasonable" denominators for percentage features ---
MAX_DAILY_TRADES: int = 30          # daily_trades_count_pct = trades_today / this
MAX_CONSECUTIVE_LOSSES: int = 10    # consecutive-loss % denominator
MAX_OPEN_POSITIONS: int = 5         # portfolio open-positions % denominator

# --- float32 everywhere (training-speed rule) ---
FLOAT = np.float32
