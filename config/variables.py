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
FTMO_DAILY_TARGET_PCT: float = 2.5          # +%/day target
FTMO_DAILY_DRAWDOWN_PCT: float = 5.0        # daily loss limit
FTMO_MAX_TOTAL_DRAWDOWN_PCT: float = 10.0   # overall loss limit
FTMO_TRAILING_DRAWDOWN_PCT: float = 4.0     # trailing wall (Quantra 4%)
FTMO_TRAILING_ENABLED: bool = True          # trailing ON/OFF
FTMO_TWO_PHASE_ENABLED: bool = True         # +2.5% -> auto-flat -> fresh trail
FTMO_PHASE2_TRAILING_PCT: float = 1.0       # phase-2 trailing wall

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
