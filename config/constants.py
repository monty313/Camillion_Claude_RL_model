# =====================================================================
# WHEN:   2026-06-21 (created, Phase 0)
# WHO:    Claude (Camillion build agent) for Monty
# WHY:    Single source of truth for everything that is FIXED FOREVER:
#         timeframes, indicator specs, strategy-slot count, and the
#         observation-block sizes that define the RL observation contract.
# WHERE:  config/constants.py  -- imported by almost every module.
# HOW:    Pure constants. No logic, no I/O. If a number here changes, the
#         observation shape / model compatibility changes -> that is a
#         DELIBERATE contract version bump (see OBSERVATION_CONTRACT.md).
# DEPENDS_ON: (nothing)
# USED_BY: config/variables.py, src/indicators/*, src/strategies/*,
#          src/signals/*, src/observation/*, tests/*
# CHANGE_NOTES (IRAC):
#   I: A growing strategy count must never change the observation shape,
#      and indicator specs must be defined once so 5 modules agree.
#   R: Spec "Fixed observation shape (never changes when adding strategies)".
#   A: Freeze MAX_STRATEGIES slots + enumerate indicator specs + block sizes.
#   C: Locking shape here lets the policy keep training across strategy
#      additions, which is required to repeatedly pass FTMO challenges.
# =====================================================================
"""Camillion frozen constants (the observation contract lives here)."""
from __future__ import annotations

# --- Timeframes (ORDER IS PART OF THE CONTRACT -- never reorder) ---
TIMEFRAMES: tuple[str, ...] = ("1m", "5m", "30m", "4h", "1d")
N_TIMEFRAMES: int = len(TIMEFRAMES)

# --- Indicator specs (TA-Lib, RAW -- never normalized) ---
# SMA: (period, shift). shift N = value from N bars ago. 6 per timeframe.
SMA_SPECS: tuple[tuple[int, int], ...] = (
    (1, 0), (2, 1), (3, 2), (4, 3), (50, 0), (200, 0),
)
# CCI: each period emits TWO lines -> raw CCI, then SMA(2) shifted 4 bars.
CCI_PERIODS: tuple[int, ...] = (30, 100)
CCI_POST_SMA: int = 2
CCI_POST_SHIFT: int = 4
# RSI: each period emits TWO lines -> raw RSI, then SMA(2) shifted 2 bars.
RSI_PERIODS: tuple[int, ...] = (4, 14)
RSI_POST_SMA: int = 2
RSI_POST_SHIFT: int = 2
# ATR: period 14 -> raw + SMA(2) shifted 4 (volatility; the shift lets the
# bot compare ATR now vs ~4 bars ago, i.e. read its slope).
ATR_PERIODS: tuple[int, ...] = (14,)
ATR_POST_SMA: int = 2
ATR_POST_SHIFT: int = 4
# Bollinger: (period, deviation) -> upper, middle, lower (3 lines each).
BOLLINGER_PERIODS: tuple[int, ...] = (20, 200)
BOLLINGER_DEVS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
BOLLINGER_BANDS: tuple[str, ...] = ("upper", "middle", "lower")

# Per-timeframe indicator line counts (derived -> single source of truth)
N_SMA_PER_TF: int = len(SMA_SPECS)                                   # 6
N_CCI_PER_TF: int = len(CCI_PERIODS) * 2                             # 4
N_RSI_PER_TF: int = len(RSI_PERIODS) * 2                             # 4
N_BB_PER_TF: int = (
    len(BOLLINGER_PERIODS) * len(BOLLINGER_DEVS) * len(BOLLINGER_BANDS)
)                                                                    # 24
N_ATR_PER_TF: int = len(ATR_PERIODS) * 2                              # 2 (raw + shifted)
# --- v1.2.0 alpha-pack extras (appended to each timeframe, in this order) ---
EXTRA_SMA_SPECS: tuple[tuple[int, int], ...] = ((30, 0), (1, 1))      # SMA30 (close); SMA1 shift1 = prev close
SMA_HL_PERIOD: int = 4                                                # SMA(4) shift4 on HIGH and LOW
SMA_HL_SHIFT: int = 4
SMA_HL_BANDS: tuple[str, ...] = ("high", "low")
N_EXTRA_PER_TF: int = len(EXTRA_SMA_SPECS) + len(SMA_HL_BANDS)        # 4
N_INDICATORS_PER_TF: int = (
    N_SMA_PER_TF + N_CCI_PER_TF + N_RSI_PER_TF + N_ATR_PER_TF + N_BB_PER_TF + N_EXTRA_PER_TF
)  # 44
N_INDICATORS_TOTAL: int = N_INDICATORS_PER_TF * N_TIMEFRAMES          # 220

# --- Strategy / alpha slots ---
MAX_STRATEGIES: int = 64
ALPHA_BUY: int = 1
ALPHA_SELL: int = -1
ALPHA_INACTIVE: int = 0   # strategy assigned but no setup right now (NOT a hold)

# --- RL action space (SEPARATE from alpha outputs) ---
ACTION_HOLD: int = 0
ACTION_BUY: int = 1
ACTION_SELL: int = 2
ACTION_CLOSE: int = 3
ACTIONS: tuple[str, ...] = ("HOLD", "BUY", "SELL", "CLOSE")
N_ACTIONS: int = len(ACTIONS)

# --- Signal memory depth (last N bars of net signal balance) ---
SIGNAL_MEMORY_LAGS: int = 5

# =====================================================================
# OBSERVATION CONTRACT (v1.3.0) -- block sizes in concatenation order.
# Total = 461 float32 (v1.3.0 appended a 10-float SIZING block). Adding
# strategies fills alpha slots and does NOT change this number. Changing any
# size here = new contract version (see docs/OBSERVATION_CONTRACT.md).
# =====================================================================
OBS_BLOCK_INDICATORS: int = N_INDICATORS_TOTAL   # 220 raw market inputs (v1.2.0)
OBS_BLOCK_ALPHA_VALUES: int = MAX_STRATEGIES     # 64  (+1 / -1 / 0)
OBS_BLOCK_ALPHA_MASK: int = MAX_STRATEGIES       # 64  occupancy (1 assigned / 0 empty)
OBS_BLOCK_ALPHA_SUMMARY: int = 4                 # buy%, sell%, active%, net%
OBS_BLOCK_SIGNAL_MEMORY: int = SIGNAL_MEMORY_LAGS  # 5
OBS_BLOCK_SIGNAL_ACCURACY: int = 2               # 1-bar, 3-bar
OBS_BLOCK_ACCOUNT_DAILY: int = 7
OBS_BLOCK_ACCOUNT_EPISODE: int = 7
OBS_BLOCK_TIME: int = 6
OBS_BLOCK_PORTFOLIO: int = 8
OBS_BLOCK_ALPHA_STREAK: int = 64   # v1.2.0: per-alpha signal-streak (normalized)
ALPHA_STREAK_CAP: int = 50         # streak fraction = min(streak, cap)/cap
# --- v1.3.0: SIZING / target-awareness block (all fractions of the INITIAL balance) ---
# A what-if lot ladder (each size -> account-% a typical move would be worth) + how much is
# still needed today + drawdown room + the active size. OBSERVATION ONLY for now: sizing is
# NOT yet an action, so the bot learns the size<->risk/reward relationship before it can use it.
SIZING_LOTS_LADDER: tuple[float, ...] = (0.01, 0.1, 0.5, 1.0, 2.0, 4.0)
OBS_BLOCK_SIZING: int = len(SIZING_LOTS_LADDER) + 4  # 6 ladder + target_remaining,dd_room,active_lots,active_move

# Ordered list of (block_name, size). The builder MUST emit in this order.
OBS_BLOCK_ORDER: tuple[tuple[str, int], ...] = (
    ("indicators",       OBS_BLOCK_INDICATORS),
    ("alpha_values",     OBS_BLOCK_ALPHA_VALUES),
    ("alpha_mask",       OBS_BLOCK_ALPHA_MASK),
    ("alpha_summary",    OBS_BLOCK_ALPHA_SUMMARY),
    ("signal_memory",    OBS_BLOCK_SIGNAL_MEMORY),
    ("signal_accuracy",  OBS_BLOCK_SIGNAL_ACCURACY),
    ("account_daily",    OBS_BLOCK_ACCOUNT_DAILY),
    ("account_episode",  OBS_BLOCK_ACCOUNT_EPISODE),
    ("time",             OBS_BLOCK_TIME),
    ("portfolio",        OBS_BLOCK_PORTFOLIO),
    ("alpha_streak",     OBS_BLOCK_ALPHA_STREAK),
    ("sizing",           OBS_BLOCK_SIZING),       # v1.3.0 (appended -> 0..450 indices unchanged)
)
OBS_TOTAL_SIZE: int = sum(size for _, size in OBS_BLOCK_ORDER)  # 461 (v1.3.0)
OBS_SHAPE: tuple[int, ...] = (OBS_TOTAL_SIZE,)
OBS_DTYPE: str = "float32"

OBSERVATION_CONTRACT_VERSION: str = "v1.3.0"
