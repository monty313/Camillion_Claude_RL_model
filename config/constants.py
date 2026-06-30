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
# =====================================================================
# HOW WE SCALE ALPHAS IN THE FUTURE  (read this before raising MAX_STRATEGIES)
# Goal: grow toward ~1000 alphas WITHOUT ever destabilising the observation.
# The logic, settled with the operator, that every future agent must follow:
#
#  1. PER-SLOT IS INTENTIONAL, KEEP IT. Each alpha owns one fixed slot ->
#     3 obs inputs (value, mask, streak). This is what lets the POLICY learn
#     an individual weight per alpha. Do NOT replace per-slot with aggregate /
#     consensus features (that throws away the per-alpha weighting the
#     operator wants) unless the operator explicitly decides to.
#
#  2. FILLING A SLOT NEVER CHANGES THE OBS SHAPE. Assigning an alpha flips its
#     slot's VALUE (0 -> +/-1); the NUMBER of inputs is unchanged. So adding
#     alphas up to MAX_STRATEGIES is free and shape-stable. The trained policy
#     keeps working. (Rule #1 in CLAUDE.md: shape is sacred.)
#
#  3. EMPTY SLOTS DO NOT HURT LEARNING. A slot that is always 0 is invisible to
#     the network (its weight gets no gradient and never trains). 900 empty
#     slots do NOT make the bot dumber. Their ONLY cost is MEMORY (see #4).
#
#  4. THE REAL COST AT SCALE IS MEMORY, AND IT IS SOLVED, NOT AVOIDED. The env
#     precomputes a (bars x slots) alpha table (and a streak table) ONCE, then
#     the hot loop just indexes a row -- per-step CPU barely grows with slots,
#     and the per-alpha weighting lives in the NETWORK (GPU work, scales fine).
#     What grows is the table's RAM, especially when copied per parallel env.
#     To afford many slots: (a) store alpha/streak tables as int8 (values are
#     -1/0/+1 -> 1 byte, a 4x cut), (b) compute the table ONCE and SHARE it
#     read-only across envs (not one full copy per worker), (c) only grow the
#     filled count over time -- empties cost only the (now-cheap) memory.
#
#  5. RAISING THIS NUMBER IS A DELIBERATE CONTRACT BUMP. It resizes 3 obs
#     blocks -> the obs total changes -> follow the protocol: bump
#     OBSERVATION_CONTRACT_VERSION, update docs/OBSERVATION_CONTRACT.md +
#     docs/ENVIRONMENT_STATE.md + the shape tests, and the env_fingerprint
#     rolls automatically. Set it ONCE to the target with headroom rather than
#     resizing repeatedly. Cheap to do while pre-training; costly once a policy
#     you care about is trained.
#
#  TL;DR: keep per-slot, never reshape the obs casually, beat the memory with
#  int8 + a shared precomputed table -- not by switching to aggregates.
# =====================================================================
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
# OBSERVATION CONTRACT (v1.6.0) -- block sizes in concatenation order.
# Total = 499 float32 (v1.6.0 appended a 20-float RAW OHLC block at the END).
# Adding strategies fills alpha slots and does NOT change this number. Changing any
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
# --- v1.4.0: CROSS-ASSET perception so ONE policy can compare opportunity/risk across the FULL
# FTMO broker (forex/pairs, indices, metals, energies, crypto -- 130+ instruments) in COMMON
# units. Asset-class one-hot + ATR-normalized movement (move-in-ATRs, ATR/price, ATR-regime) +
# sessions (Asian, London-NY overlap). The ATR-normalized features are SCALE-FREE, so they read
# the same on a 1.1 pair, a 40000 index or a 2000 metal -> one policy generalizes to any symbol. ---
ASSET_CLASSES: tuple[str, ...] = ("pair", "index", "metal", "energy", "crypto")  # one-hot ORDER (contract)
OBS_BLOCK_CROSS_ASSET: int = len(ASSET_CLASSES) + 5  # classes + (move/ATR, ATR/price, ATR-regime) + (asian, overlap)
# --- v1.5.0: RECENT-CONTEXT block (8 floats). Recent DAILY movement (prior days + last-week avg),
# expressed RELATIVE to the symbol's own average so it is comparable across the universe, plus a
# TIME-aware "am I on pace to pass" read (days elapsed, return so far, pace vs +2.5%/day, remaining). ---
OBS_BLOCK_RECENT_CONTEXT: int = 8
# --- v1.6.0: RAW OHLC block. Open/High/Low/Close of the last CLOSED bar on each of the 5 timeframes
# (TIMEFRAMES order; fields O,H,L,C). The policy finally SEES High/Low/Open, not just close. RAW
# (never normalized), exactly like the indicator block. Built at cache time from the resampled bars
# (the env does not carry high/low) and threaded in via src/data/aux_features.py. APPENDED AT THE END
# so obs indices 0..478 are UNCHANGED; only new indices 479..498 are added. DELIBERATE contract bump
# (operator 2026-06-28): a trained v1.5.0 policy must retrain because the shape changed. ---
OBS_BLOCK_OHLC: int = N_TIMEFRAMES * 4           # 20  (O,H,L,C per timeframe)
# --- v1.7.0: TRADE-RISK block (14 floats). The live RISK state of the CURRENT symbol's open trade so the
# policy can MANAGE it (when to hold/close) and learn to RE-ENTER a winner: in-trade flag + direction,
# unrealized P&L in ATR units and as % of the pot, how close price is to the 2x-ATR(14) SOFT stop and to
# the 1m Bollinger(10,1) opposite-band HARD stop (0->1), bars held, max favorable/adverse excursion (ATR),
# re-entry context (bars since this symbol's last close, that trade's direction, price-vs-last-exit in ATR),
# and whether price is stacked ABOVE / BELOW BB200(dev1) AND BB10(dev1) on BOTH 1m and 5m (the band-stack
# enter-bonus condition). All bounded ~[-1,1] / [0,1]. APPENDED AT THE END -> obs indices 0..498 UNCHANGED;
# only new indices 499..512 are added. DELIBERATE contract bump (operator 2026-06-29): a v1.6.0 policy must
# retrain because the shape changed. The block is DYNAMIC (recomputed each step from the position state);
# the 1m+5m BB(10,1) bands it reads are precomputed in TradingEnv._precompute (no cache-format change). ---
OBS_BLOCK_TRADE_RISK: int = 14
# --- v1.8.0: CONSISTENCY block (4 floats). The bot's MULTI-DAY FTMO standing so it can VALUE and PROTECT the
# won-day STREAK (the operator's 40-in-a-row goal), paired with the stretched discount horizon: current
# won-day streak (toward 40), cumulative won days, won-day RATE (days won / days elapsed = consistency), and
# how deep into the 40-day journey it is. Shared-pot env fills these from the pot's day-scoring; the
# single-symbol env (no streak logic) emits zeros. APPENDED AT THE END -> obs indices 0..512 UNCHANGED; only
# new indices 513..516 are added. DELIBERATE contract bump (operator 2026-06-29): a v1.7.0 policy must retrain
# because the shape changed. DYNAMIC block (recomputed each step from the episode's day-scoring state). ---
OBS_BLOCK_CONSISTENCY: int = 4
# --- v1.9.0: MOMENTUM-PERCEPTION block (9 floats). Teach the PRINCIPLE of momentum, not hard-coded CCI rules
# (JORDAN_PRINCIPLES.md). The operator decomposed momentum into learnable sub-problems; each is a per-bar SCORE
# the policy consumes and LEARNS to act on: tradeability, higher-TF bias, multi-TF alignment, strength (graded
# CCI ladder), exhaustion, entry location (extension vs pullback), structure (position in recent range),
# persistence (follow-through), decay (momentum dying). STATIC (market-only, per-bar) -> placed in the static
# obs tensor, lifted byte-identical into the JAX env (auto parity). APPENDED -> obs indices 0..516 UNCHANGED;
# new indices 517..525. DELIBERATE contract bump (operator 2026-06-30): a v1.8.0 policy must retrain. ---
OBS_BLOCK_MOMENTUM: int = 9
# --- v1.10.0: SHIFTED-SMA HUGGING-PRESSURE block (15 floats). The operator's "Shifted SMA Hugging Pressure"
# agent (heavy). Across 5m / 15m / 1h, a fast SMA(2) of High & Low shifted forward 1 bar forms an envelope;
# price that keeps HUGGING one side (never touching the opposite band) for consecutive bars = sustained
# directional pressure; 2+ timeframes agreeing = strong continuation. Per-TF [side, hug_count, respecting] x3
# (9) + aggregate [agree_bull, agree_bear, net_pressure, strength, continuation_2plus, dominant_side] (6). 15m
# & 1h are a RESAMPLED side-channel from the 1m High/Low (NOT new full obs timeframes). STATIC (market-only) ->
# placed in the static obs tensor, lifted byte-identical into the JAX env (auto parity). APPENDED -> obs
# indices 0..525 UNCHANGED; new indices 526..540. DELIBERATE bump (operator 2026-06-30): a v1.9.0 policy
# retrains. The HEAVY action prior + indices/metals miss-penalty live in the reward (portfolio_env), not here. ---
OBS_BLOCK_HUG_PRESSURE: int = 15

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
    ("cross_asset",      OBS_BLOCK_CROSS_ASSET),  # v1.4.0 (appended -> 0..460 indices unchanged)
    ("recent_context",   OBS_BLOCK_RECENT_CONTEXT),  # v1.5.0 (appended -> 0..470 indices unchanged)
    ("ohlc",             OBS_BLOCK_OHLC),         # v1.6.0 (appended -> 0..478 indices unchanged)
    ("trade_risk",       OBS_BLOCK_TRADE_RISK),   # v1.7.0 (appended -> 0..498 indices unchanged)
    ("consistency",      OBS_BLOCK_CONSISTENCY),  # v1.8.0 (appended -> 0..512 indices unchanged)
    ("momentum",         OBS_BLOCK_MOMENTUM),     # v1.9.0 (appended -> 0..516 indices unchanged)
    ("hug_pressure",     OBS_BLOCK_HUG_PRESSURE), # v1.10.0 (appended -> 0..525 indices unchanged)
)
OBS_TOTAL_SIZE: int = sum(size for _, size in OBS_BLOCK_ORDER)  # 541 (v1.10.0)
OBS_SHAPE: tuple[int, ...] = (OBS_TOTAL_SIZE,)
OBS_DTYPE: str = "float32"

OBSERVATION_CONTRACT_VERSION: str = "v1.10.0"
