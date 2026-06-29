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

# --- ALPHA-SHAPING (ON by default — operator 2026-06-27). DELIBERATE, DOCUMENTED departure from the old
#     "reward = equity only / never alpha" design rule (which still holds for the single-symbol TradingEnv).
#     EVERY bonus is CAPPED at the trade's own PnL (it can amplify a real win, never fabricate reward) and
#     only pays when the DAY is net up; the penalty is applied at entry. Set False to restore alpha-free reward. ---
FTMO_ALPHA_REWARD_ENABLED: bool = True       # master switch for the three alpha terms below
FTMO_ALPHA_AGREE_BONUS: float = 0.001       # USE the alphas: profitable close that AGREED with >=50% of firing alphas
FTMO_ALPHA_AGAINST_PENALTY: float = 0.001   # penalty for OPENING a trade against >=50% of firing alphas
FTMO_ALPHA_BEAT_BONUS: float = 0.002        # BEAT the alphas: 2x the others (so beating isn't cancelled by the against-penalty); profit + day-up + capped at PnL

# --- PER-DAY consistency signal: a "won day" = the day ENDS at >= +2.5% of INITIAL (measured at midnight,
#     AFTER any give-back), NOT merely banking it intraday. Reward a won day, penalise a failed day. This
#     also makes giving back a banked +2.5% (phase-2 leash) a real cost. Tune the magnitudes. ---
FTMO_DAY_PASS_REWARD: float = 0.025         # reward when the day ENDS >= +2.5% of initial (a won day)
FTMO_DAY_FAIL_PENALTY: float = 0.025        # penalty when the day ENDS below +2.5% (a failed day)

# --- SEEK-THE-TARGET vs HIDE rebalance (operator 2026-06-28). The breach penalty (1.0) so dominates the
#     tiny per-day rewards that the easiest local optimum is to BARELY TRADE -> a bot that "hides" (never
#     breaches, but never makes +10% either). Two terms break that attractor, in PortfolioEnv only (the
#     single-symbol TradingEnv stays reward=equity-only by design):
#       1. TARGET-SEEK (dense): reward NEW progress toward the +2.5%/day target (high-water-mark, so it can't
#          be farmed by churning). This makes "move toward the day's target" the gradient, so the bot
#          actively seeks profit instead of sitting flat. Total <= seek_weight per won day.
#       2. IDLE-DAY penalty: a day that ends with ZERO trades is penalised -> "hiding" is no longer free, so
#          the bot must EXPLORE toward the target rather than collapse to do-nothing. Small (won't force bad
#          trades; the breach penalty still dominates true recklessness).
#     These are the knobs to tune from the live DASHBOARD DIAGNOSIS (hiding -> raise these; over-trading ->
#     lower seek / raise breach). Set both to 0.0 to restore the pre-rebalance reward. ---
FTMO_TARGET_SEEK_WEIGHT: float = 0.10       # dense reward for new progress toward +2.5%/day (<= this per won day)
FTMO_IDLE_DAY_PENALTY: float = 0.02         # penalty for a day with ZERO trades (anti-"hiding")

# --- DRAWDOWN-PROXIMITY penalty + a SMALLER breach cliff (operator 2026-06-28). The old design only
#     punished a breach at the -1.0 CLIFF, so "do nothing / survive" was the dominant attractor. Now:
#     (a) a GRADUAL penalty grows as equity nears the trailing wall (so approaching the wall costs before
#     you hit it, and the bot plans away from it), and (b) the breach cliff drops 1.0 -> 0.2 so steady
#     +2.5% progress beats merely surviving. The streak reset (a breach restarts the 40-won-days count) is
#     now the real breach deterrent, not a giant per-step penalty. Tune from the dashboard DIAGNOSIS. ---
FTMO_DD_PROXIMITY_COEF: float = 0.02        # per-step penalty = coef * (dd_used_fraction_of_wall)^2 (0 = off)
FTMO_BREACH_PENALTY: float = 0.2            # the breach cliff (was 1.0); the 40-streak reset is the real deterrent
FTMO_PASS_BONUS: float = 1.0               # +10% pass reward + the 4-won-days-in-a-row bonus (kept big)
# --- v1.7.0 trade-risk CLOSE bonuses (PortfolioEnv only, PnL-capped, default 0 = off). Turn on in the
# training path. band_stack: enter ABOVE/BELOW BB200 & BB10 (dev1) on 1m+5m and close in profit, day net up.
# reentry: a with-trend re-entry that pays off. Small so they amplify a real win, never fabricate reward. ---
FTMO_BAND_STACK_BONUS: float = 0.0         # bonus for a band-stacked entry that closes in profit (e.g. 0.005)
FTMO_REENTRY_BONUS: float = 0.0            # nudge for a with-trend re-entry that pays off (e.g. 0.003)

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
