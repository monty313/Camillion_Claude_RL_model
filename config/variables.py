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
# Operator 2026-06-29 reward REBALANCE (see the block comment near FTMO_DAY_PASS_REWARD): 10x the alpha terms so
# the bot develops real edge (BEAT) instead of merely shadowing the consensus. Still PnL-capped + day-up gated.
FTMO_ALPHA_AGREE_BONUS: float = 0.01        # USE the alphas: profitable close that AGREED with >=50% of firing alphas
FTMO_ALPHA_AGAINST_PENALTY: float = 0.01    # penalty for OPENING a trade against >=50% of firing alphas
FTMO_ALPHA_BEAT_BONUS: float = 0.05         # BEAT the alphas (emphasised): out-earn the consensus; profit + day-up + capped at PnL

# --- PER-DAY consistency signal (operator 2026-06-29 REBALANCE). A "won day" = the day ENDS at >= +2.5% of
#     INITIAL (measured at midnight, AFTER any give-back). The reward is now DOMINATED by the day outcome so the
#     bot optimises for CONSISTENCY (hit +2.5% and lock), not raw P&L:
#       - WON DAY (no breach):  +DAY_PASS_REWARD (10)  + an ESCALATING streak bonus (below).
#       - FAILED DAY:           -DAY_FAIL_PENALTY (5)   and the streak resets to 0.
#       - BREACH:               -BREACH_PENALTY (20)    and the streak resets (see below).
#     STREAK ESCALATION replaces the old "every 4th won day = +1.0" jackpot (which created a gamble-near-a-
#     multiple incentive): every ADDITIONAL consecutive won day pays +STREAK_BONUS more, i.e. day N of a streak
#     pays DAY_PASS + STREAK_BONUS*min(N-1, STREAK_BONUS_CAP). The CAP bounds the per-day reward (training
#     stability: an unbounded ramp blows up the value function) while keeping "each day matters more". Because
#     the smooth per-day reward is the same shape every day, there is no longer a lumpy jackpot to gamble for. ---
FTMO_DAY_PASS_REWARD: float = 10.0          # reward when the day ENDS >= +2.5% of initial (a won day)
FTMO_DAY_FAIL_PENALTY: float = 5.0          # penalty when the day ENDS below +2.5% (a failed day) — also resets the streak
FTMO_STREAK_BONUS: float = 1.0              # +this per ADDITIONAL consecutive won day (day N pays +bonus*min(N-1, cap))
FTMO_STREAK_BONUS_CAP: float = 10.0         # cap the escalation (value-function stability); 0 = no streak bonus

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
# operator 2026-06-29: SCALED UP with the day rewards. With a won day worth +10, a 0.10 seek pull would be
# invisible -> the bot would learn ONLY from the sparse midnight outcome and lose the dense "climb toward the
# target" gradient (and the proactive wall-avoidance below). Seek is kept ~1/3 of the day reward so the climb is
# felt every step but winning the day is still the bigger prize. Still high-water-mark capped (<= weight/won day).
FTMO_TARGET_SEEK_WEIGHT: float = 3.0        # dense reward for new progress toward +2.5%/day (<= this per won day)
FTMO_IDLE_DAY_PENALTY: float = 0.02         # penalty for a day with ZERO trades (anti-"hiding"; seek+day reward now dominate)

# --- DRAWDOWN-PROXIMITY penalty + a SMALLER breach cliff (operator 2026-06-28). The old design only
#     punished a breach at the -1.0 CLIFF, so "do nothing / survive" was the dominant attractor. Now:
#     (a) a GRADUAL penalty grows as equity nears the trailing wall (so approaching the wall costs before
#     you hit it, and the bot plans away from it), and (b) the breach cliff drops 1.0 -> 0.2 so steady
#     +2.5% progress beats merely surviving. The streak reset (a breach restarts the 40-won-days count) is
#     now the real breach deterrent, not a giant per-step penalty. Tune from the dashboard DIAGNOSIS. ---
# operator 2026-06-29: dd-proximity SCALED UP (0.02 -> 2.0) so nearing the 4% wall costs real points EVERY step
# (quadratic: cheap far away, painful up close) and the bot plans away from it PROACTIVELY at the new reward
# scale. breach cliff RAISED to 20 (2x a won day) so a blow-up is genuinely feared NOW (the agent is near-sighted
# at gamma 0.9995 ~ <1 day with the multi-symbol cycle, so the streak-reset alone is too delayed a deterrent).
FTMO_DD_PROXIMITY_COEF: float = 2.0        # per-step penalty = coef * (dd_used_fraction_of_wall)^2 (0 = off)
FTMO_BREACH_PENALTY: float = 20.0          # the breach cliff (2x a won day) — felt immediately + the streak resets
FTMO_PASS_BONUS: float = 1.0               # the +10% CHALLENGE pass terminal bonus (EVAL only; training continues past +10%)
# --- v1.7.0 trade-risk CLOSE bonuses (PortfolioEnv only, PnL-capped, default 0 = off). Turn on in the
# training path. band_stack: enter ABOVE/BELOW BB200 & BB10 (dev1) on 1m+5m and close in profit, day net up.
# reentry: a with-trend re-entry that pays off. Small so they amplify a real win, never fabricate reward. ---
# operator 2026-06-29 CONVICTION bonus: a PnL-capped nudge paid when a trade was ENTERED with >=2 of the 3
# strong-setup alphas (CCI|>160| 5m+30m, BB200&BB20 double-breakout any-TF, fwd-SMA(4) 5m+30m) CONFIRMING its
# direction AND it CLOSES in profit (day net up). Reads the 3 alpha slots (no precompute). Kept MODEST on
# purpose: a "big incentive to trade" fights the 40-won-day goal, so this only breaks ties toward high-
# conviction setups and can NEVER pay for a loser (the min(bonus, trade-PnL) cap is shared with the others).
FTMO_CONVICTION_BONUS: float = 0.0         # ceiling for the >=2-confirm conviction bonus (e.g. 0.1; PnL-capped)
FTMO_BAND_STACK_BONUS: float = 0.0         # bonus for a band-stacked entry that closes in profit (e.g. 0.005)
FTMO_REENTRY_BONUS: float = 0.0            # nudge for a with-trend re-entry that pays off (e.g. 0.003)
# v1.10.0 HUGGING-PRESSURE (operator's heavy "Shifted SMA Hugging Pressure" agent). PER-STEP, HEAVY by default.
# hug_pressure_bonus: ride a >=2-TF shifted-SMA hug (aligned continuation). hug_miss_penalty: sit out a CLEAN
# one on an INDEX/METAL (the stick is 2x the carrot). Gated off when exhaustion/extension/decay conflict.
FTMO_HUG_PRESSURE_BONUS: float = 0.01      # per-step bonus for riding a >=2-TF hug (very heavy; tune down if needed)
FTMO_HUG_MISS_PENALTY: float = 0.02        # per-step penalty for sitting out a clean index/metal hug (very heavy)
# v1.12.0 OVERTRADING penalty (the scalper should be selective): a discrete penalty per NEW open once today's
# trade count is at/over the soft cap. Discourages churn without a per-step accumulation blow-up.
FTMO_OVERTRADE_SOFT_CAP: float = 15.0      # trades/day before the penalty kicks in
FTMO_OVERTRADE_PENALTY: float = 0.1        # subtracted per over-cap open (interpretable; ~ -1.5/day at 30 trades)

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
