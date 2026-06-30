# =====================================================================
# WHEN:   2026-06-21 (created Phase 0; updated same day: runtime-editable risk)
# WHO:    Claude (Camillion build agent) for Monty
# WHY:    Build the FTMO and FREE rule configs FROM variables.py so target /
#         trailing-DD / trailing on-off are editable at runtime in BOTH modes
#         with NO RETRAIN. Provides a live mutator for the Jarvis UI/Colab.
# WHERE:  config/ftmo_config.py
# HOW:    Frozen dataclasses constructed from variables; loaders keyed off
#         variables.MODE; update_risk_settings() writes back to variables and
#         returns the rebuilt active config.
# DEPENDS_ON: config/variables.py
# USED_BY: src/risk/*, src/account/*, src/jarvis/*, tests/test_ftmo_free_mode.py
# CHANGE_NOTES (IRAC):
#   I: Two modes must load independently AND be retunable without retraining.
#   R: Quantra FTMO numbers (2.5% / 4% / two-phase) + Monty runtime-edit note.
#   A: Read every risk knob from variables; add update_risk_settings(...).
#   C: Live risk retuning + a stable percentage observation = reuse one model
#      across many challenge configurations.
# =====================================================================
"""FTMO + FREE rule configs, built from variables.py (runtime-editable)."""
from __future__ import annotations
from dataclasses import dataclass
from config import variables as V


@dataclass(frozen=True)
class FTMOConfig:
    """FTMO-style rules. Defaults are the locked numbers; all are editable
    via variables.py / update_risk_settings() WITHOUT retraining."""
    mode: str = "FTMO"
    starting_balance: float = 100_000.0
    daily_target_pct: float = 2.5
    daily_drawdown_pct: float = 5.0
    max_total_drawdown_pct: float = 10.0
    trailing_drawdown_pct: float = 4.0
    trailing_enabled: bool = True
    two_phase_enabled: bool = True
    phase2_trailing_pct: float = 1.0
    phase2_continue: bool = False          # after banking +2.5%: keep trading under the 1% trail?
    profit_target_total_pct: float = 10.0
    # alpha-shaping (ON by default 2026-06-27; deliberate departure from "reward=equity only" for PortfolioEnv)
    alpha_reward_enabled: bool = True
    # operator 2026-06-29 reward REBALANCE (see config/variables.py for the full rationale).
    alpha_agree_bonus: float = 0.01        # USE the alphas (10x): profitable close that agreed with >=50% firing alphas
    alpha_against_penalty: float = 0.01    # penalty for OPENING against >=50% firing alphas (10x)
    alpha_beat_bonus: float = 0.05         # BEAT the alphas (emphasised): out-earn the consensus; profit + day-up + capped at PnL
    # per-day consistency (operator 2026-06-29): a "won day" ENDS >= +2.5% of initial, paid +day_pass_reward PLUS
    # an ESCALATING streak bonus; a failed day pays -day_fail_penalty and resets the streak. The escalation
    # (streak_bonus per ADDITIONAL consecutive won day, capped) REPLACES the old every-4th jackpot.
    day_pass_reward: float = 10.0          # reward a won day
    day_fail_penalty: float = 5.0          # penalty for a failed day (ended below +2.5%) — also resets the streak
    streak_bonus: float = 1.0              # +this per ADDITIONAL consecutive won day (day N pays +bonus*min(N-1, cap))
    streak_bonus_cap: float = 10.0         # cap the escalation (value-function stability); 0 = no streak bonus
    # seek-the-target vs hide rebalance (operator 2026-06-28/29, PortfolioEnv only). target_seek_weight: dense
    # reward for NEW progress toward +2.5%/day (high-water-mark, <= this per won day), SCALED UP with the day
    # reward so the climb stays visible; idle_day_penalty: penalty for a day with ZERO trades. 0 = off.
    target_seek_weight: float = 3.0
    idle_day_penalty: float = 0.02
    # drawdown-proximity penalty + breach cliff (operator 2026-06-28/29; PortfolioEnv). dd_proximity_coef:
    # per-step penalty = coef*(dd/wall)^2 as equity nears the trailing wall (SCALED UP so it bites near the wall);
    # breach_penalty: the breach cliff (RAISED to 20 = 2x a won day, felt immediately); pass_bonus: the +10%
    # CHALLENGE pass terminal bonus (eval only — training continues past +10% for long won-day streaks).
    dd_proximity_coef: float = 2.0
    breach_penalty: float = 20.0
    pass_bonus: float = 1.0
    # v1.7.0 trade-risk CLOSE bonuses (operator 2026-06-29; PortfolioEnv only, PnL-capped, default 0 = off).
    # band_stack_bonus: paid when a trade ENTERED with price stacked above (long) / below (short) BB200(dev1) AND
    # BB10(dev1) on BOTH 1m and 5m CLOSES in profit with the day net up. reentry_bonus: a small nudge for a
    # with-trend RE-ENTRY (re-opening this symbol in its last-close direction after price kept going) that pays off.
    band_stack_bonus: float = 0.0
    reentry_bonus: float = 0.0
    # conviction_bonus (operator 2026-06-29): PnL-capped nudge when >=2 of the 3 strong-setup alphas confirmed
    # the trade's direction at entry AND it closes in profit (day net up). Reads the alpha slots; no precompute.
    conviction_bonus: float = 0.0
    # v1.10.0 HUGGING-PRESSURE reward (operator's heavy momentum agent). hug_pressure_bonus: per-step reward for
    # RIDING a >=2-TF shifted-SMA hug (aligned + continuation, no exhaustion/extension/decay conflict).
    # hug_miss_penalty: heavier per-step penalty for sitting out a CLEAN >=2-TF hug on an INDEX/METAL. HEAVY by
    # default per the operator (and tunable from the DIAGNOSIS line). Set 0.0 to disable a side.
    hug_pressure_bonus: float = 0.0
    hug_miss_penalty: float = 0.0
    # v1.12.0 OVERTRADING penalty: a discrete penalty subtracted each time a NEW trade OPENS while today's trade
    # count is already at/over the soft cap -> discourages churn (the "super scalper" should be selective).
    overtrade_soft_cap: float = 15.0
    overtrade_penalty: float = 0.0


@dataclass(frozen=True)
class FreeModeConfig:
    """Operator-defined rules from variables.py (any target / risk)."""
    mode: str
    starting_balance: float
    daily_target_pct: float
    max_daily_drawdown_pct: float
    max_total_drawdown_pct: float
    trailing_drawdown_pct: float
    trailing_enabled: bool


def load_ftmo_config() -> FTMOConfig:
    """Build FTMO config from the (editable) FTMO_* variables."""
    return FTMOConfig(
        starting_balance=V.STARTING_BALANCE,
        daily_target_pct=V.FTMO_DAILY_TARGET_PCT,
        daily_drawdown_pct=V.FTMO_DAILY_DRAWDOWN_PCT,
        max_total_drawdown_pct=V.FTMO_MAX_TOTAL_DRAWDOWN_PCT,
        trailing_drawdown_pct=V.FTMO_TRAILING_DRAWDOWN_PCT,
        trailing_enabled=V.FTMO_TRAILING_ENABLED,
        two_phase_enabled=V.FTMO_TWO_PHASE_ENABLED,
        phase2_trailing_pct=V.FTMO_PHASE2_TRAILING_PCT,
        phase2_continue=getattr(V, "FTMO_PHASE2_CONTINUE", False),
        profit_target_total_pct=V.FTMO_PROFIT_TARGET_PCT,
        alpha_reward_enabled=getattr(V, "FTMO_ALPHA_REWARD_ENABLED", True),
        alpha_agree_bonus=getattr(V, "FTMO_ALPHA_AGREE_BONUS", 0.001),
        alpha_against_penalty=getattr(V, "FTMO_ALPHA_AGAINST_PENALTY", 0.001),
        alpha_beat_bonus=getattr(V, "FTMO_ALPHA_BEAT_BONUS", 0.002),
        day_pass_reward=getattr(V, "FTMO_DAY_PASS_REWARD", 10.0),
        day_fail_penalty=getattr(V, "FTMO_DAY_FAIL_PENALTY", 5.0),
        streak_bonus=getattr(V, "FTMO_STREAK_BONUS", 1.0),
        streak_bonus_cap=getattr(V, "FTMO_STREAK_BONUS_CAP", 10.0),
        target_seek_weight=getattr(V, "FTMO_TARGET_SEEK_WEIGHT", 0.10),
        idle_day_penalty=getattr(V, "FTMO_IDLE_DAY_PENALTY", 0.02),
        dd_proximity_coef=getattr(V, "FTMO_DD_PROXIMITY_COEF", 0.02),
        breach_penalty=getattr(V, "FTMO_BREACH_PENALTY", 0.2),
        pass_bonus=getattr(V, "FTMO_PASS_BONUS", 1.0),
        band_stack_bonus=getattr(V, "FTMO_BAND_STACK_BONUS", 0.0),
        reentry_bonus=getattr(V, "FTMO_REENTRY_BONUS", 0.0),
        conviction_bonus=getattr(V, "FTMO_CONVICTION_BONUS", 0.0),
        hug_pressure_bonus=getattr(V, "FTMO_HUG_PRESSURE_BONUS", 0.0),
        hug_miss_penalty=getattr(V, "FTMO_HUG_MISS_PENALTY", 0.0),
        overtrade_soft_cap=getattr(V, "FTMO_OVERTRADE_SOFT_CAP", 15.0),
        overtrade_penalty=getattr(V, "FTMO_OVERTRADE_PENALTY", 0.0),
    )


def load_free_config() -> FreeModeConfig:
    """Build FREE config from variables.py."""
    return FreeModeConfig(
        mode="FREE",
        starting_balance=V.STARTING_BALANCE,
        daily_target_pct=V.DAILY_TARGET_PCT,
        max_daily_drawdown_pct=V.MAX_DAILY_DRAWDOWN_PCT,
        max_total_drawdown_pct=V.MAX_TOTAL_DRAWDOWN_PCT,
        trailing_drawdown_pct=V.TRAILING_DRAWDOWN_PCT,
        trailing_enabled=V.TRAILING_DRAWDOWN_ENABLED,
    )


def load_active_config():
    """Return whichever config matches variables.MODE ('FTMO' or 'FREE')."""
    return load_ftmo_config() if V.MODE.upper() == "FTMO" else load_free_config()


def update_risk_settings(
    *,
    mode: str | None = None,
    daily_target_pct: float | None = None,
    trailing_pct: float | None = None,
    trailing_enabled: bool | None = None,
    daily_drawdown_pct: float | None = None,
    max_total_drawdown_pct: float | None = None,
):
    """Change target / trailing-DD / trailing on-off at RUNTIME (NO RETRAIN).

    Writes to the ACTIVE mode's variables and returns the rebuilt config.
    Safe to call live (e.g. from the Jarvis UI) because the observation only
    exposes these as percentages, so the trained policy still reads them right.
    """
    if mode is not None:
        V.MODE = mode.upper()
    active = V.MODE.upper()
    if active == "FTMO":
        if daily_target_pct is not None:
            V.FTMO_DAILY_TARGET_PCT = float(daily_target_pct)
        if trailing_pct is not None:
            V.FTMO_TRAILING_DRAWDOWN_PCT = float(trailing_pct)
        if trailing_enabled is not None:
            V.FTMO_TRAILING_ENABLED = bool(trailing_enabled)
        if daily_drawdown_pct is not None:
            V.FTMO_DAILY_DRAWDOWN_PCT = float(daily_drawdown_pct)
        if max_total_drawdown_pct is not None:
            V.FTMO_MAX_TOTAL_DRAWDOWN_PCT = float(max_total_drawdown_pct)
    else:
        if daily_target_pct is not None:
            V.DAILY_TARGET_PCT = float(daily_target_pct)
        if trailing_pct is not None:
            V.TRAILING_DRAWDOWN_PCT = float(trailing_pct)
        if trailing_enabled is not None:
            V.TRAILING_DRAWDOWN_ENABLED = bool(trailing_enabled)
        if daily_drawdown_pct is not None:
            V.MAX_DAILY_DRAWDOWN_PCT = float(daily_drawdown_pct)
        if max_total_drawdown_pct is not None:
            V.MAX_TOTAL_DRAWDOWN_PCT = float(max_total_drawdown_pct)
    return load_active_config()
