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
    profit_target_total_pct: float = 10.0


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
