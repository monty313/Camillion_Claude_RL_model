# =====================================================================
# WHEN 2026-06-25 | WHO Claude for Monty
# WHY  Per-asset contract specs + lot-size CALIBRATION. The PnL math is
#      `position * price_move * position_size`, so a SINGLE fixed position_size
#      cannot work across EURUSD (~1.1), gold (~2000) and US30 (~40000) -- the
#      same number is sane for FX and absurd for indices. Each asset needs its own
#      contract size, and the lot count must be set so capturing a typical DAY's
#      range ~= the +2.5% daily target while a normal adverse move stays inside 4%.
# WHERE config/asset_specs.py
# HOW  contract_size = account $ P&L per 1.0 PRICE move per 1 lot (quote ~ USD).
#      env position_size = contract_size * lots. Helpers calibrate `lots` from each
#      asset's typical daily range so the challenge math is WELL-POSED.
# DEPENDS_ON: (stdlib only)  | USED_BY: env wiring, notebooks, sizing observation.
# CHANGE_NOTES(IRAC): I: fixed 100k size makes +2.5%/day impossible on real EURUSD
#   and nonsensical on gold/US30. R: operator "per-asset conversion + reachable
#   2.5%/day, safe under 4%". A: contract specs + calibrated_position_size(). C:
#   correct account-$ PnL per asset -> the bot can actually pass on real data.
# =====================================================================
"""Per-asset contract specs and lot-size calibration (so +2.5%/day is well-posed)."""
from __future__ import annotations
from dataclasses import dataclass


MAX_LEVERAGE: float = 100.0     # FTMO 1:100. Calibrated sizes use only ~2.5-3.4:1, so the
                                # 4% drawdown wall -- NOT leverage -- is the real size limit.


@dataclass(frozen=True)
class AssetSpec:
    symbol: str
    contract_size: float        # account $ per 1.0 PRICE move per 1 lot (= env position_size at 1 lot)
    pip: float                  # price increment of "1 pip / point" (readability)
    typical_daily_range: float  # typical |daily move| in PRICE units (for calibration)


# contract_size: FX standard lot = 100,000 units (1.0 price move = $100,000);
# gold = 100 oz/lot ($1 move = $100); US30 ~ $1/point. typical_daily_range in PRICE.
SPECS: dict[str, AssetSpec] = {
    "EURUSD": AssetSpec("EURUSD", contract_size=100_000.0, pip=0.0001, typical_daily_range=0.0080),  # ~80 pips
    "GBPUSD": AssetSpec("GBPUSD", contract_size=100_000.0, pip=0.0001, typical_daily_range=0.0110),  # ~110 pips
    "XAUUSD": AssetSpec("XAUUSD", contract_size=100.0,     pip=0.10,   typical_daily_range=20.0),    # ~$20/day
    "US30":   AssetSpec("US30",   contract_size=1.0,       pip=1.0,    typical_daily_range=400.0),   # ~400 pts/day
}


def value_per_point(symbol: str) -> float:
    """Account $ per 1.0 PRICE move per 1 lot (i.e. env position_size for ONE lot)."""
    return SPECS[symbol].contract_size


def lots_for_daily_target(symbol: str, *, account: float = 100_000.0,
                          target_pct: float = 2.5) -> float:
    """Lots so that capturing ~one typical daily range == the daily target.
    (A full adverse day at this size = the same %, which is < the 4% wall.)"""
    s = SPECS[symbol]
    target_usd = account * target_pct / 100.0
    per_lot_full_day = s.contract_size * s.typical_daily_range   # $ for a full day at 1 lot
    return target_usd / per_lot_full_day if per_lot_full_day > 0 else 0.0


def calibrated_position_size(symbol: str, *, account: float = 100_000.0,
                             target_pct: float = 2.5, lots: float | None = None) -> float:
    """env position_size for `symbol` = contract_size * lots (calibrated to ~target%/day).
    Pass `lots` to override the calibration with a fixed size."""
    s = SPECS[symbol]
    L = lots if lots is not None else lots_for_daily_target(symbol, account=account, target_pct=target_pct)
    return s.contract_size * L


def leverage_used(symbol: str, price: float, lots: float, *, account: float = 100_000.0) -> float:
    """Leverage = notional / account for `lots` of `symbol` at `price`. Must be <= MAX_LEVERAGE.
    Notional = lots * contract_size * price."""
    notional = lots * SPECS[symbol].contract_size * float(price)
    return notional / account if account else float("inf")

