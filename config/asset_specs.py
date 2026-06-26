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
import math
from dataclasses import dataclass
from config import constants as C

_BARS_PER_DAY = 1440   # 1-minute bars/day -> ties the daily range to a typical 1m ATR

ASSET_CLASSES = C.ASSET_CLASSES   # contract owns the one-hot order: pair/index/metal/energy/crypto

MAX_LEVERAGE: float = 100.0     # FTMO 1:100. Calibrated sizes use only ~2.5-3.4:1, so the
                                # 4% drawdown wall -- NOT leverage -- is the real size limit.


@dataclass(frozen=True)
class AssetSpec:
    symbol: str
    contract_size: float        # account $ per 1.0 PRICE move per 1 lot (= env position_size at 1 lot)
    pip: float                  # price increment of "1 pip / point" (readability)
    typical_daily_range: float  # typical |daily move| in PRICE units (for calibration)
    asset_class: str = "pair"   # "pair" | "index" | "metal" (behaviour differs by class)


# contract_size: FX standard lot = 100,000 units (1.0 price move = $100,000);
# gold = 100 oz/lot ($1 move = $100); US30 ~ $1/point. typical_daily_range in PRICE.
# MOVEMENT PROFILES (the 4 we trade -- "how they move"):
#   EURUSD ~80 pips/day  -- lowest vol, mean-reverts intraday, most active London-NY
#   GBPUSD ~110 pips/day -- more volatile than EUR ("the dragon"), London-NY
#   XAUUSD ~$20/day      -- trends hard, risk-off haven, volatile, London-NY (+ some Asian)
#   US30   ~400 pts/day  -- trends, gaps, risk-on, NY-session driven
SPECS: dict[str, AssetSpec] = {
    "EURUSD": AssetSpec("EURUSD", 100_000.0, 0.0001, 0.0080, asset_class="pair"),   # ~80 pips
    "GBPUSD": AssetSpec("GBPUSD", 100_000.0, 0.0001, 0.0110, asset_class="pair"),   # ~110 pips
    "XAUUSD": AssetSpec("XAUUSD", 100.0,     0.10,   20.0,   asset_class="metal"),  # ~$20/day
    "US30":   AssetSpec("US30",   1.0,       1.0,    400.0,  asset_class="index"),  # ~400 pts/day
}


def typical_atr(symbol: str | None) -> float | None:
    """A principled per-asset 1-minute ATR baseline = typical_daily_range / sqrt(bars_per_day)
    (random-walk scaling: a day's range ~ ATR_1m * sqrt(1440)). This is the bot's anchor for
    'how this asset normally moves', so 'volatility regime' = current ATR vs this baseline."""
    s = SPECS.get(symbol) if symbol else None
    return (s.typical_daily_range / math.sqrt(_BARS_PER_DAY)) if s else None


# Broker symbol roots for class inference -- the REAL challenge trades the FULL FTMO broker
# (130+ instruments), not just SPECS. Extend as the universe grows.
_METAL_ROOTS  = ("XAU", "XAG", "XPT", "XPD", "GOLD", "SILVER")                       # gold/silver/plat/pall
_ENERGY_ROOTS = ("USOIL", "UKOIL", "WTI", "BRENT", "OIL", "NGAS", "NATGAS", "XNG", "XBR", "XTI")
_CRYPTO_ROOTS = ("BTC", "ETH", "LTC", "XRP", "ADA", "DOT", "DOGE", "SOL", "BNB", "XLM",
                 "AAVE", "LINK", "BCH", "AVAX", "MATIC", "UNI", "ATOM", "ALGO", "TRX", "ETC")
# FTMO indices (their symbol names): US30 US100 US500 US2000 GER40 UK100 FRA40 EU50 SPN35
# JP225 HK50 AUS200 (+ .cash variants -> handled by substring after stripping non-alnum).
_INDEX_ROOTS  = ("US30", "US500", "US100", "US2000", "SPX", "NAS", "NDX", "DJI", "DOW", "GER",
                 "DE40", "DE30", "UK100", "FTSE", "JP225", "JPN", "HK50", "AUS200", "EU50",
                 "STOXX", "FRA", "FRA40", "ESP", "ESP35", "SPN", "SPN35", "NETH", "NED", "SWI",
                 "SUI", "USTEC", "CHINA50", "CN50", "US2000")


def infer_asset_class(symbol: str | None) -> str | None:
    """Best-effort class from the symbol NAME so ANY FTMO instrument is covered:
    XAU/XAG.. -> metal; oil/gas -> energy; BTC/ETH.. -> crypto; index roots -> index;
    6-letter FX (two ccy codes) -> pair; else None (unknown -> safe all-zeros one-hot)."""
    if not symbol:
        return None
    s = "".join(ch for ch in symbol.upper() if ch.isalnum())
    if any(s.startswith(r) for r in _METAL_ROOTS):
        return "metal"
    if any(r in s for r in _ENERGY_ROOTS):
        return "energy"
    if any(s.startswith(r) for r in _CRYPTO_ROOTS):
        return "crypto"
    if any(r in s for r in _INDEX_ROOTS):
        return "index"
    core = "".join(ch for ch in s if ch.isalpha())
    if len(core) == 6:                      # two 3-letter currency codes -> a forex pair
        return "pair"
    return None


def asset_class(symbol: str | None) -> str | None:
    """'pair' | 'index' | 'metal' for ANY broker symbol: SPECS override, else inferred, else None."""
    if not symbol:
        return None
    s = SPECS.get(symbol)
    return s.asset_class if s else infer_asset_class(symbol)


def class_one_hot(symbol: str | None) -> list[float]:
    """One-hot over ASSET_CLASSES (all zeros if the class is unknown -> safe degradation)."""
    cls = asset_class(symbol)
    return [1.0 if cls == c else 0.0 for c in ASSET_CLASSES]


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

