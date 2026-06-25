# Per-asset lot calibration (2026-06-25): each asset is sized so capturing ~one typical
# daily range == the +2.5% daily target, a full adverse day stays inside the 4% wall, and
# leverage stays well under 1:100. Proves the same fixed size CANNOT work across assets.
from config.asset_specs import (SPECS, value_per_point, lots_for_daily_target,
                                calibrated_position_size, leverage_used, MAX_LEVERAGE)


def test_calibrated_size_hits_daily_target_each_asset():
    for sym, s in SPECS.items():
        ps = calibrated_position_size(sym, account=100_000.0, target_pct=2.5)
        pnl = ps * s.typical_daily_range            # $ from one typical day at this size
        assert abs(pnl - 2500.0) < 1.0, (sym, pnl)  # ~= +2.5% of $100k


def test_full_adverse_day_stays_inside_4pct():
    for sym, s in SPECS.items():                    # a full day AGAINST you = ~2.5% loss
        adverse = calibrated_position_size(sym) * s.typical_daily_range
        assert adverse <= 0.04 * 100_000.0          # well under the $4,000 (4%) wall


def test_leverage_within_1_to_100():
    prices = {"EURUSD": 1.10, "GBPUSD": 1.27, "XAUUSD": 2000.0, "US30": 40000.0}
    for sym in SPECS:
        lev = leverage_used(sym, prices[sym], lots_for_daily_target(sym))
        assert 0 < lev <= MAX_LEVERAGE, (sym, lev)
        assert lev < 5.0, (sym, lev)                # in practice only ~2.5-3.4x


def test_same_fixed_size_is_wrong_across_assets():
    # 100k is sane for FX but absurd for indices/gold -> per-asset sizing is required
    assert value_per_point("EURUSD") == 100_000.0
    assert value_per_point("XAUUSD") == 100.0
    assert value_per_point("US30") == 1.0
