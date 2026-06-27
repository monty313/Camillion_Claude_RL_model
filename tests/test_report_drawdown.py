# Report drawdown fix (2026-06-27): the day-by-day report must measure TRAILING drawdown the engine's way
# -- chronologically from the running peak -- not (max - min), which pairs a later peak with an earlier
# trough and overstates it (the old "5.14% BREACH" that the engine never acted on).
from src.training.daily_report import running_drawdown_pct


def test_running_drawdown_is_chronological_not_max_minus_min():
    # up to 110 then down to 104: trailing DD = (110-104)/110 = 5.45% — NOT (110-95)/... = a bigger number.
    dd = running_drawdown_pct([100, 95, 110, 104])
    assert abs(dd - (6.0 / 110.0 * 100.0)) < 1e-6      # 5.4545%
    assert dd < 6.0                                     # would have been ~15% under the old max-minus-min bug


def test_running_drawdown_simple_cases():
    assert running_drawdown_pct([100, 96]) == 4.0       # straight 4% drop from the peak
    assert running_drawdown_pct([100, 101, 102]) == 0.0  # only rising -> no drawdown
    assert running_drawdown_pct([100]) == 0.0
