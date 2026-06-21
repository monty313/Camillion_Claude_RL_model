# Test 6: FTMO and FREE configs load correctly + independently; risk knobs are
# runtime-editable (no retrain); breach detection works in both modes.
from config import ftmo_config as F, variables as V
from src.account.account_state import AccountState
from src.risk import breach_detector as BD


def test_configs_load_independently():
    ftmo, free = F.load_ftmo_config(), F.load_free_config()
    assert ftmo.mode == "FTMO" and free.mode == "FREE"
    assert ftmo.daily_target_pct == 2.5 and ftmo.trailing_drawdown_pct == 4.0
    assert hasattr(free, "max_daily_drawdown_pct")  # FREE-specific field


def test_runtime_editable_no_retrain():
    try:
        cfg = F.update_risk_settings(mode="FTMO", daily_target_pct=3.0,
                                     trailing_pct=6.0, trailing_enabled=False)
        assert cfg.daily_target_pct == 3.0
        assert cfg.trailing_drawdown_pct == 6.0
        assert cfg.trailing_enabled is False
        cfg2 = F.update_risk_settings(mode="FREE", daily_target_pct=1.0, trailing_pct=2.0)
        assert cfg2.mode == "FREE" and cfg2.daily_target_pct == 1.0
    finally:
        F.update_risk_settings(mode="FTMO", daily_target_pct=2.5,
                               trailing_pct=4.0, trailing_enabled=True)
        V.MODE = "FTMO"


def test_active_config_switch():
    try:
        V.MODE = "FREE"
        assert F.load_active_config().mode == "FREE"
        V.MODE = "FTMO"
        assert F.load_active_config().mode == "FTMO"
    finally:
        V.MODE = "FTMO"


def test_breach_detection_both_modes():
    acc = AccountState(100000.0)
    acc.mark_equity(100000.0)
    acc.mark_equity(94000.0)        # -6% from peak -> breach
    try:
        V.MODE = "FTMO"
        assert BD.detect(acc).breached
        V.MODE = "FREE"
        assert BD.detect(acc).breached
    finally:
        V.MODE = "FTMO"
