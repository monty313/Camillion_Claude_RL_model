# Tests 9 & 10: Jarvis UI imports without crashing; Barbershop modules import
# and run without crashing.
import numpy as np
from src.observation import builder as B
from src.account.account_state import AccountState


def test_barbershop_modules_import_and_run():
    from src.barbershop import (scoreboard, risk_doctor, feature_doctor,
                                day_replay, trade_autopsy, signal_doctor)
    acc = AccountState(100000.0)
    assert "daily" in scoreboard.scoreboard(acc)
    assert "reasons" in risk_doctor.diagnose(acc)
    assert feature_doctor.inspect(B.zeros())["shape_ok"]
    assert callable(day_replay.build_replay) and callable(trade_autopsy.autopsy)
    assert callable(signal_doctor.report)


def test_jarvis_modules_import_without_crashing():
    from src.jarvis import (app, workflow_map, agent_bus,
                            omega_agent, justice_agent, jarvis_judge)
    from src.jarvis.agent_bus import AgentBus
    from src.jarvis.jarvis_judge import JarvisJudge
    bus = AgentBus()
    verdict = JarvisJudge().cycle(B.zeros(), AccountState(100000.0), bus)
    assert verdict.kind == "verdict"
    assert len(workflow_map.all_modules()) == 9
    assert callable(app.main)   # exists but not invoked (needs streamlit)
