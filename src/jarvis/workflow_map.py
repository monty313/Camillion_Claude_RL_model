# WHEN 2026-06-21 (Phase 0) | WHO Claude for Monty
# WHY  The clickable Jarvis workflow graph data: each module's what/inputs/
#      outputs/files/dependents/tests/status. UI reads this; no logic here.
# WHERE src/jarvis/workflow_map.py | HOW a list of module dicts + lookup.
# DEPENDS_ON (none) | USED_BY src/jarvis/app.py, src/jarvis/*_agent.py.
"""Jarvis workflow map: the 9 pipeline modules and their metadata."""
from __future__ import annotations

MODULES: list[dict] = [
    {"id": "01", "title": "Data", "what": "MT5 feed / Drive / backtest cache",
     "inputs": ["MT5", "CSV/parquet"], "outputs": ["OHLCV bars"],
     "files": ["src/data/mt5_loader.py", "src/data/cache_builder.py",
               "src/data/data_contracts.py"],
     "depends_on": [], "tests": ["test_indicator_shapes.py"], "status": "phase1"},
    {"id": "02", "title": "Universe", "what": "symbols + timeframes",
     "inputs": ["config.variables.SYMBOLS", "constants.TIMEFRAMES"],
     "outputs": ["selected universe"], "files": ["config/variables.py", "config/constants.py"],
     "depends_on": ["01"], "tests": [], "status": "ready"},
    {"id": "03", "title": "Alphas", "what": "1..N strategy signals (+1/-1/0)",
     "inputs": ["indicators"], "outputs": ["64 alpha slots", "occupancy mask"],
     "files": ["src/strategies/base.py", "src/strategies/registry.py"],
     "depends_on": ["01", "02"],
     "tests": ["test_strategy_registry_shape.py"], "status": "ready"},
    {"id": "04", "title": "Alpha Combination", "what": "confluence / net signal / conflict",
     "inputs": ["alpha slots"], "outputs": ["buy/sell/active/net %", "last-5 memory", "accuracy"],
     "files": ["src/signals/signal_summary.py", "src/signals/signal_memory.py",
               "src/signals/signal_accuracy.py"],
     "depends_on": ["03"],
     "tests": ["test_signal_percentages.py", "test_signal_memory_last5.py",
               "test_signal_accuracy_no_leakage.py"], "status": "ready"},
    {"id": "R", "title": "Risk Model", "what": "FTMO drawdown guard (editable, no retrain)",
     "inputs": ["AccountState", "active config"], "outputs": ["breach report"],
     "files": ["src/risk/ftmo_rules.py", "src/risk/free_mode_rules.py",
               "src/risk/breach_detector.py"],
     "depends_on": ["06"], "tests": ["test_ftmo_free_mode.py"], "status": "ready"},
    {"id": "05", "title": "Portfolio Construction", "what": "trade selection / exposure",
     "inputs": ["net signal", "risk"], "outputs": ["position intents"],
     "files": ["src/env/trading_env.py"], "depends_on": ["04", "R"],
     "tests": [], "status": "phase1"},
    {"id": "O", "title": "Objective", "what": "+2.5%/day target (editable)",
     "inputs": ["config"], "outputs": ["reward shaping (Phase 1)"],
     "files": ["config/ftmo_config.py"], "depends_on": ["R"], "tests": [], "status": "phase1"},
    {"id": "C", "title": "Constraints", "what": "drawdown wall + trade limits",
     "inputs": ["config"], "outputs": ["action masks (Phase 1)"],
     "files": ["src/risk/breach_detector.py"], "depends_on": ["R"], "tests": [], "status": "phase1"},
    {"id": "06", "title": "Trading", "what": "execution + FTMO metrics",
     "inputs": ["position intents"], "outputs": ["AccountState", "trades"],
     "files": ["src/account/account_state.py", "src/account/trade_history.py"],
     "depends_on": ["05"], "tests": ["test_observation_contract.py"], "status": "phase1"},
]

_BY_ID = {m["id"]: m for m in MODULES}

def get_module(module_id: str) -> dict | None:
    return _BY_ID.get(module_id)

def all_modules() -> list[dict]:
    return list(MODULES)
