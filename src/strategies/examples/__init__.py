# WHEN 2026-06-21 (Phase 1) | WHO Claude for Monty
# WHY  Example STRATEGIES (internal logic) whose OUTPUTS become alphas.
# WHERE src/strategies/examples/__init__.py
# DEPENDS_ON the three example strategy modules, src/strategies/registry.py
# USED_BY notebooks, src/env (Phase 1), tests.
"""Example strategies + register_examples(alpha_registry) helper.

Reminder of the contract: these classes are STRATEGIES (internal logic). Each
one's output is an ALPHA (+1/-1/0) that the policy sees in a fixed alpha slot.
"""
from __future__ import annotations
from src.strategies.examples.sma_trend import SmaTrendStrategy
from src.strategies.examples.rsi_reversion import RsiReversionStrategy
from src.strategies.examples.bollinger_breakout import BollingerBreakoutStrategy

EXAMPLE_STRATEGIES = (SmaTrendStrategy, RsiReversionStrategy, BollingerBreakoutStrategy)
EXAMPLE_ALPHAS = EXAMPLE_STRATEGIES  # backward-compat alias


def register_examples(alpha_registry, tf: str = "1m") -> list[int]:
    """Register the 3 example strategies into alpha slots. Returns slot ids."""
    return [alpha_registry.register(cls(tf=tf)) for cls in EXAMPLE_STRATEGIES]
