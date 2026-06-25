# Registration helper for SmaStackPullback5m30mAlpha (auto-assigns the next free slot).
from src.strategies.sma_stack_pullback_5m_30m_alpha import SmaStackPullback5m30mAlpha


def register(registry) -> int:
    return registry.register(SmaStackPullback5m30mAlpha())
