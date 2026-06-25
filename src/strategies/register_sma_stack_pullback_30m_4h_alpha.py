# Registration helper for SmaStackPullback30m4hAlpha (auto-assigns the next free slot).
from src.strategies.sma_stack_pullback_30m_4h_alpha import SmaStackPullback30m4hAlpha


def register(registry) -> int:
    return registry.register(SmaStackPullback30m4hAlpha())
