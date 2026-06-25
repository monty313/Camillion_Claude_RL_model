# Registration helper for SmaStackTrend5m30mAlpha (auto-assigns the next free slot).
from src.strategies.sma_stack_trend_5m_30m_alpha import SmaStackTrend5m30mAlpha


def register(registry) -> int:
    return registry.register(SmaStackTrend5m30mAlpha())
