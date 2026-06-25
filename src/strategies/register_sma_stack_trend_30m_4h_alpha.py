# Registration helper for SmaStackTrend30m4hAlpha (auto-assigns the next free slot).
from src.strategies.sma_stack_trend_30m_4h_alpha import SmaStackTrend30m4hAlpha


def register(registry) -> int:
    return registry.register(SmaStackTrend30m4hAlpha())
