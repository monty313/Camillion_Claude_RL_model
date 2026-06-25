# Registration helper for RegimePulseTrend5m30mAlpha (auto-assigns the next free slot).
from src.strategies.regime_pulse_trend_5m_30m_alpha import RegimePulseTrend5m30mAlpha


def register(registry) -> int:
    return registry.register(RegimePulseTrend5m30mAlpha())
