# Registration helper for RegimePulseTrend30m4hAlpha (auto-assigns the next free slot).
from src.strategies.regime_pulse_trend_30m_4h_alpha import RegimePulseTrend30m4hAlpha


def register(registry) -> int:
    return registry.register(RegimePulseTrend30m4hAlpha())
