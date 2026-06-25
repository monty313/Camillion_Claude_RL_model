# Registration helper for RegimePulsePullback30m4hAlpha (auto-assigns the next free slot).
from src.strategies.regime_pulse_pullback_30m_4h_alpha import RegimePulsePullback30m4hAlpha


def register(registry) -> int:
    return registry.register(RegimePulsePullback30m4hAlpha())
