# Registration helper for RegimePulsePullback5m30mAlpha (auto-assigns the next free slot).
from src.strategies.regime_pulse_pullback_5m_30m_alpha import RegimePulsePullback5m30mAlpha


def register(registry) -> int:
    return registry.register(RegimePulsePullback5m30mAlpha())
