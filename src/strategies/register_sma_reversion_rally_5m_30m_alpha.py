# Registration helper for SmaReversionRally5m30mAlpha (auto-assigns the next free slot).
from src.strategies.sma_reversion_rally_5m_30m_alpha import SmaReversionRally5m30mAlpha


def register(registry) -> int:
    return registry.register(SmaReversionRally5m30mAlpha())
