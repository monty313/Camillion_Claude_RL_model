# Registration helper for SmaReversionRally30m4hAlpha (auto-assigns the next free slot).
from src.strategies.sma_reversion_rally_30m_4h_alpha import SmaReversionRally30m4hAlpha


def register(registry) -> int:
    return registry.register(SmaReversionRally30m4hAlpha())
