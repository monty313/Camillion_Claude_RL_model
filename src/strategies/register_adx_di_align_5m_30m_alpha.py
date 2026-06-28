# Registration helper for AdxDiAlign5m30mAlpha (auto-assigns the next free slot).
from src.strategies.adx_di_align_5m_30m_alpha import AdxDiAlign5m30mAlpha


def register(registry) -> int:
    return registry.register(AdxDiAlign5m30mAlpha())
