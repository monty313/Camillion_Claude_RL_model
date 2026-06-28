# Registration helper for AdxDiAlign30m4hAlpha (auto-assigns the next free slot).
from src.strategies.adx_di_align_30m_4h_alpha import AdxDiAlign30m4hAlpha


def register(registry) -> int:
    return registry.register(AdxDiAlign30m4hAlpha())
