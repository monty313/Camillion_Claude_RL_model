# Registration helper for FwdSma4Align5m30mAlpha (auto-assigns the next free slot).
from src.strategies.fwd_sma4_align_5m_30m_alpha import FwdSma4Align5m30mAlpha


def register(registry) -> int:
    return registry.register(FwdSma4Align5m30mAlpha())
