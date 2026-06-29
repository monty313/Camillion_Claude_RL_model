# Registration helper for CciX160Align5m30mAlpha (auto-assigns the next free slot).
from src.strategies.cci_x160_align_5m_30m_alpha import CciX160Align5m30mAlpha


def register(registry) -> int:
    return registry.register(CciX160Align5m30mAlpha())
