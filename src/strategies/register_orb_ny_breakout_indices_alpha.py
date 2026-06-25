# Registration helper for OrbNyBreakoutIndicesAlpha (auto-assigns the next free slot).
from src.strategies.orb_ny_breakout_indices_alpha import OrbNyBreakoutIndicesAlpha


def register(registry) -> int:
    return registry.register(OrbNyBreakoutIndicesAlpha())
