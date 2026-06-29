# Registration helper for BbDoubleBreakoutAnyTfAlpha (auto-assigns the next free slot).
from src.strategies.bb_double_breakout_anytf_alpha import BbDoubleBreakoutAnyTfAlpha


def register(registry) -> int:
    return registry.register(BbDoubleBreakoutAnyTfAlpha())
