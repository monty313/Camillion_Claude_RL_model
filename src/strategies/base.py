# =====================================================================
# WHEN 2026-06-21 (Phase 0) | WHO Claude for Monty
# WHY  The contract every strategy ("alpha") obeys: emit a single directional
#      signal +1 (buy) / -1 (sell) / 0 (inactive = no setup, NOT a HOLD).
# WHERE src/strategies/base.py
# HOW  Abstract base class; subclasses implement compute_signal(ctx). The
#      public signal() wrapper validates the output is exactly in {-1,0,1}.
# DEPENDS_ON: config/constants.py
# USED_BY: src/strategies/registry.py, src/strategies/examples/* (Phase 1)
# CHANGE_NOTES(IRAC): I: alphas must be uniform & safe to combine. R: spec
#   "+1/-1/0; 0 is inactive not hold". A: ABC + strict output validation.
#   C: uniform alphas let the RL agent weight/ignore each one cleanly.
# =====================================================================
"""BaseStrategy: every alpha emits +1 / -1 / 0 (0 = inactive, not a hold)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from config import constants as C

VALID_SIGNALS = (C.ALPHA_SELL, C.ALPHA_INACTIVE, C.ALPHA_BUY)  # -1, 0, +1


class BaseStrategy(ABC):
    """Subclass and implement compute_signal(ctx) -> int in {-1, 0, +1}.

    TWO KINDS OF ALPHA (both live in slots; both are weighted per-slot by the policy):
      * DIRECTIONAL (default): +1 = buy, -1 = sell, 0 = inactive. These vote in the
        directional consensus (alpha_summary buy%/sell%/net% + signal_accuracy).
      * NON-DIRECTIONAL gate/filter (set DIRECTIONAL = False): 1 = condition TRUE
        (e.g. "the market is moving"), 0 = FALSE. A gate's 1 is NOT a buy, so it is
        EXCLUDED from the directional consensus -- otherwise it would be miscounted as
        a bullish vote. The policy still sees it in its own alpha slot + streak and
        learns its purpose (e.g. "only act when this gate is on").
    """

    #: human-readable name (override in subclass or pass to __init__)
    name: str = "unnamed_strategy"
    #: True = directional (+1/-1/0 votes in the consensus); False = a 1/0 gate (excluded).
    DIRECTIONAL: bool = True

    def __init__(self, name: str | None = None) -> None:
        if name is not None:
            self.name = name

    @abstractmethod
    def compute_signal(self, ctx) -> int:
        """Return +1 buy / -1 sell / 0 inactive given a market context `ctx`."""
        raise NotImplementedError

    def signal(self, ctx) -> int:
        """Validated public entry point. Guarantees output in {-1, 0, +1}."""
        out = int(self.compute_signal(ctx))
        if out not in VALID_SIGNALS:
            raise ValueError(
                f"{self.name}.compute_signal returned {out!r}; "
                f"must be one of {VALID_SIGNALS} (+1 buy / -1 sell / 0 inactive)."
            )
        return out

    def reset(self) -> None:
        """Optional per-episode reset hook (override if the alpha is stateful)."""
        return None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r}>"
