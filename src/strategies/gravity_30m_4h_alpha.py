# =====================================================================
# WHEN 2026-06-21 (Phase 2) | WHO Claude for Monty (logic by Monty)
# WHY  First real alpha: "gravity" = signed distance from equilibrium on RAW
#      CCI (30,100), RAW RSI (4,14), all 8 Bollinger configs (20/200 x 0.5/1/2/4),
#      and the SMA fan. Per-timeframe majority vote; fires only if 30m AND 4h
#      agree. One alpha slot, +1/-1/0. Does NOT change the 367 contract.
# WHERE src/strategies/gravity_30m_4h_alpha.py
# HOW  Subclass BaseStrategy; read indicators via ctx.ind(col, tf). All column
#      lists are overridable class attributes. VOTE_MODE selects "flat" (every
#      detector votes -- per Monty's spec; BB-heavy) or "family" (CCI/RSI/BB/SMA
#      each cast ONE vote -- balanced).
# DEPENDS_ON: src/strategies/base.py, src/strategies/context.py
# USED_BY: alpha registry via register_gravity_alpha().
# CHANGE_NOTES(IRAC): I: Monty finalized RAW-only CCI/RSI + all BB variants.
#   R: gravity from equilibrium, 30m & 4h confluence. A: 13 detectors/TF (2 CCI,
#   2 RSI, 8 BB, 1 SMA-fan); flat or family voting. C: a tunable confluence alpha.
# =====================================================================
"""Gravity alpha: raw CCI/RSI + all BB configs + SMA fan; per-TF majority; 30m & 4h must agree."""
from __future__ import annotations
import math
from src.strategies.base import BaseStrategy


def _majority(votes):
    up = sum(1 for v in votes if v > 0)
    dn = sum(1 for v in votes if v < 0)
    return 1 if up > dn else (-1 if dn > up else 0)


class Gravity30m4hAlpha(BaseStrategy):
    name = "gravity_30m_4h_agree"

    # ---- dead zones (no signal inside these) ----
    CCI_DEAD_LOW, CCI_DEAD_HIGH = -25.0, 25.0
    RSI_DEAD_LOW, RSI_DEAD_HIGH = 45.0, 55.0
    BB_DEAD_LOW, BB_DEAD_HIGH = -0.25, 0.25
    SMA_DEAD = None        # no SMA-fan deadband for this alpha -> vote purely by sign

    # ---- detector columns (RAW only for CCI/RSI), all overridable ----
    CCI_COLS = ("cci30_raw", "cci100_raw")
    RSI_COLS = ("rsi4_raw", "rsi14_raw")
    BB_CONFIGS = ("bb20_dev0.5", "bb20_dev1.0", "bb20_dev2.0", "bb20_dev4.0",
                  "bb200_dev0.5", "bb200_dev1.0", "bb200_dev2.0", "bb200_dev4.0")
    SMA_FAN = ("sma_p1_s0", "sma_p2_s1", "sma_p3_s2", "sma_p4_s3")
    CLOSE_COL = "sma_p1_s0"          # SMA(1) = that TF's last-closed close
    TIMEFRAMES = ("30m", "4h")

    # "family" (DEFAULT) -> CCI/RSI/BB/SMA each cast ONE vote (balanced; no single
    #   family can outvote the other three). "flat" -> every detector votes (BB
    #   casts 8 of 13; kept as an override for ablation/debug).
    VOTE_MODE = "family"

    def __init__(self, name: str | None = None, vote_mode: str | None = None):
        super().__init__(name or self.name)
        if vote_mode:
            self.VOTE_MODE = vote_mode

    # final: 30m and 4h must agree on the same non-zero direction
    def compute_signal(self, ctx) -> int:
        a = self._tf_vote(ctx, "30m")
        b = self._tf_vote(ctx, "4h")
        return a if (a != 0 and a == b) else 0

    def _tf_vote(self, ctx, tf: str) -> int:
        cci = [self._cci(ctx, c, tf) for c in self.CCI_COLS]
        rsi = [self._rsi(ctx, c, tf) for c in self.RSI_COLS]
        bb = [self._bb(ctx, cfg, tf) for cfg in self.BB_CONFIGS]
        sma = self._sma_fan(ctx, tf)
        if self.VOTE_MODE == "family":
            return _majority([_majority(cci), _majority(rsi), _majority(bb), sma])
        return _majority(cci + rsi + bb + [sma])          # flat: every detector

    def debug_votes(self, ctx) -> dict:
        """Per-TF family votes + BOTH flat/family results, for side-by-side debugging."""
        out = {}
        for tf in self.TIMEFRAMES:
            cci = [self._cci(ctx, c, tf) for c in self.CCI_COLS]
            rsi = [self._rsi(ctx, c, tf) for c in self.RSI_COLS]
            bb = [self._bb(ctx, cfg, tf) for cfg in self.BB_CONFIGS]
            sma = self._sma_fan(ctx, tf)
            out[tf] = dict(cci=cci, rsi=rsi, bb=bb, sma=sma,
                           flat=_majority(cci + rsi + bb + [sma]),
                           family=_majority([_majority(cci), _majority(rsi), _majority(bb), sma]))
        return out

    def _v(self, ctx, col, tf):
        x = ctx.ind(col, tf)
        return None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)

    def _cci(self, ctx, col, tf):
        c = self._v(ctx, col, tf)
        if c is None: return 0
        return 1 if c > self.CCI_DEAD_HIGH else (-1 if c < self.CCI_DEAD_LOW else 0)

    def _rsi(self, ctx, col, tf):
        r = self._v(ctx, col, tf)
        if r is None: return 0
        return 1 if r > self.RSI_DEAD_HIGH else (-1 if r < self.RSI_DEAD_LOW else 0)

    def _bb(self, ctx, cfg, tf):
        p = self._v(ctx, self.CLOSE_COL, tf)
        u = self._v(ctx, f"{cfg}_upper", tf)
        m = self._v(ctx, f"{cfg}_middle", tf)
        l = self._v(ctx, f"{cfg}_lower", tf)
        if None in (p, u, m, l): return 0
        half = (u - l) / 2.0                               # rel: 0 at middle, +-1 at bands
        if half <= 0: return 0
        rel = (p - m) / half
        return 1 if rel > self.BB_DEAD_HIGH else (-1 if rel < self.BB_DEAD_LOW else 0)

    def _sma_fan(self, ctx, tf):
        vals = [self._v(ctx, c, tf) for c in self.SMA_FAN]
        if any(v is None for v in vals): return 0
        s0, s1, s2, s3 = vals
        past = (s1 + s2 + s3) / 3.0
        if past == 0: return 0
        rel = (s0 - past) / abs(past)
        if self.SMA_DEAD is not None and -self.SMA_DEAD <= rel <= self.SMA_DEAD: return 0
        return 1 if rel > 0 else (-1 if rel < 0 else 0)


def register_gravity_alpha(alpha_registry, name: str | None = None, vote_mode: str | None = None) -> int:
    """Register one Gravity30m4hAlpha into the next free slot. Returns the slot id."""
    return alpha_registry.register(Gravity30m4hAlpha(name=name, vote_mode=vote_mode))
