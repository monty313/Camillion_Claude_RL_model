# =====================================================================
# WHEN 2026-06-26 (Phase 2 JARVIS) | WHO Claude for Monty
# WHY  The bot is a PORTFOLIO trader -- one pot, the WHOLE FTMO universe at once, not a
#      single-asset trader. MarketView holds a read-only StateProvider per symbol and
#      produces the MARKET HEATMAP (buy/sell signal of every symbol) for its own cockpit
#      tab, plus the per-symbol positions + a portfolio summary JARVIS reasons over.
# WHERE src/jarvis/market_view.py
# HOW  rows() -> one heatmap row per symbol (direction, strength, buy/sell %, hottest alpha);
#      primary() -> the symbol to feature in /state; step() advances all; from_synthetic/
#      from_cache build the universe. Read-only (it only snapshots providers).
# DEPENDS_ON: src/jarvis/state_provider.py, config/asset_specs.py, config/variables.py
# USED_BY: jarvis_bridge.py (GET /heatmap, portfolio fields on /state), src/jarvis/council.py
# CHANGE_NOTES(IRAC): I: the cockpit modeled ONE symbol; the bot trades the whole FTMO book.
#   R: operator 2026-06-26 -- portfolio trader + a heatmap with its own tab. A: a per-symbol
#   provider map -> a market heatmap + portfolio view; honest that the shared-pot portfolio
#   ENV is the next build (today each symbol is its own env). C: JARVIS sees the whole market
#   and coaches the portfolio toward a consistent pass.
# =====================================================================
"""MarketView: the FTMO-universe heatmap (buy/sell per symbol) + the portfolio view (read-only)."""
from __future__ import annotations
from src.jarvis.state_provider import StateProvider


def _direction(net: float) -> str:
    return "BUY" if net > 0.12 else ("SELL" if net < -0.12 else "FLAT")


class MarketView:
    """Read-only portfolio view: one StateProvider per symbol -> the market heatmap."""

    def __init__(self, providers: dict[str, StateProvider]):
        if not providers:
            raise ValueError("MarketView needs at least one {symbol: StateProvider}")
        self.providers = dict(providers)

    # ---- construction ----
    @classmethod
    def from_synthetic(cls, symbols=None, n: int = 600):
        from config import variables as V
        syms = list(symbols or V.SYMBOLS)
        return cls({s: StateProvider.from_synthetic(n=n, symbol=s, seed=i) for i, s in enumerate(syms)})

    @classmethod
    def from_cache(cls, data_dir: str, symbols, policy=None):
        return cls({s: StateProvider.from_cache(data_dir, symbol=s, policy=policy) for s in symbols})

    @classmethod
    def wrap(cls, obj):
        """Accept a MarketView, a single StateProvider, or None -> always return a MarketView."""
        if isinstance(obj, MarketView):
            return obj
        if isinstance(obj, StateProvider):
            return cls({obj.env.symbol or "PRIMARY": obj})
        return cls.from_synthetic()

    # ---- read-only stepping ----
    def step(self):
        for p in self.providers.values():
            p.step()
        return self

    # ---- the heatmap ----
    def rows(self) -> list[dict]:
        from config import asset_specs as A
        out = []
        for s, p in self.providers.items():
            snap = p.snapshot()
            net = float(snap["net_signal"])
            dir_alphas = [a for a in snap["alphas"] if a.get("directional", True)]
            n = max(1, len(dir_alphas))
            buy = sum(1 for a in dir_alphas if a["signal"] == 1) / n
            sell = sum(1 for a in dir_alphas if a["signal"] == -1) / n
            firing = [a for a in snap["alphas"] if a["signal"] != 0]
            hottest = max(firing, key=lambda a: abs(a["signal"]) * (a["streak"] + 1), default=None)
            out.append({
                "symbol": s, "asset_class": A.asset_class(s) or "n/a",
                "net_signal": round(net, 3), "direction": _direction(net), "strength": round(abs(net), 3),
                "buy_pct": round(buy, 3), "sell_pct": round(sell, 3),
                "n_directional": snap["n_directional"],
                "hottest_alpha": hottest["name"] if hottest else None,
                "position": snap["position"]["dir"], "price": snap["position"]["price"],
            })
        out.sort(key=lambda r: r["strength"], reverse=True)   # hottest first
        return out

    # ---- the symbol to feature in /state (strongest conviction, else first) ----
    def primary(self) -> StateProvider:
        best, score = None, -1.0
        for p in self.providers.values():
            s = abs(float(p.snapshot()["net_signal"]))
            if s > score:
                best, score = p, s
        return best or next(iter(self.providers.values()))

    def universe(self) -> list[str]:
        return list(self.providers.keys())

    def positions(self) -> list[dict]:
        return [{"symbol": s, **p.snapshot()["position"]} for s, p in self.providers.items()]

    def portfolio(self) -> dict:
        rows = self.rows()
        active = [r["symbol"] for r in rows if r["direction"] != "FLAT"]
        net_lean = round(sum(r["net_signal"] for r in rows) / max(1, len(rows)), 3)
        return {
            "symbols": len(rows), "universe": [r["symbol"] for r in rows],
            "net_lean": net_lean, "active_symbols": active,
            "open_positions": sum(1 for r in rows if r["position"] != "FLAT"),
            # honest: per-symbol envs today; the true SHARED equity/DD pot is the next env build
            "shared_pot": False,
        }

    def summary(self) -> str:
        """A JARVIS-ready one-liner of the whole market."""
        rows = self.rows()
        lean = sum(r["net_signal"] for r in rows) / max(1, len(rows))
        hot = [r for r in rows if r["direction"] != "FLAT"][:5]
        head = f"MARKET HEATMAP ({len(rows)} symbols, portfolio lean {lean:+.2f}): "
        body = "; ".join(f"{r['symbol']} {r['direction']}({r['strength']:.2f})" for r in hot) or "all flat"
        return head + body
