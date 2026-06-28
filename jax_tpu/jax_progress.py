# =====================================================================
# WHEN 2026-06-28 | WHO Claude for Monty
# WHY  SEE the bot getting better at passing the FTMO challenge CONSISTENTLY *while it
#      trains* — not just at the end. Operator 2026-06-28. Turns each held-out eval into
#      a plain-language readout (probability of passing at the real 2.5%/4% rules + the
#      "N passes in a row" progress toward 40) and a live-updating Colab dashboard.
# WHERE jax_tpu/jax_progress.py
# HOW   format_eval() renders one eval row as a 3-line block with trend arrows + a
#       consistency bar. LiveDashboard is an on_eval callback the trainer calls every
#       eval; in Colab it redraws P(pass)/streak/breach curves, else it prints text.
# DEPENDS_ON: (stdlib); optional matplotlib + IPython for the live plot
# USED_BY: jax_tpu/jax_trainer.py (default streaming readout + on_eval), the notebook
# CHANGE_NOTES(IRAC): I: progress toward a CONSISTENT FTMO pass must be visible live.
#   R: operator "I need to see progress relative to passing the FTMO challenge
#   consistently as training is going". A: a consistency-focused per-eval readout + a
#   live dashboard fed by the trainer's eval ledger. C: you watch P(pass) and the
#   40-in-a-row streak climb in real time and know if it's actually getting consistent.
# =====================================================================
"""Live, FTMO-consistency-focused training progress: per-eval readout + Colab dashboard."""
from __future__ import annotations


def _humanize(n: int) -> str:
    n = float(n)
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.1f}{unit}"
    return str(int(n))


def _arrow(curr, prev) -> str:
    if prev is None:
        return " "
    if curr > prev + 1e-9:
        return "▲"   # ▲ improving
    if curr < prev - 1e-9:
        return "▼"   # ▼ worse
    return "·"       # · flat


def consistency_bar(streak: int, target: int, width: int = 24) -> str:
    filled = 0 if target <= 0 else min(width, int(round(width * streak / target)))
    return "█" * filled + "░" * (width - filled)   # █ / ░


def diagnose(row: dict, prev: dict | None = None) -> tuple:
    """Read the eval as a DIAGNOSIS, not a scoreboard (operator's rubric). Returns (label, advice).

      - OVER-TRADING : breach rate high          -> sizing/frequency too aggressive
      - HIDING       : breach~0 AND return~0 AND barely trades -> reward-seeking too weak
      - LEARNING     : P(pass) rising with a controlled breach rate
      - STRONG       : high P(pass) with controlled breach
      - DEVELOPING   : none of the above yet
    Thresholds are deliberately loose; tune via the seek/idle/breach knobs the advice points to."""
    br = float(row.get("eval_breach_rate", 0.0))
    ret = float(row.get("eval_mean_return", 0.0))
    pr = float(row.get("eval_pass_rate", 0.0))
    trades = row.get("eval_mean_trades", None)
    prev_pr = None if prev is None else prev.get("eval_pass_rate")
    if br > 0.30:
        return ("OVER-TRADING", "breaching too often — cut size/frequency (lower target-seek, or raise the breach cost)")
    barely = (trades is None) or (float(trades) < 0.5)
    if abs(ret) < 0.005 and pr < 0.02 and barely:
        return ("HIDING", "barely trades & flat returns — reward-seeking too weak: RAISE target_seek_weight / idle_day_penalty")
    if prev_pr is not None and pr > prev_pr + 1e-6 and br <= 0.30:
        return ("LEARNING", "P(pass) rising with controlled breach — this is what you want; keep going")
    if pr >= 0.5 and br <= 0.30:
        return ("STRONG", "high P(pass) with controlled breach")
    return ("DEVELOPING", "no clear signal yet — watch breach vs P(pass) over the next few evals")


def format_eval(row: dict, prev: dict | None = None, target: int = 40) -> str:
    """A 3-line, plain-language progress block for ONE held-out eval.

    Shows the probability of passing at the REAL FTMO rules (2.5% target / 4% trailing),
    the average held-out return, how often it blew a drawdown wall, and the streak toward
    `target` consecutive challenge passes (the consistency gate)."""
    pr = row.get("eval_pass_rate", 0.0)
    # primary consistency = WINNING DAYS in a row (the stop metric); fall back to challenge passes (single-symbol)
    won = row.get("won_day_streak")
    if won is not None:
        streak = int(won); unit = "winning days in a row"
    else:
        streak = int(row.get("consecutive_passes", 0)); unit = "passes in a row"
    best = int(row.get("best_streak_global", streak))
    breach = row.get("eval_breach_rate", 0.0)
    ret = row.get("eval_mean_return", 0.0)
    pr_arrow = _arrow(pr, None if prev is None else prev.get("eval_pass_rate"))
    # breach: lower is better, so improving (prev>curr) shows ▲ -> compare (prev, curr)
    br_arrow = " " if prev is None else _arrow(prev.get("eval_breach_rate", breach), breach)
    trades = row.get("eval_mean_trades")
    act = "" if trades is None else f"    activity {float(trades):.1f} trades/win"
    label, advice = diagnose(row, prev)
    line0 = (f" update {row.get('update', 0):>7,} · {_humanize(row.get('timesteps', 0))} steps "
             f"· {row.get('iters_per_s', '?')} it/s · ent {row.get('ent_coef', 0):.4f}")
    line1 = (f" FTMO @ 2.5%/4%:  P(pass) {pr:5.1%} {pr_arrow}    "
             f"avg return {ret:+6.2%}    breach {breach:4.1%} {br_arrow}{act}")
    line2 = (f" CONSISTENCY:  {consistency_bar(streak, target)}  "
             f"{streak:>2}/{target} {unit}   (best {best})")
    line3 = f" DIAGNOSIS:  {label} — {advice}"
    rule = " " + "─" * 72
    out = f"{rule}\n{line0}\n{line1}\n{line2}\n{line3}"
    # vs-alphas line (only when the beat baseline ran — portfolio): is the bot BEATING or following the alphas?
    if row.get("eval_beats_alphas") is not None:
        beats = bool(row["eval_beats_alphas"]); margin = float(row.get("eval_beat_margin", 0.0))
        ar = float(row.get("eval_alpha_return", 0.0))
        verb = "BEATING" if beats else "TRAILING"
        out += (f"\n VS ALPHAS:  {verb} the alphas by {margin:+.2%}   "
                f"(bot {ret:+.2%} vs follow-alphas {ar:+.2%})")
    # action mix (HOLD-collapse check) + per-symbol exposure (is it trading ALL symbols?)
    am = row.get("action_mix")
    if am:
        names = ["HOLD", "BUY", "SELL", "CLOSE"]
        out += "\n ACTIONS:  " + "  ".join(f"{n} {p:.0%}" for n, p in zip(names, am))
    sx = row.get("symbol_exposure")
    if sx:
        syms = row.get("symbols") or [f"sym{i}" for i in range(len(sx))]
        pairs = sorted(zip(syms, sx), key=lambda x: -x[1])
        conc = row.get("symbol_concentration")
        warn = "  ⚠ CONCENTRATED on one symbol" if (conc is not None and conc > 0.70 and len(sx) > 1) else ""
        out += "\n SYMBOLS:  " + "  ".join(f"{s} {p:.0%}" for s, p in pairs) + warn
    return f"{out}\n{rule}"


def heartbeat(update: int, timesteps: int, mean_reward: float, iters_per_s: float,
              best_streak: int, target: int = 40) -> str:
    """A light one-liner between evals so you can see training is alive + the best streak so far."""
    return (f"   .. update {update:>7,} · {_humanize(timesteps)} steps · "
            f"reward {mean_reward:+.5f} · {iters_per_s:.1f} it/s · "
            f"best streak {best_streak}/{target}")


class LiveDashboard:
    """on_eval callback: in Colab/Jupyter, redraws live FTMO-progress charts each eval;
    everywhere else, prints the formatted readout. Pass to train(on_eval=LiveDashboard())."""

    def __init__(self, target: int = 40, title: str = "Camillion JAX/TPU — progress to a consistent FTMO pass"):
        self.target = int(target)
        self.title = title
        self._prev = None
        self._ok_plot = None  # lazily detected

    def __call__(self, row: dict, rows: list[dict]) -> None:
        print(format_eval(row, self._prev, self.target), flush=True)
        self._prev = row
        self._draw(rows)

    def _draw(self, rows: list[dict]) -> None:
        if self._ok_plot is None:
            try:
                import matplotlib  # noqa: F401
                from IPython import get_ipython  # noqa: F401
                self._ok_plot = True
            except Exception:
                self._ok_plot = False
        if not self._ok_plot or len(rows) < 2:
            return
        try:
            import matplotlib.pyplot as plt
            from IPython.display import clear_output
            clear_output(wait=True)
            ts = [r.get("timesteps", 0) for r in rows]
            pr = [r.get("eval_pass_rate", 0.0) for r in rows]
            stk = [r.get("consecutive_passes", 0) for r in rows]
            br = [r.get("eval_breach_rate", 0.0) for r in rows]
            label, _ = diagnose(rows[-1], rows[-2] if len(rows) >= 2 else None)
            fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
            fig.suptitle(f"{self.title}    [DIAGNOSIS: {label}]")
            ax[0].plot(ts, [p * 100 for p in pr], "-o", ms=3, color="tab:green")
            ax[0].set_title("P(pass) @ 2.5%/4%  (held-out)"); ax[0].set_ylabel("%"); ax[0].set_ylim(-2, 102)
            ax[0].set_xlabel("env steps")
            ax[1].plot(ts, stk, "-o", ms=3, color="tab:blue")
            ax[1].axhline(self.target, ls="--", c="r", label=f"goal {self.target}")
            ax[1].set_title("challenge passes IN A ROW"); ax[1].set_xlabel("env steps"); ax[1].legend(loc="lower right")
            ax[2].plot(ts, [b * 100 for b in br], "-o", ms=3, color="tab:red")
            ax[2].set_title("breach rate (lower = safer)"); ax[2].set_ylabel("%"); ax[2].set_xlabel("env steps")
            for a in ax:
                a.grid(alpha=0.3)
            plt.tight_layout(); plt.show()
        except Exception as e:  # never let a plotting hiccup stop training
            print(f"   (live plot skipped: {e})")
