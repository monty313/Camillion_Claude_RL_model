# =====================================================================
# WHEN 2026-06-26 (Phase 2 JARVIS) | WHO Claude for Monty
# WHY  A simple, persistent REGISTRY of trained policies so Monty can EASILY add a
#      policy and JARVIS can ORGANIZE them -- he knows each policy's detail RELATIVE
#      TO PASSING THE FTMO CHALLENGE CONSISTENTLY (walk-forward pass-rate, average/
#      worst day, max drawdown, day-to-day concentration) and can recommend which to
#      run. The bot is a PORTFOLIO trader (one pot, all FTMO symbols), so a policy is
#      tagged with its symbol universe + the env fingerprint that makes it comparable.
# WHERE src/jarvis/policy_registry.py
# HOW  Plain JSON at records/policy_registry.json (override with CAMILLION_POLICY_REGISTRY
#      or a path=). add_policy/list/champion/set_status/summary -- stdlib only. A tiny CLI
#      (`python -m src.jarvis.policy_registry add ...`) makes adding one a one-liner.
# DEPENDS_ON: (stdlib only)
# USED_BY: src/jarvis/council.py (JARVIS knows the roster), jarvis_bridge.py (/policies),
#          src/jarvis/knowledge.py (organization fixes)
# CHANGE_NOTES(IRAC): I: operator wants to add policies easily and have JARVIS organize
#   them by consistency. R: that request, 2026-06-26. A: a persistent registry keyed by
#   fingerprint, scored by a consistency metric, with a champion selector; JARVIS reads it
#   every deliberation. C: one clear, ranked view of which policy passes most consistently
#   -> faster, safer choice of what to run.
# =====================================================================
"""policy_registry: add/organize trained policies, ranked by how CONSISTENTLY they pass FTMO."""
from __future__ import annotations
import json
import os

def _resolve(path: str | None) -> str:
    # resolved at CALL time so CAMILLION_POLICY_REGISTRY (and tests) can redirect it.
    return path or os.environ.get("CAMILLION_POLICY_REGISTRY") or "records/policy_registry.json"


# A policy entry (all optional except id). Metrics are RELATIVE TO PASSING CONSISTENTLY.
_FIELDS = ("id", "name", "path", "fingerprint", "universe", "trainer",
           "walk_forward_pass_rate", "avg_daily_pct", "worst_day_pct", "max_dd_pct",
           "largest_day_share_pct", "days_tested", "status", "notes", "created")
_STATUSES = ("champion", "candidate", "rejected")


def _load(path: str | None = None) -> list[dict]:
    p = _resolve(path)
    if not os.path.exists(p):
        return []
    try:
        with open(p) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save(entries: list[dict], path: str | None = None) -> None:
    p = _resolve(path)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w") as f:
        json.dump(entries, f, indent=2)


def consistency_score(e: dict) -> int:
    """0..100 'how CONSISTENTLY does this policy pass' — blends pass-rate, low drawdown, and
    low day-to-day concentration (a steady build beats one lucky day). Higher is better."""
    pr = float(e.get("walk_forward_pass_rate") or 0.0)           # 0..1
    share = float(e.get("largest_day_share_pct") or 33.0)         # %
    dd = float(e.get("max_dd_pct") or 10.0)                       # %
    score = pr * 70.0 - max(0.0, share - 33.0) * 0.5 - max(0.0, dd - 4.0) * 2.0 + 30.0
    return int(max(0, min(100, round(score))))


def add_policy(path: str | None = None, **fields) -> dict:
    """Register (or update, by id) a policy. EASILY add one with just an id + path + pass-rate."""
    import datetime  # local: a real timestamp only when actually adding
    entries = _load(path)
    pid = str(fields.get("id") or fields.get("name") or f"policy-{len(entries) + 1}")
    entry = {k: fields.get(k) for k in _FIELDS}
    entry["id"] = pid
    entry["status"] = fields.get("status") if fields.get("status") in _STATUSES else "candidate"
    if not entry.get("created"):
        try:
            entry["created"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        except Exception:
            entry["created"] = "n/a"
    entry["consistency_score"] = consistency_score(entry)
    entries = [e for e in entries if e.get("id") != pid] + [entry]
    _save(entries, path)
    return entry


def list_policies(path: str | None = None) -> list[dict]:
    """All policies, best-CONSISTENCY first (rejected sink to the bottom)."""
    entries = _load(path)
    for e in entries:
        e["consistency_score"] = consistency_score(e)
    return sorted(entries, key=lambda e: (e.get("status") != "rejected", e["consistency_score"],
                                          float(e.get("walk_forward_pass_rate") or 0.0)), reverse=True)


def get(pid: str, path: str | None = None) -> dict | None:
    for e in _load(path):
        if e.get("id") == pid:
            e["consistency_score"] = consistency_score(e)
            return e
    return None


def set_status(pid: str, status: str, path: str | None = None) -> dict | None:
    if status not in _STATUSES:
        raise ValueError(f"status must be one of {_STATUSES}")
    entries = _load(path)
    hit = None
    for e in entries:
        if e.get("id") == pid:
            e["status"] = status
            hit = e
    if hit is not None:
        _save(entries, path)
    return hit


def remove(pid: str, path: str | None = None) -> bool:
    entries = _load(path)
    kept = [e for e in entries if e.get("id") != pid]
    if len(kept) != len(entries):
        _save(kept, path)
        return True
    return False


def champion(fingerprint: str | None = None, path: str | None = None) -> dict | None:
    """The policy to RUN: highest consistency among non-rejected entries; if a fingerprint is
    given, only policies trained on that SAME environment are comparable."""
    pool = [e for e in list_policies(path) if e.get("status") != "rejected"]
    if fingerprint:
        pool = [e for e in pool if e.get("fingerprint") == fingerprint]
    return pool[0] if pool else None


def summary(path: str | None = None, k: int = 6) -> str:
    """A compact, JARVIS-ready roster (best first), each line = consistency-relevant detail."""
    entries = list_policies(path)
    if not entries:
        return "POLICIES: none registered yet. Add one with policy_registry.add_policy(id=..., path=..., walk_forward_pass_rate=...)."
    champ = champion(path=path)
    lines = [f"POLICIES ({len(entries)} registered; champion = {champ['id'] if champ else 'n/a'}):"]
    for e in entries[:k]:
        pr = e.get("walk_forward_pass_rate")
        lines.append(
            f"  [{e.get('status', '?')}] {e['id']}: consistency {e['consistency_score']}/100, "
            f"pass-rate {round(float(pr) * 100) if pr is not None else '?'}%, "
            f"max-DD {e.get('max_dd_pct', '?')}%, largest-day {e.get('largest_day_share_pct', '?')}%, "
            f"universe {e.get('universe') or 'n/a'}, fp {str(e.get('fingerprint') or 'n/a')[:8]}")
    return "\n".join(lines)


def _cli():  # pragma: no cover - convenience entry point
    import argparse
    ap = argparse.ArgumentParser(description="Organize trained policies by FTMO consistency.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add"); a.add_argument("--id", required=True); a.add_argument("--path", default=None)
    a.add_argument("--fingerprint", default=None); a.add_argument("--universe", default=None)
    a.add_argument("--pass-rate", type=float, dest="walk_forward_pass_rate", default=None)
    a.add_argument("--max-dd", type=float, dest="max_dd_pct", default=None)
    a.add_argument("--largest-day", type=float, dest="largest_day_share_pct", default=None)
    a.add_argument("--status", default="candidate"); a.add_argument("--notes", default=None)
    sub.add_parser("list")
    s = sub.add_parser("status"); s.add_argument("--id", required=True); s.add_argument("--to", required=True)
    args = ap.parse_args()
    if args.cmd == "add":
        print(json.dumps(add_policy(**{k: v for k, v in vars(args).items() if k != "cmd" and v is not None}), indent=2))
    elif args.cmd == "list":
        print(summary())
    elif args.cmd == "status":
        print(set_status(args.id, args.to))


if __name__ == "__main__":  # pragma: no cover
    _cli()
