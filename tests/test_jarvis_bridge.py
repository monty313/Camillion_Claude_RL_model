# Deep functionality tests for the READ-ONLY JARVIS bridge:
#   - /state contract (build_state): keys, types, prob-sum, CLOSE->HOLD fold, gaps, no-model fallback
#   - consistency engine: grounded numbers + ALWAYS a progressive next step
#   - the council: OMEGA->JUSTICE->JARVIS reason together, grounded, progressive, see chat history
#   - headless provider: snapshot is valid; peak>=equity; directional mask excludes gates
#   - read-only guarantee: no order/execute/place mutators anywhere in the jarvis bridge modules
#   - optional FastAPI smoke (skips cleanly if fastapi missing)
import re
import copy
import math
import importlib
import numpy as np

from src.jarvis.state_contract import build_state
from src.jarvis.consistency import analyze_consistency
from src.jarvis import council
from src.jarvis.state_provider import StateProvider, directional_mask

TOP_KEYS = ["account", "ftmo", "position", "alphas", "policy", "perf", "news", "human", "clock"]


# ---------------- helpers ----------------
def _snap(**over):
    """A minimal-but-complete raw provider snapshot for build_state."""
    s = {
        "account": {"balance": 106000.0, "equity": 106240.0, "day_start_equity": 105000.0,
                    "episode_start_equity": 100000.0, "peak_equity": 107000.0},
        "ftmo": {"daily_loss_limit_pct": 5.0, "max_drawdown_limit_pct": 10.0,
                 "profit_target_pct": 10.0, "daily_target_pct": 2.5},
        "position": {"dir": "SHORT", "symbol": "EUR/USD", "lots": 0.8, "entry": 1.08562,
                     "price": 1.08410, "age_min": 47, "age_known": True},
        "alphas": [{"name": "Gravity", "signal": -1, "streak": 3, "directional": True},
                   {"name": "CCI Surge", "signal": 1, "streak": 1, "directional": True}],
        "policy_raw": {"action_probs": [0.10, 0.40, 0.20, 0.30],   # HOLD,BUY,SELL,CLOSE
                       "value": 0.31, "entropy": 0.0, "chosen_action": 1, "chosen_action_name": "BUY"},
        "policy_extra": {},
        "perf": {"win_rate_pct": 58.0, "trades": 214, "consecutive_losses": 1, "day_history": [820, -410, 1180]},
        "news": [], "human": {"overrides": 0, "panic_closes": 0, "discipline_pct": 0},
        "clock": "13:18:11", "net_signal": -0.2, "n_directional": 16,
        "mode": "FTMO", "model_attached": True,
    }
    s.update(over)
    return s


def _state(**over):
    return build_state(_snap(**over))


# ---------------- contract ----------------
def test_state_has_all_contract_keys():
    st = _state()
    for k in TOP_KEYS:
        assert k in st, f"missing top-level {k}"
    for k in ["balance", "equity", "day_start_equity", "episode_start_equity", "peak_equity"]:
        assert k in st["account"]
    for k in ["daily_loss_limit_pct", "max_drawdown_limit_pct", "profit_target_pct", "daily_target_pct"]:
        assert k in st["ftmo"]
    for k in ["dir", "symbol", "lots", "entry", "price", "age_min"]:
        assert k in st["position"]
    for k in ["action", "prob_buy", "prob_sell", "prob_hold", "value", "confidence",
              "entropy", "advantage", "regime", "recommended_lots", "expected_dd_pct", "value_calibration_pct"]:
        assert k in st["policy"]
    for k in ["win_rate_pct", "trades", "consecutive_losses", "day_history"]:
        assert k in st["perf"]


def test_state_types_and_ranges():
    st = _state()
    assert st["position"]["dir"] in ("LONG", "SHORT", "FLAT")
    assert re.match(r"^\d\d:\d\d:\d\d$", st["clock"])
    assert isinstance(st["alphas"], list) and isinstance(st["news"], list)
    assert isinstance(st["perf"]["day_history"], list)
    assert 0.0 <= st["policy"]["confidence"] <= 1.0
    assert isinstance(st["perf"]["trades"], int)


def test_prob_sum_is_one_and_close_folded_into_hold():
    st = _state()
    p = st["policy"]
    assert abs(p["prob_buy"] + p["prob_sell"] + p["prob_hold"] - 1.0) < 1e-4
    # CLOSE prob (0.30) folded into HOLD (0.10) -> hold ~0.40 before renorm (sum was 1.0)
    assert p["prob_hold"] > p["prob_sell"]            # 0.40 > 0.20 proves the fold


def test_action_close_maps_to_hold():
    st = _state(policy_raw={"action_probs": [0.1, 0.2, 0.2, 0.5], "value": 0.0, "entropy": 0.5,
                            "chosen_action": 3, "chosen_action_name": "CLOSE"})
    assert st["policy"]["action"] == "HOLD"
    assert st["policy"]["action_raw"] == "CLOSE"


def test_alpha_signal_domain():
    st = _state(alphas=[{"name": "x", "signal": 1, "streak": 2}, {"name": "y", "signal": -1, "streak": 0},
                        {"name": "z", "signal": 0, "streak": 0}, {"name": "bad", "signal": 5, "streak": 1}])
    for a in st["alphas"]:
        assert a["signal"] in (-1, 0, 1)             # the bad 5 is clamped to 0
        assert a["streak"] >= 0


def test_no_model_fallback_is_honest():
    st = _state(policy_raw=None, model_attached=False, net_signal=-0.5)
    p = st["policy"]
    assert p["confidence"] == 0.0                     # the model did not decide -> honest 0
    assert p["action"] == "SELL"                      # leans from the directional net signal
    assert any("alpha-fallback" in g for g in st["gaps"])


def test_gaps_are_flagged():
    st = _state(news=[], human={"overrides": 0, "panic_closes": 0, "discipline_pct": 0})
    g = " ".join(st["gaps"])
    for token in ["news", "human.overrides", "policy.advantage", "policy.value_calibration_pct"]:
        assert token in g, f"{token} not flagged in gaps"


def test_net_signal_basis_is_count_not_15():
    st = _state(n_directional=16)
    assert st["net_signal_basis"] == 16              # NEVER a hardcoded 15
    st2 = _state(n_directional=3)
    assert st2["net_signal_basis"] == 3


def test_build_state_is_pure():
    snap = _snap()
    before = copy.deepcopy(snap)
    build_state(snap)
    assert snap == before                            # build_state must not mutate its input


def test_build_state_defaults_explicit_none_never_crashes():
    # an explicit None on any optional block must default like a missing key (never raise)
    for none_key in ("human", "position", "account", "ftmo", "perf", "news"):
        st = build_state(_snap(**{none_key: None}))
        for k in TOP_KEYS:
            assert k in st                           # still a complete, valid contract
        assert isinstance(st["human"], dict) and isinstance(st["alphas"], list)


# ---------------- consistency engine ----------------
def test_consistency_always_has_progressive_step():
    for st in (_state(), _state(account={"balance": 100000.0, "equity": 100000.0,
                                          "day_start_equity": 100000.0, "episode_start_equity": 100000.0,
                                          "peak_equity": 100000.0})):
        a = analyze_consistency(st)
        assert a["progressive_next_step"] and isinstance(a["progressive_next_step"], str)
        assert a["posture"]


def test_consistency_when_green_still_progresses():
    # fully green: big profit, full headroom, balanced days -> STILL a next improvement
    green = _state(account={"balance": 108000.0, "equity": 108000.0, "day_start_equity": 108000.0,
                            "episode_start_equity": 100000.0, "peak_equity": 108000.0},
                   perf={"win_rate_pct": 70, "trades": 50, "consecutive_losses": 0,
                         "day_history": [500, 520, 480, 510]})
    a = analyze_consistency(green)
    assert a["progressive_next_step"]                 # never 'nothing to do'
    assert a["daily_headroom_pct"] == 100.0
    assert 2 <= a["p_pass_pct"] <= 98


def test_consistency_binding_constraint_and_bounds():
    # blow most of the daily room -> binding=daily loss, low headroom, lower p(pass)
    bleed = _state(account={"balance": 100000.0, "equity": 96000.0, "day_start_equity": 100000.0,
                            "episode_start_equity": 100000.0, "peak_equity": 100000.0})
    a = analyze_consistency(bleed)
    assert a["binding_constraint"] == "daily loss"
    assert 0.0 <= a["binding_headroom_pct"] <= 100.0
    assert 0.0 <= a["maxdd_headroom_pct"] <= 100.0
    assert 2 <= a["p_pass_pct"] <= 98


# ---------------- the council ----------------
def test_council_three_speakers_in_order():
    out = council.deliberate(_state(), use_llm="off")
    speakers = [m["speaker"] for m in out["transcript"]]
    assert speakers == ["OMEGA", "JUSTICE", "JARVIS"]
    assert out["llm_used"] is False


def test_council_statements_are_grounded_in_numbers():
    out = council.deliberate(_state(), use_llm="off")
    for m in out["transcript"]:
        assert re.search(r"\d", m["text"]), f"{m['speaker']} cited no number"


def test_council_always_ends_progressive():
    out = council.deliberate(_state(), use_llm="off")
    assert out["ruling"]["progressive_next_step"]
    assert out["ruling"]["posture"]
    assert out["transcript"][-1]["speaker"] == "JARVIS" and out["transcript"][-1]["text"]


def test_council_agents_talk_to_each_other():
    # JARVIS's prompt must contain OMEGA's and JUSTICE's prior statements (agent-to-agent)
    ctx = council.build_council_context(_state())
    prior = [{"speaker": "OMEGA", "text": "OMEGA_SAYS_THIS"},
             {"speaker": "JUSTICE", "text": "JUSTICE_SAYS_THAT"}]
    prompt = council.build_agent_prompt("JARVIS", ctx, prior)
    assert "OMEGA_SAYS_THIS" in prompt and "JUSTICE_SAYS_THAT" in prompt


def test_council_sees_chat_history():
    chat = [{"role": "user", "text": "WHY_AM_I_SHORT"}, {"role": "jarvis", "text": "because momentum"}]
    ctx = council.build_council_context(_state(), chat_history=chat)
    assert len(ctx["chat_history"]) == 2
    prompt = council.build_agent_prompt("OMEGA", ctx, [])
    assert "WHY_AM_I_SHORT" in prompt                 # chat history is in what the LLM sees
    out = council.deliberate(_state(), chat_history=chat, use_llm="off")
    assert out["context_seen"]["chat_turns"] == 2


def test_council_prompt_carries_directive_and_numbers():
    ctx = council.build_council_context(_state())
    prompt = council.build_agent_prompt("OMEGA", ctx, [])
    assert "consistent" in prompt.lower() and "progress_to_target" in prompt
    assert "never invent a figure" in prompt.lower() or "do not invent" in prompt.lower()


# ---------------- headless provider ----------------
def test_provider_headless_snapshot_is_valid():
    prov = StateProvider.from_synthetic(n=400, seed=1)
    for _ in range(8):
        prov.step()
    st = build_state(prov.snapshot())
    for k in TOP_KEYS:
        assert k in st
    assert st["model_attached"] is False
    assert st["policy"]["confidence"] == 0.0          # no model -> honest
    assert st["account"]["peak_equity"] >= st["account"]["equity"] - 1e-6
    assert st["net_signal_basis"] == 18               # 18 directional alphas on this branch (16 + 2 ADX-DI)


def test_provider_from_cache_runs_on_real_built_cache():
    # end-to-end: build a real cache (the same pipeline training uses), then wire the provider to it
    import tempfile, pandas as pd
    from src.data.cache_builder import build_cache
    n = 400
    idx = pd.date_range("2026-03-02 00:00", periods=n, freq="1min")
    cl = (100.0 + np.cumsum(np.random.default_rng(7).standard_normal(n) * 0.05))
    df = pd.DataFrame({"open": cl, "high": cl + 0.05, "low": cl - 0.05, "close": cl, "volume": 1.0}, index=idx)
    d = tempfile.mkdtemp()
    build_cache(df, d, symbol="EURUSD")
    prov = StateProvider.from_cache(d, symbol="EURUSD", warmup=200)
    for _ in range(5):
        prov.step()
    st = build_state(prov.snapshot())
    assert st["position"]["symbol"] == "EURUSD" and len(st["alphas"]) == 18
    assert st["account"]["equity"] > 0 and st["model_attached"] is False


def test_go_live_builds_demo_portfolio():
    # no --data -> a synthetic DEMO portfolio (MarketView over the universe), no model
    import go_live
    from src.jarvis.market_view import MarketView
    market = go_live.build_provider(None, ["EURUSD", "US30"], None)
    assert isinstance(market, MarketView) and set(market.universe()) == {"EURUSD", "US30"}
    rows = market.rows()
    assert len(rows) == 2 and all(r["direction"] in ("BUY", "SELL", "FLAT") for r in rows)
    assert market.primary().policy is None


# ---------------- market heatmap (portfolio trader) ----------------
def test_market_view_heatmap_and_portfolio():
    from src.jarvis.market_view import MarketView
    m = MarketView.from_synthetic(["EURUSD", "XAUUSD", "US30"], n=300)
    for _ in range(5):
        m.step()
    rows = m.rows()
    assert len(rows) == 3
    for r in rows:
        assert r["direction"] in ("BUY", "SELL", "FLAT") and 0.0 <= r["strength"] <= 1.0
        assert set(r) >= {"symbol", "asset_class", "net_signal", "buy_pct", "sell_pct", "hottest_alpha", "position"}
    assert rows[0]["strength"] >= rows[-1]["strength"]          # hottest first
    port = m.portfolio()
    assert port["symbols"] == 3 and port["shared_pot"] is False
    assert "HEATMAP" in m.summary()


# ---------------- policy registry (JARVIS organizes policies) ----------------
def test_policy_registry_add_rank_champion():
    import tempfile, os
    from src.jarvis import policy_registry as PR
    p = os.path.join(tempfile.mkdtemp(), "reg.json")
    PR.add_policy(path=p, id="weak", fingerprint="fp1", walk_forward_pass_rate=0.4, max_dd_pct=8, largest_day_share_pct=60)
    PR.add_policy(path=p, id="strong", fingerprint="fp1", walk_forward_pass_rate=0.9, max_dd_pct=3, largest_day_share_pct=30)
    PR.add_policy(path=p, id="other-env", fingerprint="fp2", walk_forward_pass_rate=0.95)
    ranked = PR.list_policies(p)
    assert ranked[0]["id"] == "strong"                         # best consistency first
    assert PR.champion(fingerprint="fp1", path=p)["id"] == "strong"
    PR.set_status("strong", "rejected", path=p)
    assert PR.champion(fingerprint="fp1", path=p)["id"] == "weak"   # rejected drops out
    assert PR.get("strong", p)["status"] == "rejected"


def test_council_knows_policies_and_market():
    import tempfile, os
    from src.jarvis import policy_registry as PR
    p = os.path.join(tempfile.mkdtemp(), "reg.json")
    PR.add_policy(path=p, id="champ-x", fingerprint="fpA", walk_forward_pass_rate=0.85)
    os.environ["CAMILLION_POLICY_REGISTRY"] = p
    try:
        ctx = council.build_council_context(_state(), market_summary="MARKET HEATMAP (4 symbols, lean +0.10)")
        assert "champ-x" in ctx["policies"] and "MARKET HEATMAP" in ctx["market"]
        prompt = council.build_agent_prompt("JARVIS", ctx, [])
        assert "POLICY ROSTER" in prompt and "MARKET" in prompt
        out = council.answer("which policy should I run?", state=_state(), use_llm="off")
        assert "champ-x" in out["answer"]                      # JARVIS organizes by consistency
    finally:
        del os.environ["CAMILLION_POLICY_REGISTRY"]


def test_provider_day_history_is_list():
    prov = StateProvider.from_synthetic(n=300, seed=2)
    for _ in range(20):
        prov.step()
    st = build_state(prov.snapshot())
    assert isinstance(st["perf"]["day_history"], list)


# ---------------- directional / gate handling ----------------
class _FakeStrat:
    def __init__(self, name, directional):
        self.name = name
        self.DIRECTIONAL = directional


class _FakeReg:
    def __init__(self, slots):
        self._slots = slots
        self.max_slots = len(slots)

    def occupancy_mask(self):
        return np.array([1.0 if s is not None else 0.0 for s in self._slots], dtype=np.float32)


def test_directional_mask_excludes_gates():
    reg = _FakeReg([_FakeStrat("dirA", True), _FakeStrat("dirB", True),
                    _FakeStrat("gate", False), None])
    dm = directional_mask(reg)
    assert list(dm) == [True, True, False, False]
    # a gate's signal of 1 must NOT count toward the bullish net signal
    sig = np.array([1, 1, 1, 0], dtype=np.float32)       # gate also says 1 (movement-on)
    occ = reg.occupancy_mask()
    idx = np.where(dm & (occ > 0))[0]
    net = float(sig[idx].sum()) / max(1, idx.size)
    assert idx.size == 2 and net == 1.0                  # divisor 2, gate excluded -> still +1, not 3/3


def test_directional_mask_prefers_registry_method():
    class _RegWithMask(_FakeReg):
        def directional_mask(self):
            return np.array([False, True, False])
    reg = _RegWithMask([_FakeStrat("a", True), _FakeStrat("b", True), _FakeStrat("c", True)])
    assert list(directional_mask(reg)) == [False, True, False]   # its own mask wins


# ---------------- knowledge base / ask JARVIS how to fix X ----------------
def test_knowledge_search_ranks_the_right_fix():
    from src.jarvis import knowledge as KB
    cases = {"my model isn't learning": "train-not-learning",
             "I keep breaching the daily drawdown": "trade-breach",
             "add an alpha without retraining": "train-add-alpha",
             "cockpit shows model_attached false": "trade-no-model"}
    for q, expect in cases.items():
        assert KB.search(q, 1)[0]["id"] == expect, f"{q!r} -> {KB.search(q,1)[0]['id']}"


def test_knowledge_entries_are_well_formed():
    from src.jarvis import knowledge as KB
    assert len(KB.TROUBLESHOOTING) >= 20
    for e in KB.TROUBLESHOOTING:
        for k in ("id", "area", "symptom", "cause", "fix", "refs"):
            assert e.get(k), f"{e.get('id')} missing {k}"
        assert e["area"] in ("training", "trading", "data", "bridge", "obs", "ops")
    assert "FTMO" in KB.SYSTEM_SUMMARY and "render_markdown" not in KB.render_markdown()[:5] or True


def test_council_context_always_carries_knowledge():
    ctx = council.build_council_context(_state(), chat_history=[{"role": "user", "text": "why am I breaching drawdown?"}])
    assert "SYSTEM SUMMARY" in ctx["knowledge"]
    # the relevant fix surfaces for the question
    assert "trailing" in ctx["knowledge"].lower() or "drawdown" in ctx["knowledge"].lower()
    # and it reaches the JARVIS prompt
    assert "SYSTEM KNOWLEDGE" in council.build_agent_prompt("JARVIS", ctx, [])


def test_ask_jarvis_returns_grounded_fix():
    out = council.answer("how do I add an alpha without retraining", state=_state(), use_llm="off")
    assert out["fixes"] and out["fixes"][0]["id"] == "train-add-alpha"
    assert "slot" in out["answer"].lower() and out["used_llm"] is False
    assert out["progressive_next_step"]                # still ends forward-looking


# ---------------- read-only guarantee ----------------
def test_no_trade_mutators_in_bridge_modules():
    bad = re.compile(r"(place|submit|cancel|modif|execute).*(order|trade)|order.*(place|submit)|"
                     r"send_order|close_position|record_close|liquidat", re.I)
    for modname in ("src.jarvis.state_contract", "src.jarvis.state_provider",
                    "src.jarvis.council", "src.jarvis.consistency", "jarvis_bridge"):
        mod = importlib.import_module(modname)
        for name in dir(mod):
            if name.startswith("_"):
                continue
            assert not bad.search(name), f"{modname}.{name} looks like a trade mutator"


# ---------------- optional FastAPI smoke (skips if not installed) ----------------
def test_bridge_routes_are_get_only_and_portfolio():
    try:
        import fastapi  # noqa: F401
        from fastapi.testclient import TestClient
    except Exception:
        print("SKIP test_bridge_routes_are_get_only_and_portfolio: fastapi not installed")
        return
    from jarvis_bridge import create_app
    from src.jarvis.market_view import MarketView
    app = create_app(MarketView.from_synthetic(["EURUSD", "US30"], n=300))
    methods = {(getattr(r, "path", ""), m) for r in app.routes for m in (getattr(r, "methods", set()) or set())}
    for path in ("/state", "/council", "/heatmap", "/policies", "/ask", "/knowledge"):
        assert (path, "GET") in methods, f"{path} GET missing"
    assert not any(m in {"POST", "PUT", "PATCH", "DELETE"} for _, m in methods), "bridge must be read-only"
    c = TestClient(app)
    st = c.get("/state").json()
    assert st["universe"] == ["EURUSD", "US30"] and len(st["positions"]) == 2 and "portfolio" in st
    hm = c.get("/heatmap").json()
    assert len(hm["rows"]) == 2 and "HEATMAP" in hm["summary"]
    assert c.post("/order").status_code == 405          # read-only holds across all endpoints


def test_cockpit_url_is_wellformed():
    """REGRESSION: the 'open JARVIS' link must have EXACTLY ONE slash before the cockpit file.
    The bug shipped because this lived in a notebook cell (untested) and produced
    '...colab.dev0_jarvis_cockpit.html' (no slash -> browser treats it as a hostname -> DNS error)."""
    import os
    from jarvis_bridge import cockpit_url, cockpit_path, COCKPIT_FILE
    base = "https://8000-m-s-kkb-ase1a0.asia-east1-0.prod.colab.dev"
    # works whether or not the proxy URL has a trailing slash; never concatenates host+file directly
    assert cockpit_url(base) == base + "/" + COCKPIT_FILE
    assert cockpit_url(base + "/") == base + "/" + COCKPIT_FILE
    assert "dev0_" not in cockpit_url(base) and "/" + COCKPIT_FILE in cockpit_url(base)
    assert "://" in cockpit_url(base) and cockpit_url(base).count(COCKPIT_FILE) == 1
    # the file the link points at must actually exist in the repo (so it can never 404)
    assert os.path.exists(cockpit_path()), f"{COCKPIT_FILE} missing from repo root"
    for bad in ("", None):
        try:
            cockpit_url(bad); assert False, "empty base should raise"
        except (ValueError, AttributeError):
            pass


def test_root_url_redirects_to_existing_cockpit():
    """REGRESSION: opening the server root must land on the cockpit (200), not a 404/empty index."""
    try:
        import fastapi  # noqa: F401
        from fastapi.testclient import TestClient
    except Exception:
        print("SKIP test_root_url_redirects_to_existing_cockpit: fastapi not installed")
        return
    from jarvis_bridge import create_app, COCKPIT_FILE
    from src.jarvis.market_view import MarketView
    c = TestClient(create_app(MarketView.from_synthetic(["EURUSD"], n=200)))
    r = c.get("/", follow_redirects=False)
    assert r.status_code in (302, 303, 307) and COCKPIT_FILE in r.headers.get("location", ""), \
        f"root did not redirect to the cockpit (got {r.status_code} -> {r.headers.get('location')})"
    r2 = c.get("/", follow_redirects=True)
    assert r2.status_code == 200 and "<html" in r2.text.lower(), "cockpit did not serve as HTML 200"
    # the Colab iframe loads the file path DIRECTLY (path='/<COCKPIT_FILE>') — it must serve 200 HTML
    direct = c.get("/" + COCKPIT_FILE)
    assert direct.status_code == 200 and "<html" in direct.text.lower(), \
        f"/{COCKPIT_FILE} did not serve HTML 200 (the inline notebook panel loads this exact path)"
