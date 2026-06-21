# WHEN 2026-06-21 (Phase 0) | WHO Claude for Monty
# WHY  "JARVIS // FTMO QUANT COMMAND" Streamlit dashboard shell. Imports WITHOUT
#      launching Streamlit (streamlit is imported inside main()), so the smoke
#      test can import this module even when streamlit isn't installed.
# WHERE src/jarvis/app.py | HOW Phase-2 fills the top bar / clickable graph /
#      agent comms panel; this is the wired-up starting point.
# DEPENDS_ON src/jarvis/{workflow_map,agent_bus,jarvis_judge}.py
# USED_BY operator: `streamlit run src/jarvis/app.py`.
"""Jarvis dashboard shell (Phase-0; safe to import without streamlit)."""
from __future__ import annotations
from src.jarvis.workflow_map import all_modules

def main() -> None:  # pragma: no cover - requires streamlit + a running session
    import streamlit as st
    from src.jarvis.agent_bus import AgentBus
    from src.jarvis.jarvis_judge import JarvisJudge
    from src.account.account_state import AccountState
    from src.observation import builder as B

    st.set_page_config(page_title="JARVIS // FTMO QUANT COMMAND", layout="wide")
    st.title("JARVIS // FTMO QUANT COMMAND")
    st.caption("Phase 0 shell — full top bar, clickable graph and IRAC loop land in Phase 2.")

    left, right = st.columns([3, 1])
    with left:
        st.subheader("Workflow")
        for m in all_modules():
            with st.expander(f"{m['id']}  {m['title']} — {m['what']}"):
                st.write({"inputs": m["inputs"], "outputs": m["outputs"],
                          "files": m["files"], "depends_on": m["depends_on"],
                          "tests": m["tests"], "status": m["status"]})
    with right:
        st.subheader("AGENT COMMS — IRAC LOOP")
        bus = AgentBus()
        JarvisJudge().cycle(B.zeros(), AccountState(100000.0), bus)
        for msg in bus.history():
            st.write(f"**{msg.sender}** ({msg.kind}): {msg.text}")

if __name__ == "__main__":  # pragma: no cover
    main()
