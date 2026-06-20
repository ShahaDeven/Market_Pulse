"""MarketPulse Reviewer — Streamlit UI.

Chunk 1 gave us a read-only memo browser. Chunk 2 adds the "active run" path:
submit a query, watch its LangGraph events stream in, answer a human-in-the-loop
(HITL) pause, and read the final memo — all in the browser.

The page is a four-phase state machine (see reviewer_ui/session.py):
    idle       query box + past-runs browser (the original Chunk 1 view)
    running    live event timeline while the agent thread streams
    paused     HITL panel: Approve / Reject / Retry the low-confidence work
    completed  the final memo (or a rejection / error notice)

Execution model: the agent runs on a background thread (reviewer_ui/
agent_runner.py) and reports events through a queue this script drains on a
3-second auto-refresh. MemorySaver checkpointer is in-process, so a paused run
lives only as long as this Streamlit server.

Run from the project root:
    uv run streamlit run reviewer_ui/app.py
"""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import asyncio
import datetime
import json
import time
from typing import Any, Optional
import streamlit as st
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# streamlit-autorefresh drives the live polling. Degrade gracefully if it isn't
# installed yet (see README): the active phases fall back to a manual Refresh
# button instead of crashing the whole page.
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:  # pragma: no cover - optional dependency
    st_autorefresh = None  # type: ignore[assignment]

# Streamlit may launch this file as a top-level script (no package context), so
# support both "python -m" style and direct-script imports of sibling modules.
try:
    from reviewer_ui import db, session
except ImportError:  # running as a bare script: reviewer_ui isn't on sys.path
    import db  # type: ignore[no-redef]
    import session  # type: ignore[no-redef]

# All known sub-agents and outcomes — drive the filter widgets. 'error' is a
# valid outcome but rare; we still let reviewers filter for it.
DATA_SOURCE_OPTIONS = ["yelp", "sec", "fred"]
OUTCOME_OPTIONS = ["synthesized", "rejected", "error"]

# Badge glyph per outcome for the run title row.
OUTCOME_BADGE = {
    "synthesized": "🟢 synthesized",
    "rejected": "🔴 rejected",
    "error": "⚠️ error",
}

# Icon per node for the live event timeline.
_NODE_ICON = {
    "supervisor": "🔍",
    "yelp_agent": "🛠️",
    "sec_agent": "🛠️",
    "fred_agent": "🛠️",
    "check_confidence": "🧮",
    "synthesize": "📝",
    "hitl": "⏸",
}

# How often (ms) the running phase polls the agent thread for new events.
REFRESH_INTERVAL_MS = 3000


# ============================================================================
# Chunk 1 helpers — read-only memo browser (unchanged behaviour)
# ============================================================================


@st.cache_data(ttl=30)
def load_memos(
    data_sources_filter: Optional[tuple[str, ...]],
    outcomes_filter: Optional[tuple[str, ...]],
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Cached wrapper over db.get_memos.

    The 30s TTL keeps filter changes from hammering Postgres while still letting
    a freshly completed run appear within 30 seconds. Filter args are tuples
    (hashable) so st.cache_data can key on them.
    """
    return asyncio.run(
        db.get_memos(
            data_sources_filter=list(data_sources_filter) if data_sources_filter else None,
            outcomes_filter=list(outcomes_filter) if outcomes_filter else None,
            limit=limit,
        )
    )


def _relative_time(when: datetime.datetime) -> str:
    """Human-friendly 'time ago' string for a timestamptz value."""
    now = datetime.datetime.now(datetime.timezone.utc)
    # Postgres timestamptz comes back tz-aware; guard just in case.
    if when.tzinfo is None:
        when = when.replace(tzinfo=datetime.timezone.utc)
    delta = now - when
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    return f"{days}d ago"


def _render_memo_body(memo: dict[str, Any]) -> None:
    """Render the parsed memo_json dict inside an expander."""
    st.markdown("#### Executive summary")
    st.markdown(memo.get("executive_summary", "_(none)_"))

    st.markdown("#### Key findings")
    findings = memo.get("findings") or []
    if findings:
        for i, finding in enumerate(findings, 1):
            st.markdown(f"**{i}. {finding.get('headline', '')}**")
            st.markdown(finding.get("detail", ""))
            citation = finding.get("citation")
            if citation:
                st.caption(f"Citation: `{citation}`")
    else:
        st.markdown("_No findings recorded._")

    st.markdown("#### Data sources")
    sources = memo.get("data_sources_used") or []
    st.markdown(", ".join(sources) if sources else "_None._")

    st.markdown("#### Caveats")
    caveats = memo.get("caveats") or []
    if caveats:
        for caveat in caveats:
            st.markdown(f"- {caveat}")
    else:
        st.markdown("_None noted._")

    st.markdown("#### Confidence")
    st.markdown(memo.get("confidence_summary", "_(none)_"))


def _render_run(row: dict[str, Any]) -> None:
    """Render one memo row as an expandable card."""
    user_query = row.get("user_query") or "(no query)"
    outcome = row.get("outcome", "")
    badge = OUTCOME_BADGE.get(outcome, outcome)
    n_findings = row.get("n_findings", 0)
    created_at = row.get("created_at")

    title_query = user_query[:100] + ("…" if len(user_query) > 100 else "")
    rel = _relative_time(created_at) if isinstance(created_at, datetime.datetime) else ""
    label = f"{title_query}  ·  {badge}  ·  {n_findings} findings  ·  {rel}"

    with st.expander(label):
        memo = row.get("memo_json")
        if memo:
            _render_memo_body(memo)
        else:
            # rejected / error outcomes carry no memo.
            st.info(f"No memo produced for this run (outcome: {outcome}).")

        st.markdown("---")
        st.caption(f"query_id: `{row.get('query_id')}`")
        if isinstance(created_at, datetime.datetime):
            st.caption(f"created_at: {created_at.isoformat()}")


def _render_past_runs() -> None:
    """Filters + chronological list of past memos (the Chunk 1 browser)."""
    col_sources, col_outcomes = st.columns(2)
    with col_sources:
        data_sources_filter = st.multiselect(
            "Data sources",
            options=DATA_SOURCE_OPTIONS,
            default=[],
            help="Show runs that used any of the selected sub-agents.",
        )
    with col_outcomes:
        outcomes_filter = st.multiselect(
            "Outcome",
            options=OUTCOME_OPTIONS,
            default=[],
            help="Filter by how the run ended.",
        )

    try:
        rows = load_memos(
            data_sources_filter=tuple(data_sources_filter) or None,
            outcomes_filter=tuple(outcomes_filter) or None,
        )
    except Exception as exc:  # surface DB/config errors instead of a stack trace
        st.error(
            "Could not load memos from Postgres. Check that the database is "
            "running, migrations (including 004_memos.sql) are applied, and "
            f"DATABASE_URL is set.\n\nDetails: {exc}"
        )
        return

    if not rows:
        st.info("No memos yet. Run a query above to populate this list.")
        return

    st.caption(f"{len(rows)} run(s)")
    for row in rows:
        _render_run(row)


# ============================================================================
# Event formatting (live timeline)
# ============================================================================


def format_event(event: dict) -> str:
    """One-line display string for a streamed event.

    Events arrive as {node_name: state_update} dicts (LangGraph updates mode
    yields one node per event). We pull the most informative bit from the
    node's delta — matching the same state shape agent/cli.py formats.
    """
    for node, update in event.items():
        if not isinstance(update, dict):
            continue
        return _format_node(node, update)
    return ""


def _format_node(node: str, update: dict) -> str:
    icon = _NODE_ICON.get(node, "•")

    if node == "supervisor":
        decisions = update.get("supervisor_log") or []
        if decisions:
            return f"{icon} supervisor → {decisions[-1]}"
        target = update.get("target_agent")
        return f"{icon} supervisor → route to {target}" if target else f"{icon} supervisor → deciding"

    if node in ("yelp_agent", "sec_agent", "fred_agent"):
        agent = node[:-6]  # strip the "_agent" suffix → yelp|sec|fred
        n_calls = len(update.get("tool_calls_made") or [])
        confidence = (update.get(f"{agent}_data") or {}).get("confidence")
        if confidence is not None:
            return f"✅ {node} → {n_calls} tool call(s), confidence {confidence:.2f}"
        return f"{icon} {node} → {n_calls} tool call(s)"

    if node == "check_confidence":
        if update.get("hitl_pending"):
            return f"{icon} check_confidence → HITL triggered"
        return f"{icon} check_confidence → confidence OK"

    if node == "synthesize":
        if update.get("final_memo"):
            return f"{icon} synthesize → memo complete"
        return f"{icon} synthesize → synthesizing"

    if node == "hitl":
        return f"{icon} hitl → awaiting review"

    return f"{icon} {node}"


def _render_event_log() -> None:
    """Render the accumulated events as a vertical timeline."""
    events = st.session_state.events
    if not events:
        st.caption("Waiting for the first event…")
        return
    for event in events:
        line = format_event(event)
        if line:
            st.markdown(line)


# ============================================================================
# Phase views
# ============================================================================


def render_idle() -> None:
    """idle: query box + the past-runs browser."""
    query = st.text_input(
        "Query",
        placeholder="e.g. Find coffee shops in Philadelphia",
        key="query_input",
    )
    if st.button("Run Query", type="primary"):
        if query and query.strip():
            session.start_run(query.strip())
            st.rerun()
        else:
            st.warning("Enter a query first.")

    st.markdown("---")
    st.subheader("Past runs")
    _render_past_runs()


def render_running() -> None:
    """running: live event timeline while the agent thread streams."""
    query = st.session_state.query
    st.subheader(f"Running: {query}")

    started_at = st.session_state.started_at or time.time()
    elapsed = int(time.time() - started_at)
    st.caption(f"⏳ Started {elapsed}s ago · {len(st.session_state.events)} event(s)")

    _render_event_log()


def render_paused() -> None:
    """paused: HITL panel — Approve / Reject / Retry the low-confidence work."""
    payload = st.session_state.hitl_payload or {}

    st.warning("⏸ Paused for human review")
    st.markdown(f"**Query:** {payload.get('query', '')}")

    low_conf = payload.get("low_confidence_agents") or []
    if low_conf:
        st.markdown("**Low-confidence sub-agent(s):**")
        for agent in low_conf:
            conf = agent.get("confidence")
            conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "?"
            st.markdown(f"- **{agent.get('agent')}** · confidence {conf_str}")
            if agent.get("reason"):
                st.caption(agent["reason"])

    data_gathered = payload.get("data_gathered") or {}
    preview = json.dumps(data_gathered, default=str)
    if len(preview) > 500:
        preview = preview[:500] + " … (truncated)"
    st.markdown("**Data preview:**")
    st.code(preview, language="json")

    # Decision buttons. Each click reruns Streamlit; for decisions we also
    # st.rerun() after sending so the running view renders immediately.
    col_a, col_r, col_t = st.columns(3)
    if col_a.button("Approve", type="primary", use_container_width=True):
        session.submit_hitl_decision("approve", "")
        st.rerun()
    if col_r.button("Reject", use_container_width=True):
        session.submit_hitl_decision("reject", "")
        st.rerun()
    if col_t.button("Retry with hint", use_container_width=True):
        st.session_state.show_retry_input = not st.session_state.show_retry_input

    if st.session_state.show_retry_input:
        hint = st.text_input("Optional hint", key="retry_hint_input")
        if st.button("Submit retry"):
            session.submit_hitl_decision("retry", hint or "")
            st.rerun()

    st.markdown("---")
    st.markdown("**Event log so far**")
    _render_event_log()


def render_completed() -> None:
    """completed: the final memo, or a rejection / error notice."""
    query = st.session_state.query
    outcome = st.session_state.run_outcome
    st.subheader(f"Completed: {query}")

    if outcome == "synthesized" and st.session_state.final_memo_json:
        try:
            memo = json.loads(st.session_state.final_memo_json)
            _render_memo_body(memo)
        except json.JSONDecodeError:
            st.error("A memo was produced but could not be parsed as JSON.")
            st.code(st.session_state.final_memo_json)
    elif outcome == "rejected":
        st.warning("🔴 Run rejected during human review. No memo was produced.")
        st.caption(f"{len(st.session_state.events)} event(s) before rejection.")
    elif outcome == "error":
        st.error(f"Run failed: {st.session_state.run_error}")
    else:  # incomplete
        st.info("Run ended without producing a memo (incomplete).")

    st.markdown("---")
    with st.expander("Event log"):
        _render_event_log()

    if st.button("Start new query", type="primary"):
        # Drop the memo cache so the just-persisted run shows in the idle list.
        load_memos.clear()
        session.reset_session_state()
        st.rerun()


# ============================================================================
# Entry point
# ============================================================================


def main() -> None:
    st.set_page_config(page_title="MarketPulse Reviewer", layout="wide")
    session.init_session_state()

    st.title("MarketPulse Reviewer")
    st.markdown(
        "Submit a query to watch the agent run live, or browse past memos below."
    )

    # Drain the agent thread's queue for the active phases BEFORE rendering, so
    # we render the freshest events and any phase transition this tick.
    if st.session_state.phase in ("running", "paused"):
        session.poll_runner()

    # Live refresh only while the agent thread is actively streaming. The paused
    # phase deliberately does NOT auto-refresh: the agent is blocked waiting for
    # the reviewer, so nothing new arrives until a button is clicked — and
    # refreshing would steal focus from the retry-hint box.
    if st.session_state.phase == "running":
        if st_autorefresh is not None:
            st_autorefresh(interval=REFRESH_INTERVAL_MS, key="refresh_running")
        else:
            st.button("Refresh", help="Install streamlit-autorefresh for live updates.")

    phase = st.session_state.phase
    if phase == "idle":
        render_idle()
    elif phase == "running":
        render_running()
    elif phase == "paused":
        render_paused()
    elif phase == "completed":
        render_completed()


# Streamlit runs this file as a script on every interaction (and re-runs it on
# each widget change), so call main() unconditionally at module top level.
main()
