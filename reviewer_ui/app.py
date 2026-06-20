"""MarketPulse Reviewer — Streamlit UI (Chunk 1: read-only memo browser).

One page: a filterable list of past runs, each expandable to the full memo
(executive summary, findings with citations, data sources, caveats,
confidence). Reads straight from Postgres via reviewer_ui/db.py — no JSON file
dependency, no query submission, no HITL controls (those land in Chunks 2/3).

Run from the project root:
    uv run streamlit run reviewer_ui/app.py
"""

from __future__ import annotations

import asyncio
import datetime
from typing import Any, Optional
from dotenv import load_dotenv
import streamlit as st
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()
# Streamlit may launch this file as a top-level script (no package context), so
# support both "python -m" style and direct-script imports of the db module.
try:
    from reviewer_ui import db
except ImportError:  # running as a bare script: reviewer_ui isn't on sys.path
    import db  # type: ignore[no-redef]

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


def main() -> None:
    st.set_page_config(page_title="MarketPulse Reviewer", layout="wide")

    st.title("MarketPulse Reviewer")
    st.markdown(
        "Browse memos produced by the agent. Click any row to expand the full "
        "memo with findings, citations, and caveats."
    )

    # --- Filters (top bar) ---
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

    # --- Runs list ---
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
        st.info("No memos yet. Run a query through the agent CLI to populate this list.")
        return

    st.caption(f"{len(rows)} run(s)")
    for row in rows:
        _render_run(row)


# Streamlit runs this file as a script on every interaction (and re-runs it on
# each widget change), so call main() unconditionally at module top level.
main()
