"""Postgres read access for the reviewer UI.

Keeps SQL out of the Streamlit page (app.py). All reads go through this module,
which uses the same DATABASE_URL as agent/audit_log.py and agent/memo_store.py
and the same connection-per-call async pattern.

Two read paths:
- get_memos:        list runs from the memos artifact store, with optional
                    filtering by data source and outcome. Powers the runs list.
- get_run_metadata: reassemble the supervisor decisions, tool calls, and
                    sub-agent confidences for one run from audit_log. Powers the
                    optional "Details" expansion (deferred / supporting for
                    Chunk 1).

Streamlit reruns the whole script on every interaction. Callers should cache
these functions (st.cache_data) so filter changes don't reopen connections on
every keystroke.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import psycopg
from psycopg.rows import dict_row


def _get_dsn() -> str:
    """Return DATABASE_URL or raise a clear error.

    Same resolution as the agent modules — one DATABASE_URL for the system.

    Raises:
        RuntimeError: if DATABASE_URL is not set in the environment.
    """
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env and set it. "
            "Remember the host port is 5433, not 5432."
        )
    return dsn


async def get_memos(
    data_sources_filter: Optional[list[str]] = None,
    outcomes_filter: Optional[list[str]] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return memo rows, newest first, optionally filtered.

    Args:
        data_sources_filter: if non-empty, keep only rows whose
            data_sources_used array OVERLAPS this list (e.g. ['yelp', 'fred']
            returns runs touching either). None/empty means no source filter.
        outcomes_filter: if non-empty, keep only rows whose outcome is in this
            list. None/empty means no outcome filter.
        limit: maximum number of rows to return.

    Returns:
        A list of dicts, one per memo row, with all memos table columns.
        memo_json is already parsed to a dict by psycopg's JSONB adapter (or
        None for non-synthesized outcomes).
    """
    where: list[str] = []
    params: list[Any] = []

    # Array overlap (&&): row matches if it shares ANY selected source.
    if data_sources_filter:
        where.append("data_sources_used && %s")
        params.append(list(data_sources_filter))

    if outcomes_filter:
        where.append("outcome = ANY(%s)")
        params.append(list(outcomes_filter))

    sql = (
        "SELECT id, query_id, user_query, memo_json, data_sources_used, "
        "       n_findings, outcome, created_at "
        "FROM memos"
    )
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at DESC LIMIT %s"
    params.append(limit)

    async with await psycopg.AsyncConnection.connect(_get_dsn()) as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()

    # query_id is a UUID; render as str so the UI can display/key on it.
    for row in rows:
        row["query_id"] = str(row["query_id"])
    return rows


async def get_run_metadata(query_id: str) -> dict[str, Any]:
    """Reassemble supporting run detail from audit_log for one run.

    Reads the run's audit events and groups the interesting ones:
    supervisor decisions, tool calls, and sub-agent confidence scores. This is
    supporting detail for the memo (the primary artifact); the UI may surface it
    in a "Details" expansion. Optional for Chunk 1 — safe to leave unused until
    Chunk 3 wires the details panel.

    Args:
        query_id: the run to summarize.

    Returns:
        {
          "supervisor_decisions": [ {payload...}, ... ],
          "tool_calls":           [ {payload...}, ... ],
          "sub_agent_confidences": [ {"actor": ..., "confidence": ...}, ... ],
        }
    """
    sql = (
        "SELECT event_type, actor, payload, created_at "
        "FROM audit_log WHERE query_id = %s ORDER BY id ASC"
    )

    async with await psycopg.AsyncConnection.connect(_get_dsn()) as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, (query_id,))
            events = await cur.fetchall()

    supervisor_decisions: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    sub_agent_confidences: list[dict[str, Any]] = []

    for event in events:
        event_type = event["event_type"]
        payload = event["payload"] or {}
        if event_type == "supervisor_decision":
            supervisor_decisions.append(payload)
        elif event_type == "tool_call":
            tool_calls.append(payload)
            # tool_call payloads may carry a per-call confidence.
            if "confidence" in payload:
                sub_agent_confidences.append(
                    {"actor": event["actor"], "confidence": payload["confidence"]}
                )

    return {
        "supervisor_decisions": supervisor_decisions,
        "tool_calls": tool_calls,
        "sub_agent_confidences": sub_agent_confidences,
    }
