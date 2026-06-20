"""Persistence for synthesized memos — the artifact store.

Where audit_log.py records the *events* of a run (a hash-chained, append-only
stream), this module persists the *output*: the full structured Memo, one row
per run outcome, in the ``memos`` table (migration 004).

Design mirrors audit_log.py deliberately:
- Same DATABASE_URL connection, resolved the same way.
- Connection-per-call, no pool. Memo writes are low-frequency (once per run),
  so the overhead of a pool isn't justified yet.
- Writes are best-effort: a memo persistence failure is logged but never
  raised, exactly like audit_log.write_event. Losing the artifact row must not
  crash the agent mid-run.

The memos table CHECK constraint ties memo_json's presence to the outcome:
'synthesized' rows carry a memo; 'rejected'/'error' rows carry NULL. This
module enforces the same rule client-side before insert (defense in depth).
"""

from __future__ import annotations

import datetime
import os
import uuid
from typing import Literal, Optional

import psycopg
import structlog
from psycopg.types.json import Jsonb

from .memo import Memo

log = structlog.get_logger(__name__)

# Run outcomes recognized by the memos table. Mirrors the DB CHECK constraint
# in 004_memos.sql.
ALLOWED_OUTCOMES = frozenset({"synthesized", "rejected", "error"})

MemoOutcome = Literal["synthesized", "rejected", "error"]


def _get_dsn() -> str:
    """Return DATABASE_URL or raise a clear error.

    Same resolution as audit_log._get_dsn — both read the one DATABASE_URL.

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


async def write_memo(
    query_id: str | uuid.UUID,
    user_query: str,
    memo: Optional[Memo],
    outcome: MemoOutcome,
) -> Optional[int]:
    """Persist a run outcome to the memos table, returning the inserted row id.

    Args:
        query_id: The run's query_id (links to audit_log.query_id).
        user_query: The user's original question (denormalized for display).
        memo: The synthesized Memo for 'synthesized' outcomes; None otherwise.
            This function serializes the Memo — pass the object, not JSON.
        outcome: 'synthesized', 'rejected', or 'error'.

    Returns:
        The inserted row's id, or None if the write was skipped/failed. Like
        audit_log writes, DB errors are caught and logged rather than raised:
        a failure here must not crash the agent.
    """
    # Client-side mirror of the DB CHECK: synthesized <=> memo present.
    if outcome not in ALLOWED_OUTCOMES:
        log.warning(
            "write_memo_invalid_outcome",
            query_id=str(query_id),
            outcome=outcome,
        )
        return None
    if outcome == "synthesized" and memo is None:
        log.warning("write_memo_synthesized_without_memo", query_id=str(query_id))
        return None
    if outcome != "synthesized" and memo is not None:
        log.warning("write_memo_nonsynthesized_with_memo", query_id=str(query_id))
        return None

    # Derive the denormalized columns from the memo (or empty/zero defaults).
    if memo is not None:
        memo_payload: Optional[Jsonb] = Jsonb(memo.model_dump())
        data_sources_used = list(memo.data_sources_used)
        n_findings = len(memo.findings)
    else:
        memo_payload = None
        data_sources_used = []
        n_findings = 0

    created_at = datetime.datetime.now(datetime.timezone.utc)

    try:
        # Connection-per-call. The `async with` commits on success, rolls back
        # on exception, and closes the connection.
        async with await psycopg.AsyncConnection.connect(_get_dsn()) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO memos
                        (query_id, user_query, memo_json,
                         data_sources_used, n_findings, outcome, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        str(query_id),
                        user_query,
                        memo_payload,
                        data_sources_used,
                        n_findings,
                        outcome,
                        created_at,
                    ),
                )
                inserted = await cur.fetchone()
                row_id = inserted[0]
    except Exception as exc:
        # Best-effort: never let a memo persistence failure crash the agent.
        log.warning(
            "memo_write_failed",
            query_id=str(query_id),
            outcome=outcome,
            error=str(exc),
        )
        return None

    log.info(
        "memo_written",
        id=row_id,
        query_id=str(query_id),
        outcome=outcome,
        n_findings=n_findings,
    )
    return row_id
