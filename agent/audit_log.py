"""Append-only, hash-chained audit log for MarketPulse agent runs.
...
"""

from __future__ import annotations
import datetime
import hashlib
import json
import os
import uuid
from typing import Any
import psycopg
import structlog
from psycopg.types.json import Jsonb

log = structlog.get_logger(__name__)

# Arbitrary but stable bigint key for the advisory lock. The value is
# irrelevant beyond being identical across every writer; here it's the hex
# bytes for "Marp" (MarketPulse). Any writer that takes this same lock is
# serialized against every other writer between "read latest row" and "insert".
ADVISORY_LOCK_KEY = 0x4D617270

ALLOWED_EVENT_TYPES = frozenset(
    {
        "supervisor_decision",
        "sub_agent_start",
        "tool_call",
        "sub_agent_end",
        "synthesis",
        "hitl_request",
        "hitl_response",
        "error",
    }
)

ALLOWED_ACTORS = frozenset(
    {
        "supervisor",
        "yelp_agent",
        "sec_agent",
        "fred_agent",
        "synthesize",
        "hitl",
        "system",
    }
)


def _get_dsn() -> str:
    """Return DATABASE_URL or raise a clear error.

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


def _canonical_json(obj: Any) -> str:
    """Canonical JSON encoding for hashing.

    Sorted keys, no whitespace separators, str fallback for non-JSON types
    like Decimal, datetime, UUID. Deterministic across runs.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _compute_row_hash(
    query_id: str,
    event_type: str,
    actor: str,
    payload: dict,
    prev_hash: str | None,
    created_at: datetime.datetime,
) -> str:
    """Compute SHA-256 hex of the canonical content envelope.

    The envelope is a dict with sorted keys serialized via _canonical_json.
    The output is always 64 hex chars; we sanity-check that.
    """
    envelope = {
        "query_id": str(query_id),
        "event_type": event_type,
        "actor": actor,
        "payload": payload,
        "prev_hash": prev_hash,
        "created_at": created_at.isoformat(),
    }
    canonical = _canonical_json(envelope)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    # SHA-256 hex is always 64 chars; if not, something is very wrong.
    if len(digest) != 64:
        raise RuntimeError(
            f"sha256 hex digest is {len(digest)} chars, expected 64. "
            "hashlib is broken or the input encoding is wrong."
        )
    return digest


def _validate_event(
    query_id: str | uuid.UUID,
    event_type: str,
    actor: str,
    payload: dict,
) -> None:
    """Client-side validation, mirroring the DB CHECK constraints.

    Defense in depth: the DB is the last line of defense, but failing here
    gives a Python-native error with a useful stack if a typo creeps in.
    """
    if event_type not in ALLOWED_EVENT_TYPES:
        raise ValueError(
            f"event_type {event_type!r} not allowed; "
            f"must be one of {sorted(ALLOWED_EVENT_TYPES)}"
        )
    if actor not in ALLOWED_ACTORS:
        raise ValueError(
            f"actor {actor!r} not allowed; must be one of {sorted(ALLOWED_ACTORS)}"
        )
    if not isinstance(payload, dict):
        raise TypeError(f"payload must be a dict, got {type(payload).__name__}")
    if not str(query_id).strip():
        raise ValueError("query_id must be a non-empty string or UUID")


async def write_event(
    query_id: str | uuid.UUID,
    event_type: str,
    actor: str,
    payload: dict,
) -> int:
    """Write one event to the audit log, returning the inserted row's id.

    Computes prev_hash from the latest row, computes row_hash from the
    canonical content envelope, and inserts the row in a transaction guarded
    by an advisory lock.
    """
    # 1. Validate inputs client-side.
    _validate_event(query_id, event_type, actor, payload)

    # 2. Compute the timestamp in Python so the stored row matches the hash.
    created_at = datetime.datetime.now(datetime.timezone.utc)

    # 3. Open a fresh async connection (connection-per-call, no pool yet).
    #    The `async with` block commits on success, rolls back on exception,
    #    and closes the connection. autocommit defaults to False, so the
    #    advisory lock is held until the transaction ends.
    async with await psycopg.AsyncConnection.connect(_get_dsn()) as conn:
        async with conn.cursor() as cur:
            # 4. Serialize writers: only one writer may sit between the
            #    "latest row" read and the insert below. Auto-released on
            #    commit or rollback.
            await cur.execute(
                "SELECT pg_advisory_xact_lock(%s)", (ADVISORY_LOCK_KEY,)
            )

            # 5. Read the tail of the chain. No rows -> this is the genesis row.
            await cur.execute(
                "SELECT row_hash FROM audit_log ORDER BY id DESC LIMIT 1"
            )
            latest = await cur.fetchone()
            prev_hash = latest[0] if latest is not None else None

            # 6. Compute this row's hash over the canonical envelope.
            row_hash = _compute_row_hash(
                query_id=str(query_id),
                event_type=event_type,
                actor=actor,
                payload=payload,
                prev_hash=prev_hash,
                created_at=created_at,
            )

            # 7. Insert, passing created_at explicitly (not the column DEFAULT)
            #    so the persisted value matches what we hashed.
            await cur.execute(
                """
                INSERT INTO audit_log
                    (query_id, event_type, actor, payload,
                     prev_hash, row_hash, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    str(query_id),
                    event_type,
                    actor,
                    Jsonb(payload),
                    prev_hash,
                    row_hash,
                    created_at,
                ),
            )
            inserted = await cur.fetchone()
            row_id = inserted[0]

    # 8. Commit happened on exiting the `async with conn` block above.
    log.info(
        "audit_event_written",
        id=row_id,
        event_type=event_type,
        query_id=str(query_id),
    )
    return row_id


async def verify_chain(
    query_id: str | uuid.UUID | None = None,
    limit: int | None = None,
) -> tuple[bool, int | None]:
    """Verify the prev_hash chain is intact.

    If query_id is provided, verify only that query's rows (in id order).
    If limit is provided, only check the first `limit` rows after any
    filter is applied.

    Returns:
        (True, None) if intact
        (False, broken_row_id) at the first row where prev_hash does NOT
        match the previous row's row_hash

    Caveat: when filtering by query_id, the first row of the subset is NOT
    cross-validated against the preceding global row. A query's events are
    generally interleaved with other queries in the global chain, so the
    subset's first prev_hash legitimately points at some other query's row.
    We therefore accept any first prev_hash under a query_id filter and only
    validate linkage between consecutive rows within the returned subset.

    Note: this verifies prev_hash linkage only. To also verify each row's
    content (re-hash and compare to row_hash), a separate function
    verify_row_hashes() would be needed — not in scope here.
    """
    sql = "SELECT id, prev_hash, row_hash FROM audit_log"
    params: list[Any] = []
    if query_id is not None:
        sql += " WHERE query_id = %s"
        params.append(str(query_id))
    sql += " ORDER BY id ASC"
    if limit is not None:
        sql += " LIMIT %s"
        params.append(limit)

    async with await psycopg.AsyncConnection.connect(_get_dsn()) as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()

    prev_row_hash: str | None = None
    for index, (row_id, prev_hash, row_hash) in enumerate(rows):
        if index == 0:
            # First row of the (possibly filtered) subset.
            if query_id is None and row_id == 1:
                # True genesis row of the whole log: prev_hash MUST be NULL.
                if prev_hash is not None:
                    log.warning(
                        "audit_chain_broken",
                        broken_row_id=row_id,
                        reason="genesis_prev_hash_not_null",
                    )
                    return (False, row_id)
            # Otherwise accept whatever the first prev_hash is (see caveat).
        else:
            # Every subsequent row must link to the previous row's row_hash.
            if prev_hash != prev_row_hash:
                log.warning(
                    "audit_chain_broken",
                    broken_row_id=row_id,
                    reason="prev_hash_mismatch",
                )
                return (False, row_id)
        prev_row_hash = row_hash

    log.info(
        "audit_chain_verified",
        query_id=str(query_id) if query_id is not None else None,
        n_rows=len(rows),
    )
    return (True, None)


async def _smoke_test() -> None:
    """Write three linked events, verify the chain, and print a summary."""
    query_id = uuid.uuid4()
    print(f"smoke test query_id = {query_id}")

    id1 = await write_event(
        query_id=query_id,
        event_type="supervisor_decision",
        actor="supervisor",
        payload={"target": "yelp", "reasoning": "smoke test"},
    )
    id2 = await write_event(
        query_id=query_id,
        event_type="tool_call",
        actor="yelp_agent",
        payload={
            "tool": "find_businesses_by_category",
            "input": {"category": "restaurants", "city": "Austin"},
            "output_preview": "12 businesses found...",
            "confidence": 0.85,
        },
    )
    id3 = await write_event(
        query_id=query_id,
        event_type="sub_agent_end",
        actor="yelp_agent",
        payload={"n_calls": 1, "n_errors": 0},
    )
    print(f"inserted ids: {id1}, {id2}, {id3}")

    is_intact, broken_id = await verify_chain(query_id=query_id)
    assert is_intact, f"chain broken at row {broken_id}"

    async with await psycopg.AsyncConnection.connect(_get_dsn()) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, event_type,
                       LEFT(prev_hash, 12) AS prev_hash_12,
                       LEFT(row_hash, 12) AS row_hash_12
                FROM audit_log
                WHERE query_id = %s
                ORDER BY id
                """,
                (str(query_id),),
            )
            rows = await cur.fetchall()

    print(f"\nrows for this query: {len(rows)}, intact: {is_intact}")
    print(f"{'id':>6}  {'event_type':<20}  {'prev_hash':<14}  {'row_hash':<14}")
    print("-" * 60)
    for row_id, event_type, prev_h, row_h in rows:
        prev_display = prev_h if prev_h is not None else "(genesis)"
        print(f"{row_id:>6}  {event_type:<20}  {prev_display:<14}  {row_h:<14}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(_smoke_test())
