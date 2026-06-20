-- ============================================================================
-- Migration: 004_memos
-- Table:     memos
--
-- WHAT
--   The artifact store for synthesized memos. While audit_log records the
--   stream of *events* during a run (supervisor decisions, tool calls, HITL
--   responses, ...), this table stores the *output*: the full structured Memo
--   produced by the synthesis step, one row per run outcome.
--
-- WHY A SEPARATE TABLE
--   audit_log is append-only and event-shaped — querying it for "the memo for
--   run X" means reassembling a synthesis event's truncated payload. The memos
--   table holds the complete Memo as JSONB plus a handful of denormalized
--   columns so the reviewer UI can list and filter runs without joining or
--   reparsing audit rows.
--
-- LINK TO AUDIT_LOG
--   memos.query_id references the same query_id used throughout audit_log, but
--   it is intentionally NOT a hard foreign key. audit_log has no UNIQUE
--   constraint on query_id (many event rows share one query_id), so a FK is
--   not expressible. The link is by convention; idx_memos_query_id supports
--   joining back to the audit trail when showing run details.
--
-- OUTCOMES
--   A run can end three ways:
--     'synthesized' — synthesis ran and produced a Memo (memo_json present).
--     'rejected'    — a human rejected the run at HITL; no memo was produced.
--     'error'       — catastrophic failure; no memo was produced.
--   The CHECK below ties memo_json's presence to the outcome: a synthesized
--   row MUST carry a memo, and any other outcome MUST NOT.
--
-- REFERENCES
--   ADR-005 — Human-in-the-loop (HITL) design
--   ADR-006 — Audit log (append-only, hash-chained)
-- ============================================================================

CREATE TABLE IF NOT EXISTS memos (
    -- Monotonic insert order / surrogate key.
    id                 BIGSERIAL    PRIMARY KEY,

    -- The agent run this memo belongs to. Links to audit_log.query_id by
    -- convention (no hard FK — see header).
    query_id           UUID         NOT NULL,

    -- The user's original question, denormalized so the UI can display a run
    -- without joining audit_log.
    user_query         TEXT         NOT NULL,

    -- The full Memo serialized as JSON. NULL for non-'synthesized' outcomes
    -- (rejected / error), where no memo was produced. Queryable via JSONB
    -- operators for ad-hoc inspection.
    memo_json          JSONB,

    -- Which sub-agents contributed, denormalized for filtering (e.g.
    -- "show only yelp+fred runs"). Empty array allowed for non-synthesized
    -- outcomes.
    data_sources_used  TEXT[]       NOT NULL,

    -- Number of findings in the memo, denormalized for list display. 0 for
    -- non-synthesized outcomes.
    n_findings         INTEGER      NOT NULL,

    -- How the run ended. See header for semantics.
    outcome            TEXT         NOT NULL,

    -- When this row was written.
    created_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),

    -- outcome must be one of the recognized run outcomes.
    CONSTRAINT memos_outcome_check CHECK (
        outcome IN ('synthesized', 'rejected', 'error')
    ),

    -- Tie memo_json's presence to the outcome: a synthesized run carries a
    -- memo; any other outcome does not. Written correctly the first time so we
    -- never have to ALTER this table later.
    CONSTRAINT memo_json_required_if_synthesized CHECK (
        (outcome = 'synthesized' AND memo_json IS NOT NULL) OR
        (outcome <> 'synthesized' AND memo_json IS NULL)
    )
);

-- Join back to the audit trail when showing a run's details.
CREATE INDEX IF NOT EXISTS idx_memos_query_id
    ON memos (query_id);

-- Chronological browsing (most recent first) — the default UI ordering.
CREATE INDEX IF NOT EXISTS idx_memos_created_at
    ON memos (created_at DESC);

-- Filtering by contributing sub-agent (TEXT[] containment) for the data
-- sources filter in the reviewer UI.
CREATE INDEX IF NOT EXISTS idx_memos_data_sources_gin
    ON memos USING GIN (data_sources_used);
