-- ============================================================================
-- Migration: 003_audit_log
-- Table:     audit_log
--
-- WHAT
--   An append-only, hash-chained audit log for MarketPulse. Every meaningful
--   event during an agent run is recorded here: supervisor decisions,
--   sub-agent invocations, individual MCP tool calls, synthesis events, and
--   human-in-the-loop (HITL) approval/rejection decisions. There is exactly
--   one audit_log table for the whole system; individual rows are linked to a
--   specific agent run via query_id.
--
-- WHY A HASH CHAIN
--   The table is tamper-evident. Each row stores the SHA-256 digest of the
--   previous row (prev_hash) and the SHA-256 digest of its own canonical
--   content (row_hash). Because each row commits to the one before it, any
--   modification or deletion of a historical row breaks the chain: the
--   recomputed hash of the altered row will no longer match the prev_hash
--   stored in its successor, and the mismatch is detected on read/verify.
--   This gives us an integrity guarantee without trusting the storage layer.
--
-- APPLICATION-COMPUTED HASHES (IMPORTANT)
--   prev_hash and row_hash are computed by the APPLICATION before insert, NOT
--   by the database. row_hash is the SHA-256 hex digest over this row's
--   canonical content: query_id, event_type, actor, payload (JSON
--   canonicalized with sorted keys and no whitespace), prev_hash, and
--   created_at. Deliberately there is NO database trigger to populate
--   prev_hash: triggers obscure the chain semantics, hide the ordering rules,
--   and make rebuilding/verifying the chain harder. Computing the chain is the
--   Python writer's responsibility.
--
-- APPEND-ONLY BY CONVENTION
--   This table is append-only. We intentionally add no ON DELETE / ON UPDATE
--   rules; integrity does not depend on the database forbidding mutation. We
--   rely instead on the chain breaking if anyone modifies or removes rows.
--
-- REFERENCES
--   ADR-005 — Human-in-the-loop (HITL) design
--   ADR-006 — Audit log (append-only, hash-chained)
-- ============================================================================

CREATE TABLE IF NOT EXISTS audit_log (
    -- Monotonic insert order; also serves as the chain sequence number.
    id          BIGSERIAL    PRIMARY KEY,

    -- Which agent query/run this event belongs to.
    query_id    UUID         NOT NULL,

    -- What kind of event this is. See CHECK constraint below for allowed values.
    event_type  TEXT         NOT NULL,

    -- Which component logged this event (supervisor, yelp_agent, sec_agent,
    -- fred_agent, synthesize, hitl, system, ...).
    actor       TEXT         NOT NULL,

    -- Event-specific data, e.g.
    --   supervisor_decision: {"target": "yelp", "reasoning": "..."}
    --   tool_call:           {"tool": "UNRATE", "input": {...},
    --                         "output_preview": "...", "confidence": 0.85}
    --   hitl_request:        {"reason": "low_confidence", "confidence": 0.45}
    --   hitl_response:       {"decision": "approved", "user_comment": ""}
    payload     JSONB        NOT NULL,

    -- SHA-256 hex digest of the previous row in chain order.
    -- NULL only for the very first row in the entire table; NOT NULL otherwise.
    prev_hash   TEXT,

    -- SHA-256 hex digest of THIS row's canonical content (application-computed).
    row_hash    TEXT         NOT NULL,

    -- Event timestamp; part of the hashed canonical content.
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),

    -- event_type must be one of the recognized lifecycle/audit events.
    CONSTRAINT audit_log_event_type_check CHECK (
        event_type IN (
            'supervisor_decision',  -- supervisor routed to a target
            'sub_agent_start',      -- sub-agent invocation began
            'tool_call',            -- a specific MCP tool was called
            'sub_agent_end',        -- sub-agent finished
            'synthesis',            -- synthesis ran (Day 6 will use)
            'hitl_request',         -- graph paused for human review (Chunk 3)
            'hitl_response',        -- human approved or rejected
            'error'                 -- something went wrong
        )
    ),

    -- actor must be a non-empty string.
    CONSTRAINT audit_log_actor_nonempty_check CHECK (length(actor) > 0),

    -- row_hash must be exactly 64 hex characters (SHA-256 hex digest).
    CONSTRAINT audit_log_row_hash_len_check CHECK (length(row_hash) = 64),

    -- prev_hash is either NULL (first row only) or exactly 64 hex characters.
    CONSTRAINT audit_log_prev_hash_len_check CHECK (
        prev_hash IS NULL OR length(prev_hash) = 64
    )
);

-- All events for a single agent run.
CREATE INDEX IF NOT EXISTS idx_audit_log_query_id
    ON audit_log (query_id);

-- Time-window queries (most recent first).
CREATE INDEX IF NOT EXISTS idx_audit_log_created_at
    ON audit_log (created_at DESC);

-- Filtering by event type during audit review.
CREATE INDEX IF NOT EXISTS idx_audit_log_event_type
    ON audit_log (event_type);
