# MarketPulse — Architecture

> A multi-source equity-research agent with confidence-gated human-in-the-loop
> approval. This document explains *why* the system is built the way it is.
> For *how* to run it, see [README.md](./README.md). For Claude Code's working
> context, see [CLAUDE.md](./CLAUDE.md).

## 1. Problem statement

Equity analysts at alternative-data firms spend hours triangulating three kinds
of evidence before publishing a company-outlook memo:

1. **Filings evidence** — what the company itself disclosed (SEC 10-K, 10-Q, 8-K)
2. **Macro context** — what the broader economy is doing (FRED indicators)
3. **Consumer-sentiment signals** — what real customers are saying on the ground
   (Yelp reviews as a proxy for foot traffic and brand health)

A reasonable LLM can draft a memo from any one of these sources. The hard part
isn't the drafting — it's the *combination*: deciding which sources to pull,
when they disagree, and when to ask a human instead of guessing.

MarketPulse does that combination as a stateful agent, with a confidence score
attached to every claim, and a human-in-the-loop gate before publication.

## 2. System overview

```
                    User query
                        │
                        ▼
          ┌─────────────────────────┐
          │  LangGraph Orchestrator │
          │  (supervisor pattern)   │
          └─┬───────┬───────┬───────┘
            │       │       │
            ▼       ▼       ▼
       ┌────────┐┌──────┐┌─────────┐
       │Filings ││Macro ││Sentiment│
       │ Agent  ││Agent ││ Agent   │
       └───┬────┘└──┬───┘└────┬────┘
           │        │         │
           ▼        ▼         ▼
       ┌────────┐┌──────┐┌──────────────┐
       │sec-    ││fred- ││yelp-events-  │
       │edgar-  ││mcp-  ││mcp           │
       │mcp     ││server││(authored)    │
       │(comm.) ││(comm.)│              │
       └────┬───┘└──┬───┘└──────┬───────┘
            │       │            │
            ▼       ▼            ▼
       SEC EDGAR  FRED API   PostgreSQL
       (public)   (free)     (Yelp data)

       Synthesis → Confidence score → Decision
                                          │
                          ┌───────────────┴───────────────┐
                          │                               │
                  confidence ≥ 0.7              confidence < 0.7
                          │                               │
                          ▼                               ▼
                   Auto-publish              ┌────────────────────┐
                          │                  │ interrupt() graph  │
                          ▼                  │ Write to:          │
                   Postgres audit_log        │  - checkpoints     │
                                             │  - pending_approvals│
                                             └─────────┬──────────┘
                                                       │
                                                       ▼
                                             ┌────────────────────┐
                                             │ Reviewer UI        │
                                             │ (Streamlit)        │
                                             │ Approve/Edit/Reject│
                                             └─────────┬──────────┘
                                                       │
                                                       ▼
                                             Postgres approvals
                                                       │
                                                       ▼
                                             Resume from checkpoint
                                                       │
                                                       ▼
                                             Finalize → audit_log
```

## 3. Architecture Decision Records (ADRs)

### ADR-001: LangGraph over LangChain AgentExecutor

**Decision:** Use LangGraph 0.4+ for orchestration.

**Why:** LangChain's `AgentExecutor` is deprecated (EOL December 2026 per
LangChain's deprecation notice). LangGraph is the successor and explicitly
designed for stateful, multi-step agents with checkpointing and human-in-the-loop
patterns. These two requirements — durable state and HITL — are core to this
project, so adopting the framework that treats them as first-class is the
correct call.

**Alternatives considered:**
- CrewAI: stronger for autonomous multi-agent collaboration, weaker for stateful
  HITL with explicit checkpointing.
- AutoGen: similar tradeoff to CrewAI.
- Plain orchestration with custom state machine: rejected because re-implementing
  LangGraph's checkpointer with Postgres-backed durability is several days of
  work and undifferentiated from a portfolio-signal standpoint.

**Tradeoffs accepted:**
- Tighter coupling to LangChain ecosystem and its deprecation cycles.
- LangGraph's API is still maturing; some patterns require workarounds.

### ADR-002: Two community MCP servers, one authored

**Decision:** Use `stefanoamorelli/sec-edgar-mcp` (v1.0.8) and
`stefanoamorelli/fred-mcp-server` (v1.0.2) as community dependencies. Build
`yelp-events-mcp` from scratch.

**Why:**
- `sec-edgar-mcp` is production-grade (219 stars, built on `edgartools`, ships
  with Promptfoo evals, multi-transport support). Re-implementing it would be
  duplicative engineering with negligible learning value.
- `fred-mcp-server` covers all 800K+ FRED series through three generic tools
  (browse/search/get_series). Same reasoning.
- No community MCP server exists for the Yelp Open Dataset. The dataset is
  static (~10GB JSON), requires Postgres loading, and the analytical questions
  we ask of it (review velocity, rating delta, competitor openings) need custom
  SQL — generic CRUD wouldn't be useful.

**Why this is a stronger signal than building all three:**
Build-vs-buy decisions are a senior-engineer competency. Reimplementing
production-grade community code to pad a resume signals junior NIH thinking.
Picking deliberately, pinning versions, and authoring where building has real
value signals seniority.

**Tradeoffs accepted:**
- AGPL-3.0 license on both community deps. Acceptable for portfolio use;
  documented in the README so anyone forking knows. Not acceptable for some
  commercial contexts — flagged as something I'd revisit if productionizing.
- Dependency on an external maintainer's release cadence. Mitigated by pinning
  version `1.0.8` for SEC EDGAR explicitly (not `:latest`).

### ADR-003: PostgreSQL for state, dataset, and audit log

**Decision:** Single Postgres instance hosting three logical concerns:
Yelp dataset tables, LangGraph checkpoints (`AsyncPostgresSaver`), and audit
log + pending approvals.

**Why:** Operationally simpler than separate stores. The three concerns have
similar durability requirements (must survive restarts), similar read patterns
(point queries by ID), and similar scale (single-digit GB for the audit + state,
~10GB for Yelp). LangGraph's official Postgres checkpointer makes this the
path of least resistance.

**Alternatives considered:**
- SQLite for state: rejected because the project demos durable execution
  (kill -9 the agent, restart, resume), and SQLite under concurrent access
  is fragile.
- Redis for state, Postgres for data: rejected because two systems double the
  ops surface for marginal benefit in a single-user demo.

**Tradeoffs accepted:**
- All three concerns share connection-pool budget. At demo scale this is
  fine; at production scale you'd separate them.

### ADR-004: Supervisor pattern with three sub-agents

**Decision:** A supervisor agent decomposes queries into filings/macro/sentiment
subtasks and routes each to a specialist sub-agent. Each sub-agent has access
to exactly one MCP server.

**Why:** Two reasons:
1. **Tool scoping reduces hallucination.** Each sub-agent sees only the tools
   it needs (filings agent doesn't see FRED tools). Smaller tool surface =
   fewer wrong tool calls = less context burned.
2. **Failure isolation.** If the macro agent fails (e.g., FRED API timeout),
   the supervisor can synthesize from filings + sentiment alone with a noted
   gap, rather than the whole graph crashing.

**Alternatives considered:**
- Single agent with all tools available: simpler to build, but tool-confusion
  rate climbs with tool count (well-documented in the FuncBenchGen paper,
  arXiv:2509.26553, where syntactically valid but semantically wrong tool
  calls increase with tool surface).
- Pure ReAct loop without explicit supervisor: harder to checkpoint cleanly
  for HITL interrupts.

**Tradeoffs accepted:**
- Slightly more LLM calls (supervisor + sub-agents) means slightly higher
  latency and cost. Quantified in §6.

### ADR-005: Confidence-gated HITL, not approval-everything

**Decision:** Memos with confidence score ≥ 0.7 auto-publish; <0.7 routes to
human review.

**Why:** Approving every memo defeats the purpose of an agent. Approving none
is unsafe given LLM hallucination rates on financial claims (the 2025 Anthropic
April 23 postmortem documents how even Anthropic's own coding evals failed to
catch quality regressions). Confidence-gating is the production-realistic
middle ground.

**The 0.7 threshold is configurable and was chosen empirically against the
20-memo eval set.** The right value depends on the false-approve vs.
false-escalate cost ratio for the actual use case; 0.7 is a default that
balances them roughly evenly for the demo.

**Confidence is computed from a 5-axis rubric:** recency of evidence, agreement
across sources, citation coverage, numeric verifiability, and hallucination-risk
flags (e.g., quoted statistics that don't appear in retrieved context). The
rubric is in `agent/confidence_rubric.py` and documented in
`agent/prompts/confidence_scorer.md`.

**Alternatives considered:**
- Self-consistency vote (sample N memos, agree if M of N): expensive (N× cost)
  and doesn't help when the model is consistently wrong.
- LLM-as-judge for confidence: requires a separate "judge" prompt and adds
  another point of failure. Rejected in favor of explicit rubric.

### ADR-006: Audit log is append-only and cryptographically chained

**Decision:** `audit_log` table is append-only (no UPDATE or DELETE permitted
at the application level). Each row contains a `prev_hash` field linking to
the previous row's `current_hash`, making tampering detectable.

**Why:** Aligns with EU AI Act Article 14's requirement for "effective human
oversight" of high-risk AI systems, which regulators interpret as requiring
tamper-evident audit trails. Even though this is a portfolio project, signaling
awareness of the regulatory landscape is a differentiator versus typical
"I built an agent" projects.

**Implementation note:** The hash is SHA-256 over a canonical JSON serialization
of the row's content fields. Not blockchain — just a hash chain. Adequate for
the demo and easy to defend in interviews.

**Tradeoffs accepted:**
- Slight write overhead (one SELECT for prev_hash, one hash computation).
  Negligible at demo scale.

## 4. Data flow

For a typical query — "Draft an outlook memo on Chipotle Q3 2024" — the flow is:

1. **Planner node** receives the query, identifies the company (Chipotle, CMG),
   the period (Q3 2024), and the three evidence channels to pull.
2. **Supervisor** dispatches three sub-tasks in parallel:
   - Filings: pull CMG's Q3 2024 10-Q, extract revenue/margin/guidance sections.
   - Macro: pull CPI food-away-from-home, consumer sentiment index for Q3 2024.
   - Sentiment: query `yelp-events-mcp` for review velocity + rating delta across
     CMG's Yelp business IDs in the same window.
3. **Sub-agents** execute their tools (one MCP server each), return structured
   results to the supervisor.
4. **Synthesis node** drafts the memo from all three streams.
5. **Confidence scorer** applies the 5-axis rubric, produces a score and a list
   of low-confidence claims.
6. **Decision node** routes:
   - ≥0.7: write to `audit_log`, return finalized memo.
   - <0.7: call `interrupt()`, write to `pending_approvals`, save checkpoint.
7. **(If interrupted)** Reviewer UI polls `pending_approvals`, displays the memo
   with reasoning trace. Reviewer approves/edits/rejects.
8. **Resume node** is triggered by the UI write to `approvals`. LangGraph resumes
   from the checkpoint, applies the human edits, writes final memo to `audit_log`.

## 5. State and schemas

**Postgres logical layout:**

| Schema/Table | Purpose | Notes |
|---|---|---|
| `businesses`, `reviews`, `tips` | Yelp Open Dataset | Read-only after initial load |
| `langgraph_checkpoints` | LangGraph state snapshots | Managed by `AsyncPostgresSaver` |
| `pending_approvals` | Memos awaiting human review | Read by Reviewer UI |
| `approvals` | Reviewer decisions (approve/edit/reject) | Triggers graph resume |
| `audit_log` | Append-only event log | Hash-chained |

Pydantic models for all cross-boundary data are in `agent/state.py` and
`mcp_servers/*/tools.py`. No raw dicts are passed across components.

## 6. Operational characteristics

**Demo-scale numbers (single-user, 1 query at a time):**

| Metric | Value | Notes |
|---|---|---|
| Latency per memo (auto-published, no HITL) | 12–25 seconds | Most time in supervisor + sub-agent LLM calls |
| Latency per memo (HITL path, including human) | Bounded by human, ~minutes | |
| Token cost per memo (Claude Sonnet 4.5) | ~$0.05–$0.15 | Supervisor + 3 sub-agents + synthesizer + scorer |
| Postgres footprint | ~12GB (mostly Yelp data) | ~50MB checkpoints + audit at steady state |

> **Note:** Values above are placeholders. Replace with actual measurements
> after Week 4 benchmarking. Do not ship the project with un-measured numbers.

**Not designed for:** concurrent multi-user load, sub-second latency, or
adversarial inputs. These are out of scope for a portfolio demo and would
require meaningful additional work (load balancing, agent-pool warm-up,
prompt-injection hardening at the MCP boundary).

## 7. Evaluation strategy

The project ships with a 20-memo gold-standard eval set in `evals/gold_memos/`.
Each memo is hand-authored by me, post-hoc against publicly observable outcomes
(e.g., a CMG Q3 2024 memo written *after* CMG's Q4 2024 earnings release, where
the ground-truth signal is whether the memo's claims survived the next quarter's
disclosure).

This is a deliberate epistemic choice: most agent evals rely on LLM-as-judge,
which is expensive and circular (the judge has the same blind spots as the
agent). Earnings releases are an external, unambiguous ground-truth signal.

**The eval set is NOT the project's primary contribution.** It exists as
regression hygiene — runs on every prompt or model change to catch drift. It
does *not* exist to demonstrate evaluation rigor as a standalone skill (which
is already covered by separate portfolio projects).

## 8. Out of scope / "what I would do at 10×"

Honest list of what this project doesn't try to be, and what I'd add for a
production version:

- **Multi-tenant auth.** Single-user demo. Production needs OAuth + per-user
  audit-log partitioning.
- **Real-time streaming.** Memos take 12–25s; streaming intermediate output
  would improve UX but isn't core to the architectural story.
- **Prompt-injection hardening at the MCP boundary.** EDGAR filings are
  public-domain text, but a production system fetching arbitrary URLs would
  need PromptArmor-style filtering and tool-call anomaly detection.
- **Cost-bounded execution.** A recursive sub-agent loop could in principle
  burn unbounded tokens. Mitigated in code by per-graph max-step limits, but
  not by formal budget tracking. Would add for production.
- **Model routing.** All sub-agents currently use Claude Sonnet 4.5. A
  production system would route cheap classification tasks to Haiku or GPT-4o-mini.
- **Yelp-as-business-mapping.** Tying SEC tickers to Yelp business_ids is
  currently a manual seed file. Production would need a fuzzy-matching service.

## 9. Open questions

Things I'm still uncertain about and would discuss in an interview:

- **Is the 5-axis confidence rubric robust to adversarial prompts?** I don't know.
  The rubric is hand-designed and tested against the gold set, but I haven't
  red-teamed it.
- **Should sub-agents be tools the supervisor calls, or peers it routes to?**
  The current design treats them as routed peers via LangGraph edges. The
  alternative (sub-agents as supervisor tools, ReAct-style) has different
  failure characteristics that I haven't fully characterized.
- **What's the right default confidence threshold?** 0.7 is empirically chosen
  on a tiny eval set. In a real deployment, this would be a tuned hyperparameter.

---

## Change log

- 2026-05-29: Initial draft.