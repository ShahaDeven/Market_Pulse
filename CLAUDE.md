# CLAUDE.md — MarketPulse Project Context

## Project goal
Build a multi-source equity-research agent that drafts company-outlook memos
from SEC filings (EDGAR), macro indicators (FRED), and consumer-sentiment
signals (Yelp Open Dataset), with confidence-gated HITL approval before
publication.

## Stack (decided — do not propose alternatives)
- Python 3.11+
- Agent framework: LangGraph 0.4+ (NOT LangChain AgentExecutor — it's deprecated)
- MCP framework: FastMCP for our custom server
- LLMs: Claude Sonnet 4.5 (primary), GPT-4o-mini (fallback for cheap calls)
- State store: PostgreSQL 16 in Docker
- Reviewer UI: Streamlit
- Validation: Pydantic v2
- Package manager: uv (NOT pip directly, NOT poetry)
- Containerization: Docker + Docker Compose
- Testing: pytest

## Build-vs-buy decisions (locked)
- `sec-edgar-mcp` — using stefanoamorelli/sec-edgar-mcp v1.0.8 (community, AGPL-3.0)
- `fred-mcp-server` — using stefanoamorelli/fred-mcp-server v1.0.2 (community, AGPL-3.0)
- `yelp-events-mcp` — building from scratch (no community server exists)

## Folder structure
See ARCHITECTURE.md for full layout. Top level:
- agent/ — LangGraph orchestration
- mcp_servers/yelp_events_mcp/ — our custom MCP server
- mcp_servers/third_party/ — config for community servers (not source)
- reviewer_ui/ — Streamlit app
- db/schema/ — numbered SQL migration files
- scripts/ — one-off data loaders, eval runners
- evals/ — gold memo set + harness
- tests/ — integration tests
- docker/ — Dockerfiles
- docker-compose.yml — service orchestration at root

## Conventions
- Type hints required on all function signatures
- Pydantic models for all data crossing boundaries (MCP tool I/O, agent state)
- Logging via structlog with JSON output (never print())
- Prompts live as .md files in agent/prompts/, loaded at runtime — NOT as Python strings
- All SQL in db/schema/ files, never inline in Python (except simple queries in tools)
- One MCP tool per file once a server exceeds 5 tools; below that, group in tools.py — keep them isolated and testable
- Tests live next to code: mcp_servers/yelp_events_mcp/tests/

## Things to NOT do
- Don't use LangChain AgentExecutor (deprecated). Use LangGraph.
- Don't write prompts as multi-line Python strings. Use .md files.
- Don't put secrets in .env that's committed. Use .env.example for documentation.
- Don't create new files without checking ARCHITECTURE.md for the right location.
- Don't suggest alternative libraries when a decision is in this file.
- Don't write more than ~100 lines without stopping for human review.
- Don't generate eval data — evals/gold_memos/ is human-authored ground truth.

## Verification protocol
After every code change:
1. Show me the diff before applying.
2. Tell me what to run to verify (pytest command, manual test steps).
3. Wait for me to confirm before moving to the next task.