"""LLM-based synthesis: produce a structured Memo from gathered sub-agent data.

Uses Claude Sonnet 4.5 (per ADR-002 — high-quality LLM for user-facing
output) with Pydantic structured output. The synthesis LLM never produces
free-form text; it always produces a Memo object that's been validated
against the constraints in agent/memo.py.

If the LLM's first attempt produces output that fails Pydantic validation,
we retry ONCE with a stricter prompt that names the specific errors. If
the retry also fails, we generate a programmatic fallback memo so the
user always sees something — even if synthesis quality is degraded.

The synthesis function is called by the synthesize_node in agent/graph.py.
It writes the resulting Memo to state.final_memo as JSON and logs a
'synthesis' event to the audit log.
"""

from __future__ import annotations

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from .memo import Finding, Memo
from .state import AgentState

log = structlog.get_logger(__name__)

# Per-sub-agent data preview cap, so a chatty tool result can't balloon the
# synthesis prompt past a reasonable size.
_MAX_DATA_PREVIEW_CHARS = 4000

# One retry on validation failure before falling back to a programmatic memo.
_SYNTHESIS_RETRY_LIMIT = 1

# Sub-agents whose confidence falls below this MUST be surfaced as a caveat.
# Mirrors HITL_CONFIDENCE_THRESHOLD in graph.py (kept local to avoid importing
# the graph module from a pure data/LLM module).
_LOW_CONFIDENCE_THRESHOLD = 0.7

# High-quality LLM for the user-facing memo. Module-level singleton so the
# client is constructed once. max_tokens is generous: a memo with up to 10
# findings plus caveats needs room.
_synthesis_llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0,
    max_tokens=4096,
)

_SYSTEM_PROMPT = """You are the synthesis writer for MarketPulse, a multi-agent \
equity research system. Your job is to read the data gathered by specialist \
sub-agents and produce a structured memo answering the user's query.

You MUST follow these rules:

1. Ground EVERY finding in the data the sub-agents actually gathered. Do not
   invent statistics, dates, or facts. If you didn't see it in the data,
   don't claim it.

2. CITE every finding via the citation field. Use the format
   '<agent>:<tool_name>'. The agent must be one of: yelp, sec, fred.
   The tool_name should match the actual tool that produced the data.

3. SURFACE limitations as caveats, not silently. If a sub-agent had
   confidence < 0.7, mention this in a caveat. If a sub-agent's
   data_quality_flags included 'stale_data', mention it. If a portion of
   the user's query wasn't answered, mention it.

4. Keep findings concrete. "The business has many reviews" is too vague;
   "Reading Terminal Market has 5,721 reviews" is concrete.

5. Stay within length constraints. The Pydantic model will reject overlong
   fields. Aim for 1-3 sentences in executive_summary, <=200 chars per
   finding headline, <=1000 chars per finding detail.

6. Do NOT add commentary about the agent's process or your own reasoning.
   The memo is a research artifact, not a meta-narrative. No "I noticed
   that..." or "The agent gathered..." — just the findings and their
   citations.

7. The Yelp dataset is a 2022 snapshot scoped to Philadelphia and Tampa.
   Always include this as a caveat when Yelp data is used.

If you cannot produce a meaningful memo from the gathered data (e.g., all
sub-agents returned empty or errored), still produce a memo — with a
brief executive summary noting the gap, an empty-or-near-empty findings
list, and explicit caveats about what was missing."""

_RETRY_PROMPT = """The previous synthesis attempt produced output that failed \
validation with these errors:

{error_summary}

Please regenerate the memo, fixing these issues. Specifically:
- Ensure executive_summary is between 100 and 500 characters
- Each finding's citation is '<agent>:<tool_name>' shape (with the colon)
- data_sources_used contains only: yelp, sec, fred
- Each caveat is <= 300 characters
- confidence_summary is between 10 and 400 characters
- At least 1 finding is present, no more than 10"""


def _format_sub_agent_data(agent: str, data: dict | None) -> str:
    """Produce a compact human-readable summary of one sub-agent's data.

    Truncates the rendered tool results to _MAX_DATA_PREVIEW_CHARS so the
    synthesis prompt doesn't balloon. Includes the sub-agent name, its
    n_calls/confidence/confidence_reason/data_quality_flags, and a preview of
    each tool's result. Returns "(not invoked)" if the sub-agent didn't run.
    """
    if data is None:
        return "(not invoked)"

    lines: list[str] = []

    # Some error/no-tool paths populate a slot without the usual fields; read
    # everything defensively with .get().
    n_calls = data.get("n_calls")
    confidence = data.get("confidence")
    confidence_reason = data.get("confidence_reason")
    flags = data.get("data_quality_flags")

    lines.append(f"n_calls: {n_calls}")
    lines.append(f"confidence: {confidence}")
    if confidence_reason:
        lines.append(f"confidence_reason: {confidence_reason}")
    if flags:
        lines.append(f"data_quality_flags: {flags}")

    # Surface error / note fields when a slot carries them instead of results.
    if data.get("error"):
        lines.append(f"error: {data['error']}")
    if data.get("note"):
        lines.append(f"note: {data['note']}")

    results = data.get("results") or []
    if results:
        lines.append("tool results:")
        for item in results:
            tool = item.get("tool", "?")
            args = item.get("args", {})
            result_preview = str(item.get("result", ""))
            lines.append(f"  - tool={tool} args={args}")
            lines.append(f"    result={result_preview}")
    elif not data.get("error") and not data.get("note"):
        lines.append("tool results: (none)")

    rendered = "\n".join(lines)
    if len(rendered) > _MAX_DATA_PREVIEW_CHARS:
        rendered = (
            rendered[:_MAX_DATA_PREVIEW_CHARS]
            + f"\n... [truncated at {_MAX_DATA_PREVIEW_CHARS} chars]"
        )
    return rendered


def _build_synthesis_input(state: AgentState) -> str:
    """Build the user-message content for the synthesis LLM."""
    user_query = state.get("user_query", "")

    sections = [f"USER QUERY: {user_query}", ""]
    for agent in ("yelp", "sec", "fred"):
        sections.append(f"SUB-AGENT {agent.upper()}:")
        sections.append(_format_sub_agent_data(agent, state.get(f"{agent}_data")))
        sections.append("")

    sections.append(
        "Produce a Memo with at least one finding citing the gathered data "
        "and caveats noting any limitations."
    )
    return "\n".join(sections)


def _summarize_validation_errors(error: ValidationError) -> str:
    """Build a human-readable bullet summary of Pydantic validation errors."""
    lines: list[str] = []
    for err in error.errors():
        location = ".".join(str(part) for part in err.get("loc", ()))
        message = err.get("msg", "invalid value")
        lines.append(f"- {location}: {message}")
    return "\n".join(lines) if lines else "- (no specific error detail available)"


def _build_fallback_memo(state: AgentState) -> Memo:
    """Construct a programmatic, VALID Memo when both LLM attempts fail.

    Its only job is to ensure the user always sees a structured memo, even if
    degraded. It surfaces which sub-agents ran, one minimal finding per agent
    that produced data, and explicit caveats about the synthesis failure.
    """
    populated: list[str] = [
        agent
        for agent in ("yelp", "sec", "fred")
        if state.get(f"{agent}_data") is not None
    ]

    findings: list[Finding] = []
    confidence_bits: list[str] = []
    for agent in populated:
        data = state.get(f"{agent}_data") or {}
        n_calls = data.get("n_calls")
        confidence = data.get("confidence")
        confidence_bits.append(f"{agent}={confidence}")
        findings.append(
            Finding(
                headline=(
                    f"The {agent} sub-agent gathered data for this query."
                )[:200],
                detail=(
                    f"The {agent} sub-agent completed {n_calls} tool call(s) "
                    f"with reported confidence {confidence}. See the raw "
                    f"sub-agent data for specifics; automated synthesis was "
                    f"unavailable for this run."
                )[:1000],
                # Generic tool name — we don't reach into per-tool detail here.
                citation=f"{agent}:gathered_data",
            )
        )

    # findings must be non-empty. If no sub-agent ran, emit a single finding
    # documenting that gap so the memo still validates.
    if not findings:
        findings.append(
            Finding(
                headline="No sub-agent data was available to synthesize.",
                detail=(
                    "No sub-agent produced data for this query, and automated "
                    "synthesis was unavailable. There is nothing to report."
                ),
                citation="system:no_data",
            )
        )

    data_sources_used = populated or ["yelp"]

    caveats = [
        "synthesis_llm_failed: the synthesis LLM did not return a valid memo "
        "after a retry; this is a deterministic fallback summary.",
    ]
    for agent in populated:
        data = state.get(f"{agent}_data") or {}
        confidence = data.get("confidence")
        if isinstance(confidence, (int, float)) and confidence < _LOW_CONFIDENCE_THRESHOLD:
            reason = data.get("confidence_reason") or "low confidence"
            caveats.append(
                f"{agent} confidence was {confidence}: {reason}"[:300]
            )

    confidence_summary = (
        "Fallback memo. Sub-agent confidence scores: "
        + (", ".join(confidence_bits) if confidence_bits else "none")
        + ". The synthesis LLM failed validation, so these findings are a "
        "deterministic template rather than a reasoned synthesis."
    )[:400]

    return Memo(
        executive_summary=(
            "Automated synthesis was unavailable for this query, so this memo "
            "is a deterministic fallback assembled from the raw sub-agent "
            "outputs. It lists which sub-agents ran and their reported "
            "confidence; consult the underlying data for specifics."
        ),
        findings=findings,
        data_sources_used=data_sources_used,
        caveats=caveats,
        confidence_summary=confidence_summary,
    )


async def synthesize(state: AgentState) -> Memo:
    """Synthesize a Memo from the gathered sub-agent data in state.

    1. Build the synthesis prompt from state.
    2. Call Sonnet 4.5 with with_structured_output(Memo).
    3. If validation passes, return the Memo.
    4. If validation fails, retry ONCE with a stricter prompt naming the
       specific errors.
    5. If the retry also fails, return the programmatic fallback memo.
    """
    query_id = state.get("query_id")

    synthesis_prompt = _build_synthesis_input(state)
    structured_llm = _synthesis_llm.with_structured_output(Memo)

    # First attempt
    try:
        memo = await structured_llm.ainvoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=synthesis_prompt),
        ])
        log.info("synthesis_ok", query_id=query_id, attempt=1)
        return memo
    except ValidationError as exc:
        log.warning(
            "synthesis_validation_failed",
            query_id=query_id,
            attempt=1,
            errors=str(exc)[:500],
        )
        first_error_summary = _summarize_validation_errors(exc)

    # Retry once with a stricter prompt naming the specific errors.
    retry_message = synthesis_prompt + "\n\n" + _RETRY_PROMPT.format(
        error_summary=first_error_summary
    )
    try:
        memo = await structured_llm.ainvoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=retry_message),
        ])
        log.info("synthesis_ok", query_id=query_id, attempt=2)
        return memo
    except ValidationError as exc:
        log.warning(
            "synthesis_validation_failed",
            query_id=query_id,
            attempt=2,
            errors=str(exc)[:500],
        )

    # Both attempts failed — programmatic fallback so the user sees something.
    log.warning("synthesis_using_fallback", query_id=query_id)
    return _build_fallback_memo(state)


if __name__ == "__main__":
    import asyncio

    from .state import new_state

    async def main():
        state = new_state(
            "How are the most popular coffee shops in Philadelphia doing, "
            "and what's the current unemployment rate?"
        )
        # Populate realistic-looking sub-agent data slots by hand.
        state["yelp_data"] = {
            "results": [
                {
                    "tool": "find_businesses_by_category",
                    "args": {"category": "Coffee & Tea", "city": "Philadelphia"},
                    "result": [
                        {"name": "Reading Terminal Market", "review_count": 5721, "stars": 4.5},
                        {"name": "The Franklin Fountain", "review_count": 2062, "stars": 4.5},
                        {"name": "Cafe La Maude", "review_count": 1485, "stars": 4.5},
                    ],
                }
            ],
            "n_calls": 1,
            "confidence": 0.82,
            "confidence_reason": "Clear category match; top businesses identified.",
            "data_quality_flags": [],
        }
        state["fred_data"] = {
            "results": [
                {
                    "tool": "UNRATE",
                    "args": {},
                    "result": {"series_id": "UNRATE", "latest": 3.9, "date": "2024-05-01"},
                }
            ],
            "n_calls": 1,
            "confidence": 0.6,
            "confidence_reason": "Latest unemployment fetched but no trend window requested.",
            "data_quality_flags": [],
        }
        # sec_data left as None (sub-agent not invoked).

        memo = await synthesize(state)
        print(memo.model_dump_json(indent=2))
        assert isinstance(memo, Memo), "synthesize did not return a Memo"
        print("\nOK: synthesize returned a valid Memo instance.")

    asyncio.run(main())
