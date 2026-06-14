"""LLM-based confidence scoring for MarketPulse sub-agents.

After a sub-agent finishes its (single round of) tool calls, it asks a SECOND
LLM — Claude Haiku 4.5, the same model the sub-agents use (ADR-002) — to judge
how well the gathered data answers the user's query. The judge returns a typed
ConfidenceJudgment: a 0-1 score, a one-sentence reason, and any data-quality
flags worth surfacing.

This is deliberately ONE call per sub-agent invocation (not per tool call): the
judge sees the whole picture — query, tools called, results, errors — and emits
a single score. The score later gates HITL routing in the supervisor (Chunk 3).

On any LLM failure the judge falls back to a neutral 0.5 so a flaky judge never
crashes a sub-agent; the failure is logged and flagged via "judgment_failed".
"""

from __future__ import annotations

import json

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

log = structlog.get_logger(__name__)

# One judge LLM per process. Same model as the sub-agents (ADR-002).
# temperature=0 for repeatable scoring.
_judge_llm = ChatAnthropic(
    model="claude-haiku-4-5",
    temperature=0,
    max_tokens=1024,
)

# How much of each result we show the judge. Keeps the prompt budget bounded
# even when a tool returns a large blob (e.g. full filing text).
_RESULT_PREVIEW_CHARS = 500


class ConfidenceJudgment(BaseModel):
    """Typed verdict from the confidence judge."""

    score: float = Field(ge=0.0, le=1.0, description="Confidence 0-1")
    reason: str = Field(max_length=300, description="One-sentence rationale")
    data_quality_flags: list[str] = Field(
        default_factory=list,
        description=(
            "Any data quality signals worth flagging "
            "(e.g., 'empty_results', 'stale_data')."
        ),
    )


_JUDGE_SYSTEM_PROMPT = """You are a data-quality judge for the MarketPulse \
research agent. A specialist sub-agent has just gathered data for part of a \
user's query. Your job is to score how well the data it gathered answers the \
portion of the query relevant to that sub-agent's domain.

Score on this rubric:
- 0.9-1.0: Results directly and completely answer the relevant portion of the \
query.
- 0.7-0.9: Results substantively answer the query with minor gaps.
- 0.5-0.7: Partial answer — gathered some data but the user might want more.
- 0.3-0.5: Limited data, significant gaps.
- 0.0-0.3: No useful data gathered (empty results, all tool errors, or tools \
refused to act).

Judge ONLY the sub-agent's own domain. The query may be multi-domain; other \
sub-agents handle the other parts. Do not penalize this sub-agent for ignoring \
out-of-domain portions of the query.

Use data_quality_flags to surface notable signals. Suggested flags (use these \
where they fit, and add your own if useful):
- "empty_results"  — tool returned no data
- "stale_data"     — e.g. the Yelp dataset is a 2022 Philadelphia/Tampa snapshot
- "partial_match"  — results found but may not match query intent
- "tool_errors"    — one or more tool calls failed
- "narrow_scope"   — query asked broadly, sub-agent answered narrowly
- "no_tool_calls"  — the sub-agent did not call any tools

Return a single confidence score, a one-sentence reason, and any flags."""


def _summarize_tool_calls(tool_calls: list[dict]) -> str:
    """Render the tool calls (name + args) for the judge prompt."""
    if not tool_calls:
        return "(none — the sub-agent did not call any tools)"
    lines = []
    for call in tool_calls:
        name = call.get("tool", call.get("name", "?"))
        args = call.get("input", call.get("args", {}))
        lines.append(f"- {name}({json.dumps(args, default=str)})")
    return "\n".join(lines)


def _summarize_results(results: list[dict]) -> str:
    """Render a truncated preview of each result for the judge prompt."""
    if not results:
        return "(none)"
    lines = []
    for item in results:
        tool = item.get("tool", "?")
        preview = json.dumps(item.get("result"), default=str)[:_RESULT_PREVIEW_CHARS]
        lines.append(f"- {tool}: {preview}")
    return "\n".join(lines)


def _summarize_errors(errors: list[dict]) -> str:
    """Render any errors for the judge prompt."""
    if not errors:
        return "(none)"
    return "\n".join(f"- {json.dumps(e, default=str)}" for e in errors)


async def judge_confidence(
    agent: str,
    user_query: str,
    tool_calls: list[dict],
    results: list[dict],
    errors: list[dict],
) -> ConfidenceJudgment:
    """Use Haiku 4.5 to judge how well the gathered data answers the query.

    Falls back to a neutral 0.5 confidence on LLM failure (logged warning).
    """
    user_message = (
        f"Sub-agent: {agent}\n"
        f"User query:\n{user_query}\n\n"
        f"Tools called:\n{_summarize_tool_calls(tool_calls)}\n\n"
        f"Results gathered:\n{_summarize_results(results)}\n\n"
        f"Errors:\n{_summarize_errors(errors)}\n\n"
        "Score how well this sub-agent's gathered data answers the relevant "
        "portion of the query."
    )

    judge = _judge_llm.with_structured_output(ConfidenceJudgment)

    try:
        judgment: ConfidenceJudgment = await judge.ainvoke([
            SystemMessage(content=_JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ])
        log.info(
            "confidence_judged",
            agent=agent,
            score=judgment.score,
            flags=judgment.data_quality_flags,
        )
        return judgment
    except Exception as exc:
        log.warning("confidence_judge_failed", agent=agent, error=str(exc))
        return ConfidenceJudgment(
            score=0.5,
            reason="Confidence judgment failed; using neutral fallback",
            data_quality_flags=["judgment_failed"],
        )
