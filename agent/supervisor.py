"""LangGraph supervisor node for MarketPulse.

Receives AgentState, decides which sub-agent (yelp/sec/fred) to invoke next,
or routes to synthesis/done. The supervisor sees a server-level summary
of capabilities — NOT the 44 individual tools, which are scoped to their
respective sub-agents.

Design choices:
- Claude Sonnet 4.5 via langchain_anthropic.
- Structured output via Pydantic — the LLM returns a typed Decision object
  rather than free-form text, eliminating parse errors.
- Routing is stateless: the supervisor looks ONLY at the current state to
  decide, not at any internal supervisor memory.
- No sub-agent is invoked twice in the same query (data slot already filled
  → skip).
"""

from __future__ import annotations

from typing import Literal

import structlog
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from .state import AgentState, AgentTarget

log = structlog.get_logger(__name__)

# Supervisor LLM. Sonnet 4.5 hits the sweet spot for routing — strong enough
# to interpret nuanced queries, cheap enough not to dominate per-query cost.
_supervisor_llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0,
    max_tokens=1024,
)


SUPERVISOR_SYSTEM_PROMPT = """You are the supervisor of a multi-agent research \
system called MarketPulse. Your job is to ROUTE the user's query to the right \
sub-agent. You DO NOT answer the query yourself.

You have three specialist sub-agents:

1. **yelp** — Consumer attention and sentiment for specific local businesses.
   Has tools for: review velocity over time, rating shifts between two windows,
   and finding businesses by category in a city.
   IMPORTANT: Data is limited to Philadelphia and Tampa businesses only,
   and the dataset is a 2022 snapshot.

2. **sec** — SEC EDGAR filings for US-listed public companies.
   Has tools for: company lookup by ticker, 10-K/10-Q/8-K filings retrieval,
   financial statements, key metrics, segment data, insider trading (Form 4)
   transactions and sentiment.

3. **fred** — US macroeconomic indicators.
   Has tools for: GDP, CPI/inflation, unemployment, 10-year Treasury yields,
   mortgage rates, money supply, credit card delinquency, and many other
   FRED time series.

Routing rules:
- Look at what data has already been gathered (the `state_summary` below).
- Pick ONE sub-agent to call next that is relevant to the query AND has not
  already been called.
- If enough data has been gathered to answer the query (or if all relevant
  sub-agents have been called), route to "synthesize".
- After synthesis is complete (final_memo is set), route to "done".

You may sometimes need to call multiple sub-agents for a single query
(e.g., "how is Chipotle doing in Philadelphia?" needs both yelp AND sec).
Pick the most relevant agent FIRST.

When a human reviewer has provided a hint for retry (the hitl_retry_hint
field, surfaced in the state summary below), USE IT to inform your routing.
The hint typically indicates which sub-agent to re-invoke or what aspect of
the query needs more data. If the hint suggests a specific sub-agent or data
source, route to that one. The retry hint is more authoritative than your
prior routing decisions.

You MUST also produce a one-sentence reasoning explaining your choice."""


class SupervisorDecision(BaseModel):
    """The supervisor's structured output."""

    target: AgentTarget = Field(
        description="Which sub-agent to invoke next, or synthesize/done."
    )
    reasoning: str = Field(
        description="One sentence explaining why this target was chosen.",
        max_length=300,
    )


def _build_state_summary(state: AgentState) -> str:
    """Generate a compact summary of what data has been gathered so far.

    The supervisor uses this to avoid re-calling sub-agents and to decide
    when synthesis is appropriate.
    """
    parts = [f"User query: {state['user_query']}"]

    gathered = []
    for agent in ("yelp", "sec", "fred"):
        slot = state.get(f"{agent}_data")
        if slot is not None:
            gathered.append(agent)

    if not gathered:
        parts.append("Data gathered so far: NONE — this is the first decision.")
    else:
        parts.append(f"Data already gathered from: {', '.join(gathered)}.")

    if state.get("final_memo"):
        parts.append("Synthesis is complete; final_memo is populated.")

    if state.get("supervisor_log"):
        recent = state["supervisor_log"][-3:]  # last 3 decisions
        parts.append(f"Recent supervisor decisions: {recent}")

    # HITL context: surface low-confidence history and any reviewer retry hint
    # so the supervisor can re-route intelligently after a human review.
    low_conf = state.get("hitl_low_confidence_agents")
    if low_conf:
        flagged = ", ".join(
            f"{e['agent']} (confidence {e['confidence']:.2f})" for e in low_conf
        )
        parts.append(
            f"Previously triggered HITL for low-confidence sub-agents: {flagged}"
        )

    retry_hint = state.get("hitl_retry_hint")
    if retry_hint:
        parts.append(
            f"Human reviewer provided this hint for the retry: {retry_hint}"
        )

    return "\n".join(parts)


async def supervisor_node(state: AgentState) -> dict:
    """The supervisor node. Decides routing, returns state update.

    Returns a dict with `target_agent` and `supervisor_log` updates.
    """
    state_summary = _build_state_summary(state)

    # Use Pydantic structured output for type-safe routing.
    structured_llm = _supervisor_llm.with_structured_output(SupervisorDecision)

    decision: SupervisorDecision = await structured_llm.ainvoke(
        [
            ("system", SUPERVISOR_SYSTEM_PROMPT),
            ("user", state_summary),
        ]
    )

    log.info(
        "supervisor_decision",
        query_id=state["query_id"],
        target=decision.target,
        reasoning=decision.reasoning,
    )

    return {
        "target_agent": decision.target,
        "supervisor_log": [decision.reasoning],  # appended via add reducer
    }


if __name__ == "__main__":
    # Smoke test: dispatch a few representative queries.
    # Run with: uv run python -m agent.supervisor
    import asyncio
    from .state import new_state

    test_queries = [
        "How is Chipotle's stock doing?",
        "What's the unemployment rate trend over the last 2 years?",
        "Show me consumer attention for the top coffee shops in Philadelphia.",
        "Is Reading Terminal Market still popular based on reviews?",
    ]

    async def run():
        for q in test_queries:
            print(f"\nQuery: {q}")
            state = new_state(q)
            update = await supervisor_node(state)
            print(f"  → target: {update['target_agent']}")
            print(f"  → reasoning: {update['supervisor_log'][0]}")

    asyncio.run(run())
