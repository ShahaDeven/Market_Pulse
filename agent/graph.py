"""LangGraph StateGraph definition for MarketPulse.

The graph wires the supervisor and three sub-agents into a working pipeline:

    supervisor ↔ {yelp, sec, fred} → synthesize → END

Key design choices (see ADRs and Day 4 design docs):
- Supervisor is invoked multiple times per query (Option A flow). It reads
  state, decides what's next, and writes target_agent.
- Hard cap of 5 supervisor invocations prevents runaway loops if the
  supervisor's "synthesize" or "done" decision gets stuck.
- Sub-agents are SCC nodes — they run, populate their data slot, and
  unconditionally route back to supervisor.
- synthesize_node is a STUB in Day 4. Day 6 replaces it with real synthesis
  that consumes the gathered sub-agent data and produces final_memo.
"""

from __future__ import annotations

from typing import Literal

import structlog
from langgraph.graph import END, START, StateGraph

from .state import AgentState
from .sub_agents.fred_agent import fred_agent_node
from .sub_agents.sec_agent import sec_agent_node
from .sub_agents.yelp_agent import yelp_agent_node
from .supervisor import supervisor_node

log = structlog.get_logger(__name__)

# Hard cap on supervisor iterations. With three sub-agents and a synthesize
# step, a healthy query should converge in 2-4 supervisor invocations.
# 5 is a safety net for the edge case where the supervisor keeps deferring
# the "synthesize" decision.
MAX_SUPERVISOR_ITERATIONS = 5


async def synthesize_stub_node(state: AgentState) -> dict:
    """STUB synthesis node — Day 4 skeleton only.

    Writes a placeholder to final_memo and ends the flow. Day 6 will replace
    this with a real synthesis step that consumes the gathered sub-agent data
    and produces a structured memo.
    """
    log.info(
        "synthesize_stub_invoked",
        query_id=state.get("query_id"),
        agents_with_data=[
            a for a in ("yelp", "sec", "fred") if state.get(f"{a}_data") is not None
        ],
    )

    gathered = {}
    for agent in ("yelp", "sec", "fred"):
        data = state.get(f"{agent}_data")
        if data is not None:
            gathered[agent] = data

    stub_memo = (
        f"[Day 4 stub] Gathered data from {len(gathered)} sub-agent(s): "
        f"{', '.join(gathered.keys()) or 'none'}. "
        f"Real synthesis will replace this in Day 6."
    )

    return {"final_memo": stub_memo, "target_agent": "done"}


def _route_from_supervisor(state: AgentState) -> Literal[
    "yelp_agent", "sec_agent", "fred_agent", "synthesize", "end"
]:
    """Conditional router: reads supervisor's decision, picks next node.

    Enforces the iteration cap as a safety net. The supervisor's logic
    SHOULD return "synthesize" or "done" within a few iterations on its
    own, but the cap prevents infinite loops if the supervisor keeps
    deferring or returning an unexpected value.
    """
    iter_count = len(state.get("supervisor_log", []))
    if iter_count >= MAX_SUPERVISOR_ITERATIONS:
        log.warning(
            "supervisor_iteration_cap_exceeded",
            query_id=state.get("query_id"),
            iterations=iter_count,
        )
        return "end"

    target = state.get("target_agent")

    if target == "synthesize":
        return "synthesize"
    if target == "done":
        return "end"
    if target == "yelp":
        return "yelp_agent"
    if target == "sec":
        return "sec_agent"
    if target == "fred":
        return "fred_agent"

    # Defensive: unrecognized target → end gracefully
    log.warning(
        "unrecognized_target_agent",
        query_id=state.get("query_id"),
        target=target,
    )
    return "end"


def build_graph():
    """Construct and compile the MarketPulse StateGraph.

    Returns the compiled graph, ready for .ainvoke() or .astream() with
    an initial AgentState.
    """
    graph = StateGraph(AgentState)

    # Register all nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("yelp_agent", yelp_agent_node)
    graph.add_node("sec_agent", sec_agent_node)
    graph.add_node("fred_agent", fred_agent_node)
    graph.add_node("synthesize", synthesize_stub_node)

    # START → supervisor (always)
    graph.add_edge(START, "supervisor")

    # supervisor → conditional routing
    graph.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {
            "yelp_agent": "yelp_agent",
            "sec_agent": "sec_agent",
            "fred_agent": "fred_agent",
            "synthesize": "synthesize",
            "end": END,
        },
    )

    # Each sub-agent → supervisor (unconditional loop back)
    graph.add_edge("yelp_agent", "supervisor")
    graph.add_edge("sec_agent", "supervisor")
    graph.add_edge("fred_agent", "supervisor")

    # synthesize → END
    graph.add_edge("synthesize", END)

    return graph.compile()


# Module-level compiled graph for easy import
compiled_graph = build_graph()


if __name__ == "__main__":
    # Smoke test: run a query end-to-end through the graph
    import asyncio
    import json

    from .state import new_state

    test_query = "How are the most popular coffee shops in Philadelphia doing?"

    async def main():
        print(f"Query: {test_query}\n")
        print("=" * 70)

        state = new_state(test_query)
        final_state = await compiled_graph.ainvoke(state)

        print("\n=== Final supervisor decisions ===")
        for i, decision in enumerate(final_state.get("supervisor_log", []), 1):
            print(f"  {i}. {decision}")

        print(f"\n=== Tool calls made: {len(final_state.get('tool_calls_made', []))} ===")
        for call in final_state.get("tool_calls_made", [])[:10]:
            print(f"  - [{call.get('agent')}] {call.get('tool')}({call.get('input')})")

        print(f"\n=== Errors: {len(final_state.get('errors', []))} ===")
        for err in final_state.get("errors", []):
            print(f"  - {err}")

        print(f"\n=== Final memo (stub) ===")
        print(final_state.get("final_memo"))

        # Show what data each sub-agent gathered
        print("\n=== Sub-agent data slots ===")
        for agent in ("yelp", "sec", "fred"):
            data = final_state.get(f"{agent}_data")
            if data is not None:
                preview = json.dumps(data, default=str)[:300]
                print(f"  - {agent}: {preview}...")
            else:
                print(f"  - {agent}: (not invoked)")

    asyncio.run(main())
