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
- synthesize_node (Day 6) runs real synthesis: it consumes the gathered
  sub-agent data, calls Claude Sonnet 4.5 to produce a structured Memo, and
  writes it to final_memo as JSON.
"""

from __future__ import annotations

from typing import Literal

import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .state import AgentState
from .sub_agents.fred_agent import fred_agent_node
from .sub_agents.sec_agent import sec_agent_node
from .sub_agents.yelp_agent import yelp_agent_node
from .supervisor import supervisor_node
from .synthesize import synthesize
from .audit_log import write_event

log = structlog.get_logger(__name__)

# Hard cap on supervisor iterations. With three sub-agents and a synthesize
# step, a healthy query should converge in 2-4 supervisor invocations.
# 5 is a safety net for the edge case where the supervisor keeps deferring
# the "synthesize" decision.
MAX_SUPERVISOR_ITERATIONS = 5

# Sub-agents whose confidence falls below this threshold pause the graph for
# human review (HITL). Decided design constant — see ADR-005.
HITL_CONFIDENCE_THRESHOLD = 0.7


async def synthesize_node(state: AgentState) -> dict:
    """Real synthesis node — replaces the Day 4 stub.

    Calls synthesize() to produce a Memo, stores it as JSON in
    state.final_memo, and writes a synthesis event to the audit log.
    """
    query_id = state.get("query_id")
    log.info("synthesize_node_start", query_id=query_id)

    memo = await synthesize(state)
    final_memo_json = memo.model_dump_json(indent=2)

    # Audit log the synthesis event (simple payload, no row-id citations).
    try:
        await write_event(
            query_id=query_id,
            event_type="synthesis",
            actor="synthesize",
            payload={
                "n_findings": len(memo.findings),
                "data_sources_used": memo.data_sources_used,
                "caveats_count": len(memo.caveats),
                "executive_summary": memo.executive_summary[:200],
            },
        )
    except Exception as exc:
        log.warning("audit_log_write_failed", actor="synthesize", error=str(exc))

    log.info(
        "synthesize_node_ok",
        query_id=query_id,
        n_findings=len(memo.findings),
        data_sources_used=memo.data_sources_used,
    )

    return {"final_memo": final_memo_json, "target_agent": "done"}


def check_confidence_node(state: AgentState) -> dict:
    """Inspect sub-agent confidence and flag any below the HITL threshold.

    Synchronous: pure state inspection, no LLM/DB calls. Runs after every
    sub-agent. For each populated data slot with confidence below the
    threshold AND not already flagged in a prior pass, records a
    low-confidence entry. Dedup-by-agent ensures a given sub-agent triggers
    HITL at most once, which is what prevents an infinite HITL loop.

    Returns:
        - {"hitl_low_confidence_agents": [...], "hitl_pending": True} when this
          pass found NEW low-confidence agents (the `add` reducer appends them).
        - {"hitl_pending": False} otherwise. We always (re)set hitl_pending so
          the routing edge reads a value that reflects THIS pass only.
    """
    already_flagged = {
        e["agent"] for e in state.get("hitl_low_confidence_agents") or []
    }

    newly_flagged: list[dict] = []
    for agent in ("yelp", "sec", "fred"):
        slot = state.get(f"{agent}_data")
        if not slot:
            continue
        confidence = slot.get("confidence")
        if confidence is None:
            continue  # error slots / no-tool paths may lack a confidence score
        if confidence < HITL_CONFIDENCE_THRESHOLD and agent not in already_flagged:
            newly_flagged.append(
                {
                    "agent": agent,
                    "confidence": confidence,
                    "reason": slot.get("confidence_reason", ""),
                }
            )

    if newly_flagged:
        log.info(
            "check_confidence_hitl_triggered",
            query_id=state.get("query_id"),
            agents=[e["agent"] for e in newly_flagged],
        )
        return {"hitl_low_confidence_agents": newly_flagged, "hitl_pending": True}

    return {"hitl_pending": False}


async def hitl_node(state: AgentState) -> dict:
    """Pause the graph for human review via interrupt().

    Surfaces the gathered data and the low-confidence agents to whoever is
    driving the graph (the CLI). When resumed with Command(resume=<response>),
    interrupt() returns that response — expected shape:
        {"decision": "approve" | "reject" | "retry", "retry_hint": "<optional>"}

    Returns a state update carrying the reviewer's decision and hint. Clears
    hitl_pending so the signal does not linger past this review.
    """
    payload = {
        "query": state.get("user_query"),
        "low_confidence_agents": state.get("hitl_low_confidence_agents") or [],
        "data_gathered": {
            "yelp": state.get("yelp_data"),
            "sec": state.get("sec_data"),
            "fred": state.get("fred_data"),
        },
    }

    # Blocks here until the graph is resumed with Command(resume=...).
    response = interrupt(payload)
    response = response or {}

    decision = response.get("decision")
    retry_hint = response.get("retry_hint")

    log.info(
        "hitl_response_received",
        query_id=state.get("query_id"),
        decision=decision,
        has_hint=bool(retry_hint),
    )

    return {
        "hitl_decision": decision,
        "hitl_retry_hint": retry_hint,
        "hitl_pending": False,
    }


def _route_from_check_confidence(state: AgentState) -> Literal["hitl", "supervisor"]:
    """Route to HITL when the most recent confidence check flagged a new agent.

    Reads hitl_pending, which check_confidence_node sets fresh each pass, so a
    stale flag from an earlier iteration cannot misroute us here.
    """
    if state.get("hitl_pending"):
        return "hitl"
    return "supervisor"


def _route_from_hitl(state: AgentState) -> Literal["supervisor", "end"]:
    """Route based on the reviewer's decision.

    - reject  → END (terminal; do not synthesize)
    - approve → supervisor (continues; supervisor sees all slots filled and
      typically routes to synthesize)
    - retry   → supervisor (supervisor reads hitl_retry_hint and re-routes)
    """
    decision = state.get("hitl_decision")
    if decision == "reject":
        return "end"
    return "supervisor"


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
    graph.add_node("check_confidence", check_confidence_node)
    graph.add_node("hitl", hitl_node)
    graph.add_node("synthesize", synthesize_node)

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

    # Each sub-agent → check_confidence (was: → supervisor). The confidence
    # gate now sits between every sub-agent and the next supervisor decision.
    graph.add_edge("yelp_agent", "check_confidence")
    graph.add_edge("sec_agent", "check_confidence")
    graph.add_edge("fred_agent", "check_confidence")

    # check_confidence → hitl (low confidence) OR supervisor (continue).
    graph.add_conditional_edges(
        "check_confidence",
        _route_from_check_confidence,
        {
            "hitl": "hitl",
            "supervisor": "supervisor",
        },
    )

    # hitl → supervisor (approve/retry) OR END (reject).
    graph.add_conditional_edges(
        "hitl",
        _route_from_hitl,
        {
            "supervisor": "supervisor",
            "end": END,
        },
    )

    # synthesize → END
    graph.add_edge("synthesize", END)

    # A checkpointer is REQUIRED for interrupt() to actually pause and for the
    # graph to be resumable via Command(resume=...). MemorySaver keeps state
    # in-process only — adequate for a single-process CLI; cross-process HITL
    # (e.g. a Streamlit reviewer UI) would need a durable checkpointer.
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


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
        # A checkpointer is now compiled in, so a thread_id config is required.
        config = {"configurable": {"thread_id": state["query_id"]}}
        final_state = await compiled_graph.ainvoke(state, config=config)

        # If a sub-agent's confidence was low, the graph pauses at hitl_node and
        # surfaces an interrupt instead of finishing. The CLI (next prompt) will
        # prompt the user and resume with Command(resume=...). Here we just show it.
        if "__interrupt__" in final_state:
            print("\n=== GRAPH PAUSED FOR HITL (interrupt) ===")
            for intr in final_state["__interrupt__"]:
                payload = getattr(intr, "value", intr)
                print(json.dumps(payload, default=str, indent=2)[:1500])
            print(
                "\n(Resume with compiled_graph.ainvoke("
                "Command(resume={'decision': 'approve'}), config=config))"
            )
            return

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
