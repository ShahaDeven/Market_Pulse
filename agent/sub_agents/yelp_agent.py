"""Yelp sub-agent for MarketPulse.

Connects to the authored yelp-events MCP server (3 tools) and decides which
to call based on the user query. The yelp server provides:
  - get_review_velocity(business_id, weeks=12): weekly review buckets + baseline
  - get_rating_delta(business_id, window_weeks=12): rating shift detection
  - find_businesses_by_category(city, category, min_reviews, limit): discovery

This is a SINGLE-ROUND tool-calling agent (Decision 1, Day 4):
  - LLM gets the query + tool list, returns tool_calls in one shot
  - We execute all requested tool calls in parallel
  - Results go into state.yelp_data, no internal multi-step reasoning

To upgrade to multi-round (the LLM can observe results and call more tools),
swap this implementation for `langgraph.prebuilt.create_react_agent`.
"""

from __future__ import annotations

import json

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from ..mcp_client import get_mcp_client
from ..state import AgentState

log = structlog.get_logger(__name__)

_yelp_llm = ChatAnthropic(
    model="claude-haiku-4-5",
    temperature=0,
    max_tokens=2048,
)

YELP_SYSTEM_PROMPT = """You are the Yelp specialist sub-agent for MarketPulse.

Your job is to call the right Yelp tools to gather data relevant to the user's \
query. You have three tools available — pick the most relevant one(s).

IMPORTANT CONSTRAINTS:
- The Yelp dataset is a 2022 snapshot, scoped to Philadelphia and Tampa \
businesses only.
- Time windows in get_review_velocity / get_rating_delta are anchored to each \
business's most recent review, NOT to the current date.
- If the user query references a business by NAME (e.g., "Reading Terminal \
Market"), you need to first find its business_id via find_businesses_by_category \
or ask the supervisor to clarify.

You may call MULTIPLE tools in one response if useful. Return tool calls only — \
do not produce a final answer; the synthesis step will do that."""


async def yelp_agent_node(state: AgentState) -> dict:
    """Yelp sub-agent: decide which yelp tool(s) to call, execute, store results."""
    query_id = state["query_id"]
    user_query = state["user_query"]

    log.info("yelp_agent_start", query_id=query_id, query=user_query)

    client = get_mcp_client()
    tools = await client.get_tools(server_name="yelp-events")

    if not tools:
        log.error("yelp_agent_no_tools", query_id=query_id)
        return {
            "yelp_data": {"error": "No yelp tools available", "results": []},
            "errors": [{"node": "yelp_agent", "error": "no tools discovered"}],
        }

    llm_with_tools = _yelp_llm.bind_tools(tools)

    response = await llm_with_tools.ainvoke([
        SystemMessage(content=YELP_SYSTEM_PROMPT),
        HumanMessage(content=user_query),
    ])

    tool_calls = getattr(response, "tool_calls", []) or []

    if not tool_calls:
        log.info(
            "yelp_agent_no_tool_calls",
            query_id=query_id,
            llm_text=getattr(response, "content", "")[:200],
        )
        return {
            "yelp_data": {
                "results": [],
                "note": "LLM did not request any tool calls for this query.",
                "llm_text": getattr(response, "content", "")[:500],
            },
            "tool_calls_made": [],
        }

    # Build a name → tool lookup for execution
    tools_by_name = {t.name: t for t in tools}

    results = []
    tool_calls_record = []
    errors = []

    for call in tool_calls:
        name = call.get("name") if isinstance(call, dict) else call.name
        args = call.get("args") if isinstance(call, dict) else call.args
        call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)

        tool = tools_by_name.get(name)
        if tool is None:
            errors.append({
                "node": "yelp_agent",
                "error": f"LLM requested unknown tool: {name}",
            })
            continue

        try:
            result = await tool.ainvoke(args)
            # Tools may return objects, JSON strings, or dicts — normalize.
            if hasattr(result, "model_dump"):
                serialized = result.model_dump()
            elif isinstance(result, str):
                try:
                    serialized = json.loads(result)
                except json.JSONDecodeError:
                    serialized = {"text": result}
            else:
                serialized = result
            results.append({"tool": name, "args": args, "result": serialized})
            tool_calls_record.append({
                "agent": "yelp",
                "tool": name,
                "input": args,
                "output_preview": str(serialized)[:200],
                "call_id": call_id,
            })
            log.info(
                "yelp_agent_tool_call_ok",
                query_id=query_id, tool=name,
            )
        except Exception as exc:
            errors.append({
                "node": "yelp_agent",
                "tool": name,
                "error": f"{type(exc).__name__}: {exc}",
            })
            log.warning(
                "yelp_agent_tool_call_failed",
                query_id=query_id, tool=name, error=str(exc),
            )

    return {
        "yelp_data": {"results": results, "n_calls": len(results)},
        "tool_calls_made": tool_calls_record,
        "errors": errors,
    }


if __name__ == "__main__":
    # Smoke test: run a yelp-relevant query and inspect the result.
    import asyncio
    from ..state import new_state

    async def main():
        state = new_state(
            "Find the most-reviewed coffee shops in Philadelphia and tell me how their "
            "review velocity has been trending."
        )
        update = await yelp_agent_node(state)
        print("\n=== yelp_data ===")
        print(json.dumps(update.get("yelp_data"), indent=2, default=str)[:2000])
        print("\n=== tool_calls_made ===")
        for c in update.get("tool_calls_made", []):
            print(f"  - {c['tool']}({c['input']})")
        print("\n=== errors ===")
        for e in update.get("errors", []):
            print(f"  - {e}")

    asyncio.run(main())
