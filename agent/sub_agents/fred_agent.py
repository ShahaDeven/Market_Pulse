"""FRED sub-agent for MarketPulse.

Connects to the community fred MCP server (20 tools) and decides which to
call based on the user query. The FRED server exposes US macroeconomic time
series. 19 tools are pre-named individual series; 1 is a generic fetcher:
  - UNRATE: unemployment rate
  - CPIAUCSL: Consumer Price Index (inflation)
  - GDP / GDPC1: Gross Domestic Product (nominal / real)
  - DGS10 / T10Y2Y: 10-Year Treasury yield / 10Y-2Y spread
  - MORTGAGE30US: 30-year fixed mortgage average
  - M1SL: M1 money stock
  - DRCCLACBS: credit card delinquency rate
  - ... and other named series (WALCL, T10YIE, BAMLH0A0HYM2, etc.)
  - FREDSeries(series_id): generic fetcher for ANY FRED series by ID

This is a SINGLE-ROUND tool-calling agent (Decision 1, Day 4):
  - LLM gets the query + tool list, returns tool_calls in one shot
  - We execute all requested tool calls in parallel
  - Results go into state.fred_data, no internal multi-step reasoning

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
from ..audit_log import write_event
from ..confidence import judge_confidence

log = structlog.get_logger(__name__)

_fred_llm = ChatAnthropic(
    model="claude-haiku-4-5",
    temperature=0,
    max_tokens=2048,
)

FRED_SYSTEM_PROMPT = """You are the FRED macroeconomic data specialist sub-agent \
for MarketPulse.

Your job is to call the right FRED tools to gather data relevant to the user's \
query. You have ~20 tools for US economic indicators (GDP, CPI/inflation, \
unemployment, Treasury yields, mortgage rates, money supply, credit \
delinquency, and more) — pick the most relevant one(s).

GUIDANCE:
- For popular series, use the dedicated named tools: UNRATE (unemployment), \
CPIAUCSL (CPI/inflation), GDP / GDPC1 (output), DGS10 / T10Y2Y (Treasury \
yields), MORTGAGE30US (mortgage rates), etc.
- For a less common series not covered by a named tool, use FREDSeries with \
the appropriate series_id.

IMPORTANT: If the user query mentions topics outside your domain (e.g., 
SEC filings, local Yelp businesses, company-specific financials), IGNORE 
those portions and focus on the part relevant to US macroeconomic indicators. 
Other sub-agents handle their own domains. Do not refuse to act because 
the query is multi-domain — extract the macro-relevant part and call tools 
for it.

You may call MULTIPLE tools in one response if useful. Return tool calls only — \
do not produce a final answer; the synthesis step will do that."""


async def fred_agent_node(state: AgentState) -> dict:
    """FRED sub-agent: decide which fred tool(s) to call, execute, store results."""
    query_id = state["query_id"]
    user_query = state["user_query"]

    # Read and consume the retry hint (if present from HITL retry decision)
    retry_hint = state.get("hitl_retry_hint") or ""
    if retry_hint:
        log.info("fred_agent_using_retry_hint", query_id=query_id, hint=retry_hint)

    log.info("fred_agent_start", query_id=query_id, query=user_query)

    client = get_mcp_client()
    tools = await client.get_tools(server_name="fred")

    if not tools:
        log.error("fred_agent_no_tools", query_id=query_id)
        return {
            "fred_data": {"error": "No fred tools available", "results": []},
            "errors": [{"node": "fred_agent", "error": "no tools discovered"}],
        }

    llm_with_tools = _fred_llm.bind_tools(tools)

    # Build user message, prepending the retry hint if present
    if retry_hint:
        user_message_content = (
            f"[HUMAN REVIEWER RETRY HINT]: {retry_hint}\n\n"
            f"This is a retry — the previous attempt didn't fully answer the query. "
            f"Use the hint above to guide your tool selection and parameters.\n\n"
            f"[USER QUERY]: {user_query}"
        )
    else:
        user_message_content = user_query

    response = await llm_with_tools.ainvoke([
        SystemMessage(content=FRED_SYSTEM_PROMPT),
        HumanMessage(content=user_message_content),
    ])

    tool_calls = getattr(response, "tool_calls", []) or []

    if not tool_calls:
        log.info(
            "fred_agent_no_tool_calls",
            query_id=query_id,
            llm_text=getattr(response, "content", "")[:200],
        )
        return {
            "fred_data": {
                "results": [],
                "note": "LLM did not request any tool calls for this query.",
                "llm_text": getattr(response, "content", "")[:500],
                "confidence": 0.0,
                "confidence_reason": "Sub-agent did not call any tools",
                "data_quality_flags": ["no_tool_calls"],
            },
            "tool_calls_made": [],
            "errors": [],
            "hitl_retry_hint": None,
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
                "node": "fred_agent",
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
                "agent": "fred",
                "tool": name,
                "input": args,
                "output_preview": str(serialized)[:4000],
                "call_id": call_id,
            })
            log.info(
                "fred_agent_tool_call_ok",
                query_id=query_id, tool=name,
            )
        except Exception as exc:
            errors.append({
                "node": "fred_agent",
                "tool": name,
                "error": f"{type(exc).__name__}: {exc}",
            })
            log.warning(
                "fred_agent_tool_call_failed",
                query_id=query_id, tool=name, error=str(exc),
            )

    # Judge confidence in the gathered data
    judgment = await judge_confidence(
        agent="fred",
        user_query=user_query,
        tool_calls=tool_calls_record,
        results=results,
        errors=errors,
    )

    # Audit log the sub_agent_end event
    try:
        await write_event(
            query_id=state["query_id"],
            event_type="sub_agent_end",
            actor="fred_agent",
            payload={
                "agent": "fred",
                "n_calls": len(results),
                "n_errors": len(errors),
                "confidence": judgment.score,
                "confidence_reason": judgment.reason,
                "data_quality_flags": judgment.data_quality_flags,
            },
        )
    except Exception as exc:
        log.warning(
            "audit_log_write_failed",
            agent="fred",
            error=str(exc),
        )        

    return {
        "fred_data": {
            "results": results,
            "n_calls": len(results),
            "confidence": judgment.score,
            "confidence_reason": judgment.reason,
            "data_quality_flags": judgment.data_quality_flags,
        },
        "tool_calls_made": tool_calls_record,
        "errors": errors,
        "hitl_retry_hint": None,
    }


if __name__ == "__main__":
    # Smoke test: run a fred-relevant query and inspect the result.
    import asyncio
    from ..state import new_state

    async def main():
        state = new_state(
            "What's the unemployment rate trend over the last 2 years and what's "
            "the current 10-year Treasury yield?"
        )
        update = await fred_agent_node(state)
        print("\n=== fred_data ===")
        print(json.dumps(update.get("fred_data"), indent=2, default=str)[:2000])
        print("\n=== tool_calls_made ===")
        for c in update.get("tool_calls_made", []):
            print(f"  - {c['tool']}({c['input']})")
        fred_data = update.get("fred_data") or {}
        print(f"\n=== confidence ===")
        print(f"score: {fred_data.get('confidence')}")
        print(f"reason: {fred_data.get('confidence_reason')}")
        print(f"flags: {fred_data.get('data_quality_flags')}")
        print("\n=== errors ===")
        for e in update.get("errors", []):
            print(f"  - {e}")

    asyncio.run(main())
