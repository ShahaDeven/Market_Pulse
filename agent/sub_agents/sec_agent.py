"""SEC EDGAR sub-agent for MarketPulse.

Connects to the community sec-edgar MCP server (21 tools) and decides which
to call based on the user query. The SEC server covers company lookup,
filings, financials, insider trading, and XBRL data. Most relevant tools:
  - get_cik_by_ticker(ticker): resolve a ticker to its SEC CIK
  - get_company_info(identifier): company profile from SEC records
  - get_recent_filings(identifier): recent 10-K/10-Q/8-K filings list
  - get_filing_content(identifier, accession_number): full filing text
  - get_financials(identifier): financial statements (income/balance/cash flow)
  - get_key_metrics(identifier, metrics): selected key financial metrics
  - analyze_8k(identifier, accession_number): 8-K event analysis
  - get_insider_summary(identifier): Form 4 insider trading summary
  (plus segment data, XBRL concept extraction, Form 4 detail tools, etc.)

This is a SINGLE-ROUND tool-calling agent (Decision 1, Day 4):
  - LLM gets the query + tool list, returns tool_calls in one shot
  - We execute all requested tool calls in parallel
  - Results go into state.sec_data, no internal multi-step reasoning

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

_sec_llm = ChatAnthropic(
    model="claude-haiku-4-5",
    temperature=0,
    max_tokens=2048,
)

SEC_SYSTEM_PROMPT = """You are the SEC EDGAR specialist sub-agent for MarketPulse.

Your job is to call the right SEC tools to gather data relevant to the user's \
query. You have ~21 tools covering company filings, financial statements, \
key metrics, segment data, insider trading (Form 4), and XBRL concepts — \
pick the most relevant one(s).

GUIDANCE:
- If the query references a company by TICKER, start with get_cik_by_ticker to \
resolve the company, then use the resulting identifier with the other tools.
- Match the tool to the question: 10-K/10-Q financials → get_financials or \
get_key_metrics; recent events → get_recent_filings / analyze_8k; insider \
activity → get_insider_summary or the Form 4 tools.
- Use the advanced XBRL tools only when a specific concept is requested.

You may call MULTIPLE tools in one response if useful. Return tool calls only — \
do not produce a final answer; the synthesis step will do that."""


async def sec_agent_node(state: AgentState) -> dict:
    """SEC sub-agent: decide which sec tool(s) to call, execute, store results."""
    query_id = state["query_id"]
    user_query = state["user_query"]

    log.info("sec_agent_start", query_id=query_id, query=user_query)

    client = get_mcp_client()
    tools = await client.get_tools(server_name="sec-edgar")

    if not tools:
        log.error("sec_agent_no_tools", query_id=query_id)
        return {
            "sec_data": {"error": "No sec tools available", "results": []},
            "errors": [{"node": "sec_agent", "error": "no tools discovered"}],
        }

    llm_with_tools = _sec_llm.bind_tools(tools)

    response = await llm_with_tools.ainvoke([
        SystemMessage(content=SEC_SYSTEM_PROMPT),
        HumanMessage(content=user_query),
    ])

    tool_calls = getattr(response, "tool_calls", []) or []

    if not tool_calls:
        log.info(
            "sec_agent_no_tool_calls",
            query_id=query_id,
            llm_text=getattr(response, "content", "")[:200],
        )
        return {
            "sec_data": {
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
                "node": "sec_agent",
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
                "agent": "sec",
                "tool": name,
                "input": args,
                "output_preview": str(serialized)[:200],
                "call_id": call_id,
            })
            log.info(
                "sec_agent_tool_call_ok",
                query_id=query_id, tool=name,
            )
        except Exception as exc:
            errors.append({
                "node": "sec_agent",
                "tool": name,
                "error": f"{type(exc).__name__}: {exc}",
            })
            log.warning(
                "sec_agent_tool_call_failed",
                query_id=query_id, tool=name, error=str(exc),
            )

    return {
        "sec_data": {"results": results, "n_calls": len(results)},
        "tool_calls_made": tool_calls_record,
        "errors": errors,
    }


if __name__ == "__main__":
    # Smoke test: run a sec-relevant query and inspect the result.
    import asyncio
    from ..state import new_state

    async def main():
        state = new_state(
            "What are Chipotle's most recent SEC filings and what are the key "
            "financial metrics?"
        )
        update = await sec_agent_node(state)
        print("\n=== sec_data ===")
        print(json.dumps(update.get("sec_data"), indent=2, default=str)[:2000])
        print("\n=== tool_calls_made ===")
        for c in update.get("tool_calls_made", []):
            print(f"  - {c['tool']}({c['input']})")
        print("\n=== errors ===")
        for e in update.get("errors", []):
            print(f"  - {e}")

    asyncio.run(main())
