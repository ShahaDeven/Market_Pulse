"""MCP client manager for the MarketPulse LangGraph agent.

This module wraps MultiServerMCPClient with project-specific server configs.
It is responsible for:
  - Validating required env vars at import time (fail fast)
  - Spawning the 3 MCP servers (yelp-events authored, sec-edgar + fred
    community Docker images)
  - Caching a single client instance for the agent's lifetime

The agent calls get_mcp_client() to get the singleton, then await
client.get_tools() to retrieve LangChain Tool objects ready to bind to
LLMs via .bind_tools(tools).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

# Required env vars — fail at import time if missing rather than discovering
# during first tool call.
SEC_EDGAR_USER_AGENT = os.getenv("SEC_EDGAR_USER_AGENT")
FRED_API_KEY = os.getenv("FRED_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

if not SEC_EDGAR_USER_AGENT:
    raise RuntimeError(
        "SEC_EDGAR_USER_AGENT not set in .env (required for sec-edgar-mcp)"
    )
if not FRED_API_KEY:
    raise RuntimeError(
        "FRED_API_KEY not set in .env (required for fred-mcp-server)"
    )
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL not set in .env (required by yelp-events-mcp). "
        "Remember the host port is 5433."
    )

# Resolve repo root so yelp-events-mcp's `python -m` invocation works from
# any cwd
REPO_ROOT = Path(__file__).resolve().parent.parent

# Module-level singleton, populated lazily on first call to get_mcp_client()
_client: Optional[MultiServerMCPClient] = None


def get_mcp_client() -> MultiServerMCPClient:
    """Return the singleton MCP client, creating it on first call.

    The client manages 3 subprocesses (one per MCP server). They are spawned
    lazily — on first tool call, not on client creation — by the adapter.
    """
    global _client
    if _client is None:
        _client = MultiServerMCPClient(
            {
                "yelp-events": {
                    "command": "uv",
                    "args": [
                        "run", "python", "-m",
                        "mcp_servers.yelp_events_mcp.server",
                    ],
                    "transport": "stdio",
                    "cwd": str(REPO_ROOT),
                },
                "sec-edgar": {
                    "command": "docker",
                    "args": [
                        "run", "-i", "--rm",
                        "-e", f"SEC_EDGAR_USER_AGENT={SEC_EDGAR_USER_AGENT}",
                        "stefanoamorelli/sec-edgar-mcp:1.0.8",
                    ],
                    "transport": "stdio",
                },
                "fred": {
                    "command": "docker",
                    "args": [
                        "run", "-i", "--rm",
                        "-e", f"FRED_API_KEY={FRED_API_KEY}",
                        "stefanoamorelli/fred-mcp-server:latest",
                    ],
                    "transport": "stdio",
                },
            }
        )
    return _client


async def list_all_tools() -> list[tuple[str, str]]:
    """Return (tool_name, description_preview) for every tool across all
    servers. Useful for sanity-checking that all 3 servers are reachable
    and exposing the expected tools.
    """
    client = get_mcp_client()
    tools = await client.get_tools()
    return [
        (t.name, (t.description or "")[:120])
        for t in tools
    ]


if __name__ == "__main__":
    # Smoke test: print every tool we can reach.
    # Run with: uv run python -m agent.mcp_client
    import asyncio

    async def main():
        print("Discovering tools across all 3 MCP servers...")
        try:
            tools = await list_all_tools()
            print(f"\nFound {len(tools)} tools:\n")
            for name, desc in tools:
                print(f"  - {name}: {desc}")
        except Exception as exc:
            print(f"\nERROR: {type(exc).__name__}: {exc}")
            raise

    asyncio.run(main())
