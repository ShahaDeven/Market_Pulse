"""FastMCP server entrypoint for the yelp-events MCP server.

Registers the three Yelp dataset tools and serves them over stdio transport.
Run with:

    python -m mcp_servers.yelp_events_mcp.server

The agent launches this as an ephemeral subprocess (see ARCHITECTURE.md ADR-002
and §4 data flow). Because stdio transport reserves stdout for the JSON-RPC
protocol, ALL logging goes to stderr.
"""

from __future__ import annotations

import sys

import structlog
from mcp.server.fastmcp import FastMCP

from . import tools

# structlog -> stderr as JSON. stdout is reserved for the MCP protocol; writing
# logs there would corrupt the stream, so the logger factory targets stderr.
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.WriteLoggerFactory(file=sys.stderr),
)
log = structlog.get_logger("yelp-events")

mcp = FastMCP("yelp-events")

# Register the tool functions defined in tools.py. Registering the functions
# themselves (rather than re-declaring wrappers here) lets FastMCP read their
# real signatures and docstrings as the tool schema/description.
_TOOLS = (
    tools.get_review_velocity,
    tools.get_rating_delta,
    tools.find_businesses_by_category,
)
for _tool in _TOOLS:
    mcp.tool()(_tool)


def main() -> None:
    log.info(
        "yelp-events FastMCP server starting",
        transport="stdio",
        tools=[t.__name__ for t in _TOOLS],
    )
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
