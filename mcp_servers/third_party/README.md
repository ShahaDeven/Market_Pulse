# Third-party MCP servers

This directory holds **configuration only** for community MCP servers we depend
on — not their source code. We deliberately do not vendor or re-implement them.

Per [ARCHITECTURE.md](../../ARCHITECTURE.md) ADR-002:

| Server | Version | Source | License |
|---|---|---|---|
| `sec-edgar-mcp` | `1.0.8` | [stefanoamorelli/sec-edgar-mcp](https://github.com/stefanoamorelli/sec-edgar-mcp) | AGPL-3.0 |
| `fred-mcp-server` | `1.0.2` | [stefanoamorelli/fred-mcp-server](https://github.com/stefanoamorelli/fred-mcp-server) | AGPL-3.0 |

These are production-grade and cover their domains fully, so re-implementing
them would be duplicative engineering with negligible learning value. We pin
explicit versions (not `:latest`) to insulate against upstream release cadence.

> **License note:** Both community deps are AGPL-3.0. Acceptable for portfolio
> use; flagged for revisit if productionizing in a commercial context.

Connection/launch config for these servers will be added here as we wire them
into the supervisor graph.
