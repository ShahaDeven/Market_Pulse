# Third-Party MCP Servers — Architecture Decision Record

This directory holds the **configuration only** for the two
community-maintained MCP servers that MarketPulse depends on — not their source
code. We deliberately do not vendor or re-implement them. This file is written
as an ADR rather than a usage guide: the goal is to make the *engineering
judgment* behind a build-vs-buy decision legible to a reviewer clicking in from
the main repo.

See also [ARCHITECTURE.md](../../ARCHITECTURE.md) ADR-002.

## Overview

MarketPulse talks to three MCP servers. One is authored in-house
(`yelp-events-mcp`, see [`../yelp_events_mcp/`](../yelp_events_mcp/)); the other
two — `sec-edgar-mcp` and `fred-mcp-server` — are community packages, configured
here and pulled as pinned Docker images.

The **Model Context Protocol (MCP)** is the standardized, transport-agnostic
protocol that lets an LLM agent discover and call external tools through a
uniform interface, rather than every integration being a bespoke
function-calling shim. Because the interface is standardized, an ecosystem of
pre-built servers has grown up around common data sources — databases, search
APIs, SaaS products, and public datasets. That ecosystem is the thing this
document is making a decision *about*: when a competent server already exists,
the interesting engineering question is whether to adopt it or rebuild it.

## Build vs Buy Decision

Both SEC EDGAR and FRED are served by production-grade community MCP servers from
a single, identifiable maintainer
([stefanoamorelli](https://github.com/stefanoamorelli), a Python finance
engineer with a focused body of work in this space). Crucially, these servers
are thin protocol adapters over **mature underlying libraries** — `edgartools`
for SEC, and FRED's own well-documented REST API. The hard parts (EDGAR's
idiosyncratic filing index; FRED's series taxonomy and rate limits) are already
solved upstream and exercised by other users.

Re-implementing them would be duplicative engineering with no payoff. I would
learn no new pattern, expose no capability the community server doesn't already
expose, and ship more code to maintain, secure, and debug. That is textbook
**NIH ("Not Invented Here")** — rebuilding a working wheel to feel ownership over
it. The cost is real (my time, plus a permanent maintenance liability) and the
benefit is essentially zero.

Yelp is the deliberate contrast. At project start **no community MCP server
existed** for the Yelp Open Dataset, so authoring one was the correct call —
that is where the *authoring* skill is demonstrated (custom tools, Pydantic I/O
contracts, tests). For SEC and FRED, the skill being demonstrated is the
opposite and, frankly, the more senior one: **recognizing when not to build.**
The ability to draw that line — buy the commodity, build the differentiator — is
itself the signal. Junior engineers tend to NIH everything, because writing code
feels like progress; the discipline is in declining to.

## sec-edgar-mcp (`stefanoamorelli/sec-edgar-mcp` v1.0.8)

- **Repo:** https://github.com/stefanoamorelli/sec-edgar-mcp
- **License:** AGPL-3.0 (see [implications](#agpl-30-implications) below)
- **Version pinned:** `1.0.8` — **not** `:latest`. Pinning prevents silent
  upstream updates from introducing breaking changes or supply-chain risk into a
  build that previously passed review.
- **Underlying library:** `edgartools`, the canonical Python wrapper over SEC
  EDGAR's REST API.
- **Auth required:** `SEC_EDGAR_USER_AGENT` — identifies the requester per the
  SEC's fair-access policy. This is **public, not secret**: it must be accurate
  (the SEC throttles or blocks anonymous/invalid agents) but carries no
  confidentiality requirement.
- **What we use it for:** company CIK lookup, filing retrieval (10-K, 10-Q,
  8-K), and filing text / section extraction.
- **How to test:**

  ```bash
  docker run -i --rm \
    -e "SEC_EDGAR_USER_AGENT=Your Name (you@example.com)" \
    stefanoamorelli/sec-edgar-mcp:1.0.8
  ```

  A healthy server enters its stdio-listening state without errors.

- **Quirks observed during testing:**
  - Logs to stderr only; no INFO-level startup logs by default, so a healthy
    server looks like a process that has gone quiet — expected, not hung.
  - Invalid JSON-RPC input is logged as an error but does **not** crash the
    server. Good defensive behavior.
  - MCP Inspector v0.22.0 has SSE-handling bugs that surface as spurious
    "Not connected" errors. Verify with `docker run` (stdio) rather than trusting
    the Inspector here.

## fred-mcp-server (`stefanoamorelli/fred-mcp-server`)

- **Repo:** https://github.com/stefanoamorelli/fred-mcp-server
- **License:** AGPL-3.0
- **Version pinned:** `:latest` — **TODO:** pin to a specific tag (ideally a
  digest) once a stable version is verified. This is a known gap, tracked
  deliberately rather than left implicit.
- **Auth required:** `FRED_API_KEY` — free from
  https://fredaccount.stlouisfed.org/apikeys. Unlike the SEC user agent, this
  **is a secret** and must never be committed.
- **What we use it for:** searching FRED's 800K+ macroeconomic time series (GDP,
  CPI, employment, interest rates, sector-specific indicators).
- **Rate limit:** the FRED API enforces 120 requests/minute per key. The agent
  caches responses to stay well under this.
- **How to test:**

  ```bash
  docker run -i --rm -e "FRED_API_KEY=$FRED_API_KEY" \
    stefanoamorelli/fred-mcp-server:latest
  ```

  A healthy server prints `FRED MCP Server running on stdio` and waits.

- **Quirks observed:**
  - Cleaner startup than sec-edgar-mcp — it emits an explicit "running on stdio"
    readiness message rather than going silent.

## AGPL-3.0 Implications

Both servers are AGPL-3.0. This is the most consequential tradeoff in this
decision, so it is documented as a known constraint rather than a hidden
surprise:

- **Portfolio / educational use:** fully acceptable. This is MarketPulse's
  current use case.
- **Internal / private commercial use:** acceptable.
- **Commercial deployment exposed over a network:** the AGPL "network use" clause
  requires that anyone interacting with the system over a network be able to
  obtain the source of any *modifications*. This viral-over-the-network behavior
  is what distinguishes AGPL from ordinary GPL and is the clause that matters for
  a SaaS-shaped product.
- **Some industries exclude AGPL outright** — fintech especially, where many
  employers maintain hard AGPL-exclusion policies. If MarketPulse were deployed
  at such a company, these servers would have to be replaced with
  privately-developed or differently-licensed equivalents.

The case for buying-not-building does not evaporate under AGPL; it just means the
*exit cost* (re-authoring under a permissive license) is a known, bounded number
rather than an unknown.

## Risk Assessment

Single-maintainer community packages carry inherent risk. The relevant failure
modes, and the mitigations already in place:

- **Maintenance / bus factor:** these are the work of an individual contributor.
  If they step away, upstream updates stop. Pinning specific versions means a
  *stall* doesn't break us — we keep running the version we validated.
- **Quality variance / transitive deps:** each server pulls third-party
  libraries (`edgartools`, etc.) with their own version constraints. Pin both the
  MCP server image **and** verify SDK versions before any upgrade, not just the
  top-level tag.
- **Security:** AGPL community packages can carry undiscovered vulnerabilities.
  We limit blast radius — no proprietary research, no third-party API keys, and
  no PII flow through these servers.
- **Stdio transport ceiling:** these run as ephemeral subprocesses, not
  long-lived services, so they cannot be load-balanced or horizontally scaled as
  configured. For high-throughput production this would be a real limitation,
  pushing toward custom servers with HTTP transport.

## What I Would Do in Production

This is a portfolio/educational build. If it were productionized for commercial
use, the changes I would make — roughly in priority order:

1. **Pin by Docker digest, not tag** — immutable `@sha256:...` references so the
   image can never silently change underneath us.
2. **Add health-check probes** so the agent verifies a server responds before
   routing a tool call to it.
3. **Wrap tool calls in a circuit breaker** — community servers can fail
   intermittently in ways we don't control; fail fast and degrade gracefully.
4. **Cache common tool responses** — filed SEC documents are immutable once
   filed, and FRED series update at most daily, so most calls are cacheable.
5. **Run licensing past legal** before any commercial deployment, given the AGPL
   network-use clause.
6. **Author permissively-licensed replacements** (Apache-2.0 / MIT) if AGPL
   becomes a blocker, preserving downstream flexibility.
7. **Move to HTTP transport** for horizontal scaling if request volume warrants
   it.

## References

- MCP specification — https://spec.modelcontextprotocol.io/
- SEC EDGAR API — https://www.sec.gov/os/accessing-edgar-data
- FRED API — https://fred.stlouisfed.org/docs/api/fred/
- Build-vs-buy framing follows the standard "buy the commodity, build the
  differentiator" heuristic; the MarketPulse-specific, locked decisions live in
  the root [`CLAUDE.md`](../../CLAUDE.md) under *Build-vs-buy decisions*.
