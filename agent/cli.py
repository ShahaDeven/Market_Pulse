"""Command-line entry point for MarketPulse.

Runs a single query against the compiled LangGraph and streams node execution
events as they happen. This is the user-facing surface for testing the agent
during Day 4 and beyond.

Usage:
    uv run python -m agent.cli "your query here"
    uv run python -m agent.cli "your query" --verbose
    uv run python -m agent.cli "your query" --json
    uv run python -m agent.cli "your query" --quiet

Streaming events go to stderr (so they don't pollute piped output); the final
memo and summary go to stdout. That means:

    uv run python -m agent.cli "..." > result.txt

captures the memo cleanly while the live progress still shows on the terminal.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .graph import compiled_graph
from .state import new_state

# ASCII prefixes rather than emojis — Windows console emoji rendering is
# unreliable, and these stay readable when piped to a file.
_NODE_PREFIX = {
    "supervisor": "[SUPERVISOR]",
    "yelp_agent": "[YELP]",
    "sec_agent": "[SEC]",
    "fred_agent": "[FRED]",
    "synthesize": "[SYNTHESIZE]",
}

# Map sub-agent node names back to their short agent label / data slot key.
_AGENT_OF_NODE = {
    "yelp_agent": "yelp",
    "sec_agent": "sec",
    "fred_agent": "fred",
}


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="agent.cli",
        description="Run a MarketPulse query through the multi-agent graph.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="The natural-language research query to run.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a single machine-readable JSON object instead of pretty "
        "text. Suppresses streaming output.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Also print the full data slot gathered by each sub-agent "
        "(truncated per slot).",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress per-event streaming output; print only the final memo.",
    )
    return parser


class _RunAccumulator:
    """Accumulates the streamed partial-state updates into a final view.

    astream(stream_mode="updates") yields only each node's delta, not the
    fully reduced state. We rebuild the parts we care about by concatenating
    the append-only lists and overwriting the data slots as they arrive.
    """

    def __init__(self, query_id: str, query: str) -> None:
        self.query_id = query_id
        self.query = query
        self.supervisor_decisions: list[str] = []
        self.tool_calls: list[dict] = []
        self.errors: list[dict] = []
        self.data_slots: dict[str, dict | None] = {
            "yelp": None,
            "sec": None,
            "fred": None,
        }
        self.final_memo: str | None = None

    def absorb(self, node: str, update: dict) -> None:
        """Fold one node's partial-state update into the running totals."""
        if not isinstance(update, dict):
            return
        self.supervisor_decisions.extend(update.get("supervisor_log", []) or [])
        self.tool_calls.extend(update.get("tool_calls_made", []) or [])
        self.errors.extend(update.get("errors", []) or [])
        for slot in ("yelp", "sec", "fred"):
            value = update.get(f"{slot}_data")
            if value is not None:
                self.data_slots[slot] = value
        if update.get("final_memo") is not None:
            self.final_memo = update["final_memo"]


def _format_event(node: str, update: dict, decision_n: int) -> str | None:
    """Return a one-line, human-readable description of a streamed event.

    Returns None for events we don't surface (defensive; all known nodes are
    handled).
    """
    prefix = _NODE_PREFIX.get(node, f"[{node.upper()}]")

    if node == "supervisor":
        decisions = update.get("supervisor_log") or []
        reasoning = decisions[-1] if decisions else "(no reasoning)"
        return f"{prefix} decision {decision_n}: {reasoning}"

    if node in _AGENT_OF_NODE:
        agent = _AGENT_OF_NODE[node]
        n_calls = len(update.get("tool_calls_made") or [])
        return f"{prefix} {agent} ran {n_calls} tool call(s)"

    if node == "synthesize":
        return f"{prefix} synthesizing..."

    return None


def _print_summary(acc: _RunAccumulator, verbose: bool) -> None:
    """Print the post-run text summary to stdout."""
    out = sys.stdout

    print("", file=out)
    print("=" * 70, file=out)
    print("SUMMARY", file=out)
    print("=" * 70, file=out)

    print(f"\nSupervisor decisions: {len(acc.supervisor_decisions)}", file=out)
    for i, decision in enumerate(acc.supervisor_decisions, 1):
        print(f"  {i}. {decision}", file=out)

    # Group tool calls by agent.
    by_agent: dict[str, list[dict]] = {}
    for call in acc.tool_calls:
        by_agent.setdefault(call.get("agent", "?"), []).append(call)
    print(f"\nTool calls: {len(acc.tool_calls)}", file=out)
    for agent, calls in by_agent.items():
        print(f"  {agent} ({len(calls)}):", file=out)
        for call in calls:
            print(f"    - {call.get('tool')}({call.get('input')})", file=out)

    print(f"\nErrors: {len(acc.errors)}", file=out)
    for err in acc.errors:
        print(f"  - {err}", file=out)

    if verbose:
        print("\nData gathered:", file=out)
        for slot in ("yelp", "sec", "fred"):
            data = acc.data_slots[slot]
            if data is None:
                print(f"  {slot}: (not invoked)", file=out)
            else:
                dumped = json.dumps(data, indent=2, default=str)
                if len(dumped) > 500:
                    dumped = dumped[:500] + "\n  ... (truncated)"
                print(f"  {slot}:\n{dumped}", file=out)

    print("\nFinal memo:", file=out)
    print(acc.final_memo if acc.final_memo is not None else "(none produced)", file=out)


def _emit_json(acc: _RunAccumulator) -> None:
    """Emit the full run as a single JSON object on stdout."""
    payload = {
        "query_id": acc.query_id,
        "query": acc.query,
        "supervisor_decisions": acc.supervisor_decisions,
        "tool_calls": acc.tool_calls,
        "errors": acc.errors,
        "final_memo": acc.final_memo,
        "data_slots": acc.data_slots,
    }
    print(json.dumps(payload, indent=2, default=str))


async def run_query(query: str, *, as_json: bool, verbose: bool, quiet: bool) -> int:
    """Execute one query through the graph. Returns a process exit code."""
    initial_state = new_state(query)
    acc = _RunAccumulator(query_id=initial_state["query_id"], query=query)

    # Header + live streaming go to stderr so stdout stays pipe-clean.
    stream_events = not as_json and not quiet
    if stream_events:
        print(f"[QUERY] {query}", file=sys.stderr)
        print("-" * 70, file=sys.stderr)

    decision_n = 0
    try:
        async for event in compiled_graph.astream(initial_state):
            for node, update in event.items():
                if node == "supervisor":
                    decision_n += 1
                acc.absorb(node, update)
                if stream_events:
                    line = _format_event(node, update, decision_n)
                    if line is not None:
                        print(line, file=sys.stderr)
    except KeyboardInterrupt:
        # Let the top-level handler deal with the exit code/message.
        raise
    except Exception as exc:  # graph blew up mid-run
        detail = f"{type(exc).__name__}: {exc}"
        if as_json:
            print(json.dumps({"query_id": acc.query_id, "query": query, "error": detail}, indent=2))
        else:
            print(f"Agent run failed: {detail}", file=sys.stderr)
        return 1

    # Successful completion. Tool-execution errors recorded in state are NOT
    # a CLI failure — they're reported in the summary but we still exit 0.
    if as_json:
        _emit_json(acc)
    elif quiet:
        print(acc.final_memo if acc.final_memo is not None else "(none produced)")
    else:
        _print_summary(acc, verbose=verbose)

    return 0


async def main() -> int:
    """Parse args and dispatch. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args()

    if args.query is None:
        # No query supplied: print usage and exit 2 (argparse convention).
        parser.error("a query argument is required")

    return await run_query(
        args.query,
        as_json=args.json,
        verbose=args.verbose,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        sys.exit(130)
