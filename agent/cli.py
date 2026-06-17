"""Command-line entry point for MarketPulse.

Runs a single query against the compiled LangGraph and streams node execution
events as they happen. This is the user-facing surface for testing the agent
during Day 4 and beyond.

Usage:
    uv run python -m agent.cli "your query here"
    uv run python -m agent.cli "your query" --verbose
    uv run python -m agent.cli "your query" --json
    uv run python -m agent.cli "your query" --quiet
    uv run python -m agent.cli "your query" --auto-approve-hitl

Streaming events go to stderr (so they don't pollute piped output); the final
memo and summary go to stdout. That means:

    uv run python -m agent.cli "..." > result.txt

captures the memo cleanly while the live progress still shows on the terminal.

Human-in-the-loop (HITL): when a sub-agent's confidence falls below the
threshold, the graph pauses via interrupt(). In interactive (text) mode the CLI
prompts the user to Approve / Reject / Retry and resumes the graph with the
decision. In --json mode (or with --auto-approve-hitl) there is no interactive
terminal, so HITL pauses are auto-approved with a warning to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid

from langgraph.types import Command

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
        "text. Suppresses streaming output. HITL pauses are auto-approved.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Also print the full data slot gathered by each sub-agent "
        "(truncated per slot). At a HITL prompt, show the full low-confidence "
        "data slot rather than a short preview.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress per-event streaming output; print only the final memo. "
        "Does NOT skip HITL prompts — use --auto-approve-hitl for that.",
    )
    parser.add_argument(
        "--auto-approve-hitl",
        action="store_true",
        help="Do not prompt on HITL pauses; auto-approve and continue. Use for "
        "non-interactive runs.",
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
        # One record per HITL interrupt that fired, in order:
        #   {"low_confidence_agents": [...], "decision": str, "retry_hint": str}
        self.hitl_interrupts: list[dict] = []

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

    @property
    def outcome(self) -> str:
        """Derive the run's final outcome label for the summary."""
        if any(r["decision"] == "reject" for r in self.hitl_interrupts):
            return "rejected"
        if self.final_memo is not None:
            return "synthesized"
        return "incomplete"


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
        confidence = (update.get(f"{agent}_data") or {}).get("confidence")
        conf_str = f", confidence={confidence:.2f}" if confidence is not None else ""
        return f"{prefix} {agent} ran {n_calls} tool call(s){conf_str}"

    if node == "synthesize":
        return f"{prefix} synthesizing..."

    return None


def prompt_user_for_hitl(payload: dict, *, verbose: bool) -> dict:
    """Display a HITL interrupt payload and prompt the user for a decision.

    All output goes to stderr (the CLI reserves stdout for the final memo /
    summary). Returns {"decision": "approve"|"reject"|"retry", "retry_hint": str}.
    Re-prompts on invalid input. Raises RuntimeError if no interactive stdin is
    available (e.g. piped input), pointing the user at --auto-approve-hitl.
    """
    err = sys.stderr

    print(file=err)
    print("=" * 70, file=err)
    print("[HITL] CONFIDENCE BELOW THRESHOLD — HUMAN REVIEW REQUESTED", file=err)
    print("=" * 70, file=err)

    print(f"\nQuery: {payload.get('query')}", file=err)

    low_conf = payload.get("low_confidence_agents") or []
    print("\nLow-confidence sub-agent(s):", file=err)
    for agent in low_conf:
        print(f"  [{agent['agent']}] confidence={agent['confidence']:.2f}", file=err)
        print(f"      reason: {agent['reason']}", file=err)

    data_gathered = payload.get("data_gathered") or {}
    if verbose:
        # Show the full data slot for each low-confidence agent so the reviewer
        # can judge the actual gathered data.
        print("\nData gathered (low-confidence agents):", file=err)
        for name in {a["agent"] for a in low_conf}:
            dumped = json.dumps(data_gathered.get(name), indent=2, default=str)
            print(f"  [{name}]:\n{dumped}", file=err)
    else:
        # Brief preview, truncated to ~500 chars total so it stays scannable.
        preview = json.dumps(data_gathered, default=str)
        if len(preview) > 500:
            preview = preview[:500] + " ... (truncated)"
        print(f"\nData preview: {preview}", file=err)

    def _ask(prompt: str) -> str:
        # input() writes its prompt to stdout; we want stderr to keep stdout
        # pipe-clean, so emit the prompt ourselves and read with a bare input().
        print(prompt, end="", file=err, flush=True)
        try:
            return input()
        except EOFError as exc:  # non-interactive stdin
            raise RuntimeError(
                "HITL prompt requires an interactive terminal; no stdin "
                "available. Re-run with --auto-approve-hitl for non-interactive "
                "use."
            ) from exc

    while True:
        print(
            "\nOptions: [A]pprove (continue to synthesize)  "
            "[R]eject (end run)  [T] Retry (re-route with hint)",
            file=err,
        )
        choice = _ask("> ").strip().lower()
        if choice in ("a", "approve"):
            decision = "approve"
            break
        if choice in ("r", "reject"):
            decision = "reject"
            break
        if choice in ("t", "retry"):
            decision = "retry"
            break
        print("Invalid choice. Please enter A, R, or T.", file=err)

    retry_hint = ""
    if decision == "retry":
        retry_hint = _ask(
            "Optional hint for the agent (press Enter to skip): "
        ).strip()

    return {"decision": decision, "retry_hint": retry_hint}


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

    # HITL review history.
    print(f"\nHITL interrupts: {len(acc.hitl_interrupts)}", file=out)
    for i, rec in enumerate(acc.hitl_interrupts, 1):
        agents = ", ".join(a["agent"] for a in rec["low_confidence_agents"]) or "?"
        hint = (
            f" (hint: {rec['retry_hint']})"
            if rec["decision"] == "retry" and rec["retry_hint"]
            else ""
        )
        print(f"  {i}. [{agents}] -> {rec['decision']}{hint}", file=out)
    print(f"\nOutcome: {acc.outcome}", file=out)

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
        "outcome": acc.outcome,
        "hitl_events": [
            {
                "agent": a["agent"],
                "confidence": a["confidence"],
                "decision": rec["decision"],
                "retry_hint": rec["retry_hint"],
            }
            for rec in acc.hitl_interrupts
            for a in rec["low_confidence_agents"]
        ],
    }
    print(json.dumps(payload, indent=2, default=str))


def print_memo(memo_json_str: str | None, *, verbose: bool = False) -> None:
    """Pretty-print a memo JSON string to stdout.

    Parses the JSON, then renders the memo with section headings, numbered
    findings, and bulleted caveats. If verbose=True, ALSO dumps the raw JSON
    at the bottom under a "MEMO (JSON)" heading.

    If memo_json_str is None or empty, prints a brief notice that no memo was
    produced (e.g. HITL rejection). If it's a non-empty string that fails to
    parse as JSON, prints the raw string under a parse-failed heading.
    """
    out = sys.stdout
    bar = "=" * 70

    # Edge case 1: nothing to show (HITL rejection, no synthesis ran).
    if not memo_json_str:
        print("", file=out)
        print(bar, file=out)
        print("NO MEMO PRODUCED", file=out)
        print(bar, file=out)
        print(
            "The run terminated without synthesis (likely rejected during "
            "human review or no sub-agents produced data).",
            file=out,
        )
        return

    # Edge case 2: malformed JSON — show the raw string rather than crashing.
    try:
        memo = json.loads(memo_json_str)
    except json.JSONDecodeError:
        print("", file=out)
        print(bar, file=out)
        print("MEMO (raw, parse failed)", file=out)
        print(bar, file=out)
        print(memo_json_str, file=out)
        return

    print("", file=out)
    print(bar, file=out)
    print("MEMO", file=out)
    print(bar, file=out)

    print("\nEXECUTIVE SUMMARY", file=out)
    print("-" * 17, file=out)
    print(memo.get("executive_summary", "(missing)"), file=out)

    print("\nKEY FINDINGS", file=out)
    print("-" * 12, file=out)
    findings = memo.get("findings") or []
    if findings:
        for i, finding in enumerate(findings, 1):
            print(f"{i}. {finding.get('headline', '(no headline)')}", file=out)
            print(f"   {finding.get('detail', '')}", file=out)
            print(f"   Source: {finding.get('citation', '(uncited)')}", file=out)
            if i < len(findings):
                print("", file=out)
    else:
        print("(none)", file=out)

    print("\nDATA SOURCES", file=out)
    print("-" * 12, file=out)
    print(", ".join(memo.get("data_sources_used") or []) or "(none)", file=out)

    # Edge case 3: omit the CAVEATS section entirely when there are none —
    # reads cleaner than a "(none)" placeholder.
    caveats = memo.get("caveats") or []
    if caveats:
        print("\nCAVEATS", file=out)
        print("-" * 7, file=out)
        for caveat in caveats:
            print(f"- {caveat}", file=out)

    print("\nCONFIDENCE", file=out)
    print("-" * 10, file=out)
    print(memo.get("confidence_summary", "(missing)"), file=out)

    print("", file=out)
    print(bar, file=out)

    if verbose:
        print("", file=out)
        print(bar, file=out)
        print("MEMO (JSON)", file=out)
        print(bar, file=out)
        print(json.dumps(memo, indent=2), file=out)


async def run_query(
    query: str,
    *,
    as_json: bool,
    verbose: bool,
    quiet: bool,
    auto_approve_hitl: bool,
) -> int:
    """Execute one query through the graph, handling HITL pauses.

    Returns a process exit code.
    """
    initial_state = new_state(query)
    acc = _RunAccumulator(query_id=initial_state["query_id"], query=query)

    # The checkpointer (MemorySaver, compiled into the graph) needs a stable
    # thread_id to correlate the initial run with any resume after a HITL pause.
    thread_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}

    # Header + live streaming go to stderr so stdout stays pipe-clean.
    stream_events = not as_json and not quiet
    if stream_events:
        print(f"[QUERY] {query}", file=sys.stderr)
        print("-" * 70, file=sys.stderr)

    # JSON mode can't interactively prompt without corrupting its single JSON
    # object on stdout, so HITL is auto-approved there (and on explicit request).
    auto = as_json or auto_approve_hitl

    decision_n = 0

    def handle_event(event: dict) -> None:
        nonlocal decision_n
        for node, update in event.items():
            if node == "supervisor":
                decision_n += 1
            acc.absorb(node, update)
            if stream_events:
                line = _format_event(node, update, decision_n)
                if line is not None:
                    print(line, file=sys.stderr)

    def resolve_hitl(payload: dict) -> dict:
        if auto:
            reason = "JSON mode" if as_json else "--auto-approve-hitl"
            print(
                f"[HITL] interrupt fired; auto-approving ({reason}).",
                file=sys.stderr,
            )
            response = {"decision": "approve", "retry_hint": ""}
        else:
            response = prompt_user_for_hitl(payload, verbose=verbose)
        acc.hitl_interrupts.append(
            {
                "low_confidence_agents": payload.get("low_confidence_agents") or [],
                "decision": response["decision"],
                "retry_hint": response.get("retry_hint", ""),
            }
        )
        return response

    try:
        # Loop until the graph terminates without interrupting. Each interrupt
        # restarts the stream with a Command(resume=...); this naturally handles
        # nested interrupts (a retry can trigger HITL again on another agent).
        current_input: object = initial_state
        while True:
            interrupted = False
            async for event in compiled_graph.astream(current_input, config=config):
                if "__interrupt__" in event:
                    intr = event["__interrupt__"][0]
                    payload = getattr(intr, "value", intr)
                    response = resolve_hitl(payload)
                    current_input = Command(resume=response)
                    interrupted = True
                    break  # restart the stream to resume the paused graph
                handle_event(event)
            if not interrupted:
                break  # graph terminated naturally
    except KeyboardInterrupt:
        # Let the top-level handler deal with the exit code/message.
        raise
    except Exception as exc:  # graph blew up mid-run
        detail = f"{type(exc).__name__}: {exc}"
        if as_json:
            print(
                json.dumps(
                    {"query_id": acc.query_id, "query": query, "error": detail},
                    indent=2,
                )
            )
        else:
            print(f"Agent run failed: {detail}", file=sys.stderr)
        return 1

    # Successful completion. Tool-execution errors recorded in state are NOT
    # a CLI failure — they're reported in the summary but we still exit 0.
    if as_json:
        _emit_json(acc)
    else:
        # --quiet skips the SUMMARY section but still prints the formatted memo.
        # Default prints both; --verbose additionally dumps the raw memo JSON
        # (handled inside print_memo via its verbose flag).
        if not quiet:
            _print_summary(acc, verbose=verbose)
        print_memo(acc.final_memo, verbose=verbose)

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
        auto_approve_hitl=args.auto_approve_hitl,
    )


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        sys.exit(130)
