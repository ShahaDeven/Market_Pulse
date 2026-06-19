"""Eval runner for MarketPulse.

Loads golden memo YAML files from evals/golden_memos/, runs each query
through the agent, captures actual outputs, and writes the full results
to evals/results/eval_<timestamp>.json for later scoring.

Usage:
    uv run python -m evals.runner                  # run all goldens
    uv run python -m evals.runner --only query_001 # run a single golden

The runner does NOT score outputs against goldens — that's the scorer's
job (next chunk). This module produces the raw eval data.

Result row schema (one per golden):
    name                    golden filename stem (e.g. "query_001")
    query                   the natural-language query (from golden)
    golden_memo             the reference Memo dict (denormalized from golden)
    notes                   the golden's free-form passing-criteria notes
    actual_memo             the Memo dict the agent produced, or None
    supervisor_decisions    list[str] of routing rationales
    tool_calls              list[dict] of every tool invocation
    errors                  list[dict] of recorded tool/sub-agent errors
    hitl_events             list[dict] of HITL interrupts (auto-approved here)
    sub_agent_confidences   dict {yelp: float|None, sec: ..., fred: ...}
    run_metadata            {query_id, started_at, duration_seconds, run_error}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import structlog
import yaml
from langgraph.types import Command

from agent.graph import compiled_graph
from agent.state import new_state

GOLDENS_DIR = Path("evals/golden_memos")
RESULTS_DIR = Path("evals/results")


def _load_goldens(only: str | None = None) -> list[dict]:
    """Load all golden YAML files from evals/golden_memos/.

    Each returned dict has: name (filename stem), query, golden_memo, notes.
    If `only` is provided, returns just the matching file or raises
    FileNotFoundError.

    Note: the golden YAML stores the reference memo under ``expected_memo``;
    we surface it here as ``golden_memo`` to match the result-row schema.
    """
    if only is not None:
        path = GOLDENS_DIR / f"{only}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"No golden named {only!r} at {path}")
        paths = [path]
    else:
        paths = sorted(GOLDENS_DIR.glob("*.yaml"))
        if not paths:
            raise FileNotFoundError(f"No golden YAML files found in {GOLDENS_DIR}")

    goldens: list[dict] = []
    for path in paths:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        goldens.append(
            {
                "name": path.stem,
                "query": data["query"],
                "golden_memo": data["expected_memo"],
                "notes": data.get("notes", ""),
            }
        )
    return goldens


def _build_run_input(query: str) -> tuple[dict, dict]:
    """Build the initial state and graph config for a query run.

    Returns (initial_state, config) where:
      - initial_state is the AgentState dict from new_state(query)
      - config is the dict containing thread_id for the checkpointer
    """
    initial_state = new_state(query)
    # The MemorySaver checkpointer needs a stable thread_id to correlate the
    # initial run with any resume after a HITL pause. The query_id is unique
    # per run, so it doubles as the thread_id.
    config = {"configurable": {"thread_id": initial_state["query_id"]}}
    return initial_state, config


async def _run_one(name: str, golden: dict, log) -> dict:
    """Run a single golden through the agent, capturing all output.

    Auto-approves any HITL interrupts (a real eval shouldn't pause for
    human input). Captures supervisor decisions, tool calls, confidence
    scores, errors, HITL events, and the final memo.

    Returns a result dict matching the schema in the docstring at top
    of this file. On unrecoverable error, captures the exception in
    the result rather than crashing the eval run.
    """
    query = golden["query"]
    initial_state, config = _build_run_input(query)

    start = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()

    supervisor_decisions: list[str] = []
    tool_calls: list[dict] = []
    errors: list[dict] = []
    hitl_events: list[dict] = []
    final_memo_json: str | None = None
    final_state = None
    final_state_values: dict = {}
    run_error: str | None = None

    def _absorb(update: dict) -> None:
        """Fold one node's partial-state update into the running totals.

        astream(stream_mode="updates") yields only each node's delta, so we
        concatenate the append-only lists and overwrite the final_memo slot as
        updates arrive — mirroring agent.cli's _RunAccumulator.absorb.
        """
        if not isinstance(update, dict):
            return
        supervisor_decisions.extend(update.get("supervisor_log") or [])
        tool_calls.extend(update.get("tool_calls_made") or [])
        errors.extend(update.get("errors") or [])
        nonlocal final_memo_json
        if update.get("final_memo") is not None:
            final_memo_json = update["final_memo"]

    try:
        # Stream loop matching agent.cli's pattern, with auto-approve HITL.
        # Each interrupt restarts the stream via Command(resume=...); this
        # naturally handles a retry that re-triggers HITL on another agent.
        current_input: object = initial_state
        while True:
            interrupted = False
            async for event in compiled_graph.astream(current_input, config=config):
                # Detect HITL interrupt.
                if "__interrupt__" in event:
                    for intr in event["__interrupt__"]:
                        payload = getattr(intr, "value", intr)
                        for agent_info in payload.get("low_confidence_agents", []):
                            hitl_events.append(
                                {
                                    "agent": agent_info.get("agent"),
                                    "confidence": agent_info.get("confidence"),
                                    "reason": agent_info.get("reason"),
                                    "decision": "approve",  # auto-approved
                                }
                            )
                    current_input = Command(
                        resume={"decision": "approve", "retry_hint": ""}
                    )
                    interrupted = True
                    break  # restart the stream to resume the paused graph

                # Capture every node's state delta.
                for _node_name, update in event.items():
                    _absorb(update)

            if not interrupted:
                break  # graph terminated cleanly

        # After streaming completes, read the fully-reduced state for the
        # per-agent confidence scores (the deltas don't carry a tidy view).
        final_state = await compiled_graph.aget_state(config)
        final_state_values = final_state.values if final_state else {}

    except Exception as exc:
        run_error = f"{type(exc).__name__}: {exc}"
        log.warning("eval_run_failed", name=name, error=run_error)

    duration = time.perf_counter() - start

    # Extract sub-agent confidences from the final reduced state.
    sub_agent_confidences: dict[str, float | None] = {}
    for agent in ("yelp", "sec", "fred"):
        slot = final_state_values.get(f"{agent}_data")
        sub_agent_confidences[agent] = (
            slot.get("confidence") if isinstance(slot, dict) else None
        )

    # Parse the memo JSON for inclusion in the result (if produced).
    actual_memo = None
    if final_memo_json:
        try:
            actual_memo = json.loads(final_memo_json)
        except json.JSONDecodeError:
            actual_memo = {
                "_parse_error": "memo was not valid JSON",
                "_raw": final_memo_json[:500],
            }

    return {
        "name": name,
        "query": query,
        "golden_memo": golden["golden_memo"],
        "notes": golden.get("notes", ""),
        "actual_memo": actual_memo,
        "supervisor_decisions": supervisor_decisions,
        "tool_calls": tool_calls,
        "errors": errors,
        "hitl_events": hitl_events,
        "sub_agent_confidences": sub_agent_confidences,
        "run_metadata": {
            "query_id": initial_state.get("query_id"),
            "started_at": started_at,
            "duration_seconds": round(duration, 2),
            "run_error": run_error,
        },
    }


async def run_all(only: str | None = None) -> dict:
    """Run all golden memos (or just one) and write results to disk.

    Returns the full results dict (also written to disk).
    """
    log = structlog.get_logger("evals.runner")

    goldens = _load_goldens(only)
    log.info("eval_run_start", n_goldens=len(goldens))

    results = []
    for i, golden in enumerate(goldens, 1):
        name = golden["name"]
        print(f"[{i}/{len(goldens)}] running {name}... ", end="", flush=True)
        result = await _run_one(name, golden, log)
        outcome = "ok" if result["run_metadata"]["run_error"] is None else "ERR"
        duration = result["run_metadata"]["duration_seconds"]
        print(f"{outcome} ({duration:.1f}s)")
        results.append(result)

    # Aggregate results into a single output file.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = RESULTS_DIR / f"eval_{timestamp}.json"
    output = {
        "run_id": str(uuid.uuid4()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "n_goldens": len(goldens),
        "n_succeeded": sum(
            1 for r in results if r["run_metadata"]["run_error"] is None
        ),
        "n_failed": sum(
            1 for r in results if r["run_metadata"]["run_error"] is not None
        ),
        "results": results,
    }
    output_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults written to: {output_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Run MarketPulse eval suite.")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Run only the named golden (e.g. 'query_001'). Default: all.",
    )
    args = parser.parse_args()

    asyncio.run(run_all(only=args.only))


if __name__ == "__main__":
    main()
