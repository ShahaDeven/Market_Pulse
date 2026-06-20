"""st.session_state shape and phase transitions for the reviewer UI.

Keeps app.py focused on rendering. All mutation of the run's session state goes
through these functions, which implement the four-phase state machine:

    idle      → running   : start_run()           (user submits a query)
    running   → paused    : poll_runner()          (an __interrupt__ surfaced)
    running   → completed : poll_runner()          (RUN_DONE / RUN_ERROR)
    paused    → running   : submit_hitl_decision() (Approve / Reject / Retry)
    paused    → completed : poll_runner()          (Reject → graph ends → DONE)
    completed → idle      : reset_session_state()   ("Start new query")

Single run at a time, single tab — by design (see Chunk 2 notes). The agent
runs on a background thread (AgentRunner); these helpers only ever touch
session_state and the runner's queues, never the graph directly.
"""

from __future__ import annotations

import time

import streamlit as st

try:
    from reviewer_ui.agent_runner import AgentRunner
except ImportError:  # running as a bare script: reviewer_ui isn't on sys.path
    from agent_runner import AgentRunner  # type: ignore[no-redef]


def init_session_state() -> None:
    """Initialize all session keys if missing. Call once at top of app.py."""
    ss = st.session_state
    ss.setdefault("phase", "idle")          # idle|running|paused|completed
    ss.setdefault("runner", None)           # AgentRunner | None
    ss.setdefault("query", "")              # the active run's query
    ss.setdefault("events", [])             # accumulated event dicts
    ss.setdefault("hitl_payload", None)     # interrupt payload while paused
    ss.setdefault("run_outcome", None)      # synthesized|rejected|error|incomplete
    ss.setdefault("final_memo_json", None)  # memo JSON string once synthesized
    ss.setdefault("run_error", None)        # error text when outcome == error
    ss.setdefault("rejected", False)        # internal: a reject was submitted
    ss.setdefault("started_at", None)       # epoch seconds the run started
    ss.setdefault("show_retry_input", False)  # UI: retry hint box visible


def reset_session_state() -> None:
    """Return to idle, clearing the previous run. Used by 'Start new query'."""
    ss = st.session_state
    ss.phase = "idle"
    ss.runner = None
    ss.query = ""
    ss.events = []
    ss.hitl_payload = None
    ss.run_outcome = None
    ss.final_memo_json = None
    ss.run_error = None
    ss.rejected = False
    ss.started_at = None
    ss.show_retry_input = False


def start_run(query: str) -> None:
    """Transition idle → running: create and start an AgentRunner."""
    runner = AgentRunner(query)
    runner.start()

    ss = st.session_state
    ss.runner = runner
    ss.query = query
    ss.phase = "running"
    ss.events = []
    ss.hitl_payload = None
    ss.run_outcome = None
    ss.final_memo_json = None
    ss.run_error = None
    ss.rejected = False
    ss.started_at = time.time()
    ss.show_retry_input = False


def poll_runner() -> bool:
    """Drain the runner's queue, fold events into state, update the phase.

    Returns True if anything was drained (state may have changed), False if the
    queue was empty. app.py calls this each refresh tick while the run is
    active.
    """
    ss = st.session_state
    runner = ss.runner
    if runner is None:
        return False

    items = runner.get_events()
    if not items:
        return False

    for item in items:
        # Error sentinel: (RUN_ERROR, exception). Check before the dict/str
        # branches since a tuple matches none of them.
        if isinstance(item, tuple) and item and item[0] == AgentRunner.RUN_ERROR:
            exc = item[1] if len(item) > 1 else None
            ss.run_error = f"{type(exc).__name__}: {exc}" if exc else "unknown error"
            ss.run_outcome = "error"
            ss.phase = "completed"

        # Clean-completion sentinel.
        elif item == AgentRunner.RUN_DONE:
            _finalize(ss)

        # HITL pause: park in the paused phase with the payload to prompt on.
        elif isinstance(item, dict) and "__interrupt__" in item:
            ss.hitl_payload = item["__interrupt__"]
            ss.phase = "paused"

        # Ordinary node delta: keep it for the event log, capture any memo.
        elif isinstance(item, dict):
            ss.events.append(item)
            for update in item.values():
                if isinstance(update, dict) and update.get("final_memo"):
                    ss.final_memo_json = update["final_memo"]

    return True


def submit_hitl_decision(decision: str, retry_hint: str = "") -> None:
    """Send the reviewer's decision to the runner, transition paused → running.

    decision is "approve" | "reject" | "retry". On reject we flag the run so
    _finalize() reports the 'rejected' outcome once the graph terminates.
    """
    ss = st.session_state
    runner = ss.runner
    if runner is None:
        return

    if decision == "reject":
        ss.rejected = True

    runner.submit_hitl_response(decision, retry_hint)
    ss.hitl_payload = None
    ss.show_retry_input = False
    ss.phase = "running"


def _finalize(ss) -> None:
    """Resolve the final outcome when the run thread signals RUN_DONE."""
    ss.phase = "completed"
    if ss.rejected:
        ss.run_outcome = "rejected"
    elif ss.final_memo_json:
        ss.run_outcome = "synthesized"
    else:
        # Terminated without a memo and without an explicit reject — e.g. the
        # supervisor iteration cap tripped. Rare; surface it honestly.
        ss.run_outcome = "incomplete"
