"""Background-thread agent runner for the Streamlit reviewer UI.

Streamlit is synchronous and re-runs the whole script on every interaction, so
it can't host a long-lived ``async for`` stream itself. This module runs one
agent query on a daemon thread, exposing its LangGraph events through a
``queue.Queue`` the UI drains on each refresh tick.

Lifecycle of one run:
    runner = AgentRunner(query)
    runner.start()                      # spawns the daemon thread
    ...                                 # UI polls runner.get_events() every 3s
    runner.submit_hitl_response(...)    # when a HITL interrupt surfaces
    ...                                 # RUN_DONE sentinel marks completion

Threading model (single run, single tab — by design):
- The thread owns its own asyncio event loop (created by asyncio.run). The
  Windows selector-policy fix is applied INSIDE the thread target, because the
  policy must be set on the thread that will create the loop.
- Two queues bridge thread <-> UI:
    * event_queue          : thread -> UI. Streamed events + sentinels.
    * hitl_response_queue  : UI -> thread. The reviewer's decision dict.
- On a HITL interrupt the thread pushes the interrupt payload to event_queue
  then BLOCKS on hitl_response_queue.get(), parking the agent mid-graph until
  the UI calls submit_hitl_response(). This mirrors the CLI's resume loop, but
  the "human" is the Streamlit UI instead of a terminal prompt.

The MemorySaver checkpointer compiled into the graph is in-process, so the
parked run lives only as long as this process. That's the accepted trade-off
for the reviewer UI (see Chunk 2 design notes).
"""

from __future__ import annotations

import queue
import sys
import threading
import uuid
from typing import Any, Optional


class AgentRunner:
    """Runs one MarketPulse query on a daemon thread, streaming events.

    Construct with the user's query, call start() to spawn the thread, then
    poll get_events() to drain streamed events. When an interrupt event
    surfaces, call submit_hitl_response() to resume the parked graph.
    """

    # Sentinels pushed onto event_queue to mark terminal conditions. The UI
    # checks identity/equality against these to detect completion.
    RUN_DONE = "__RUN_DONE__"
    RUN_ERROR = "__RUN_ERROR__"

    def __init__(self, query: str) -> None:
        self.query = query
        # Stable thread_id for the checkpointer, shared across the initial
        # astream and every Command(resume=...) astream so the resume targets
        # the same parked run. Generated here, at construction.
        self.thread_id = uuid.uuid4().hex

        # thread -> UI: each item is a streamed event dict, the RUN_DONE
        # sentinel string, or a (RUN_ERROR, exception) tuple.
        self.event_queue: "queue.Queue[Any]" = queue.Queue()
        # UI -> thread: each item is a decision dict
        # {"decision": str, "retry_hint": str}.
        self.hitl_response_queue: "queue.Queue[dict]" = queue.Queue()

        self._thread: Optional[threading.Thread] = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Spawn the daemon thread that runs the agent. Idempotent-safe-ish:
        calling twice is a programming error (one runner == one run)."""
        if self._thread is not None:
            raise RuntimeError("AgentRunner.start() called twice on one runner")
        self._thread = threading.Thread(
            target=self._thread_main,
            name=f"agent-runner-{self.thread_id[:8]}",
            daemon=True,
        )
        self._thread.start()

    def is_alive(self) -> bool:
        """True while the agent thread is running (including while parked at a
        HITL interrupt waiting for a response)."""
        return self._thread is not None and self._thread.is_alive()

    # -- UI-facing queue access ---------------------------------------------

    def get_events(self) -> list[Any]:
        """Drain and return all currently-available items, non-blocking.

        Items are streamed event dicts, the RUN_DONE sentinel, or a
        (RUN_ERROR, exception) tuple. Returns [] if nothing is queued yet.
        """
        drained: list[Any] = []
        while True:
            try:
                drained.append(self.event_queue.get_nowait())
            except queue.Empty:
                break
        return drained

    def submit_hitl_response(self, decision: str, retry_hint: str = "") -> None:
        """Unblock the parked agent thread with the reviewer's HITL decision.

        decision is "approve" | "reject" | "retry"; retry_hint is only used on
        "retry". Resumes the graph via Command(resume=...) inside the thread.
        """
        self.hitl_response_queue.put(
            {"decision": decision, "retry_hint": retry_hint}
        )

    # -- thread internals ----------------------------------------------------

    def _thread_main(self) -> None:
        """Daemon-thread entry point: own event loop, run the async driver."""
        import asyncio

        # The thread creates its own loop via asyncio.run below, so the
        # selector-policy fix must be applied here, on this thread, first.
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        try:
            asyncio.run(self._drive())
        except Exception as exc:  # noqa: BLE001 — surface any driver failure to UI
            self.event_queue.put((self.RUN_ERROR, exc))
        else:
            self.event_queue.put(self.RUN_DONE)

    async def _drive(self) -> None:
        """Async driver: stream the graph, parking at interrupts for HITL.

        Mirrors agent/cli.py's resume loop. Each interrupt restarts the stream
        with Command(resume=...); this naturally handles a retry that
        re-triggers HITL on another sub-agent.
        """
        # Imported lazily inside the thread so module import (on the UI thread)
        # doesn't eagerly build the graph / touch the event loop policy.
        from langgraph.types import Command

        from agent.graph import compiled_graph
        from agent.state import new_state

        initial_state = new_state(self.query)
        config = {"configurable": {"thread_id": self.thread_id}}

        current_input: object = initial_state
        while True:
            interrupted = False
            async for event in compiled_graph.astream(current_input, config=config):
                if "__interrupt__" in event:
                    # Extract the plain payload dict from the Interrupt object
                    # and hand it to the UI, then block for the reviewer.
                    intr = event["__interrupt__"][0]
                    payload = getattr(intr, "value", intr)
                    self.event_queue.put({"__interrupt__": payload})

                    # Park the agent here until the UI submits a decision.
                    # Blocking get() is fine: we're on our own daemon thread.
                    response = self.hitl_response_queue.get()
                    current_input = Command(resume=response)
                    interrupted = True
                    break  # restart the stream to resume the parked graph

                # Normal node delta — forward as-is for the UI to format.
                self.event_queue.put(event)

            if not interrupted:
                break  # graph terminated without interrupting
