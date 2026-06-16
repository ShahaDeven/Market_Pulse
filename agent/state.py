"""LangGraph state schema for MarketPulse.

Design choices:
- TypedDict, not Pydantic. LangGraph reduces state with dict-merge semantics
  and expects TypedDict shape. Pydantic would force every node to construct
  full model instances on each update, which is verbose and slow.
- Each sub-agent has its own data slot (yelp_data, sec_data, fred_data) so
  future parallel execution can populate them concurrently without races.
- supervisor_log is a list of plain strings: the supervisor writes one
  sentence per routing decision. Useful for debugging the routing logic and
  for the audit log later.
- tool_calls_made tracks every individual tool invocation across all
  sub-agents. The audit log in Day 5 reads from this.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, TypedDict
from operator import add
import uuid


# Type alias for the routing target. "synthesize" means the supervisor has
# decided enough data has been collected; "done" is the terminal state.
AgentTarget = Literal["yelp", "sec", "fred", "synthesize", "done"]


class AgentState(TypedDict, total=False):
    """The state object that flows through the LangGraph graph.

    total=False means all fields are optional at runtime — nodes only set
    the fields they need to update. Reducers handle merging.
    """

    # Identity
    query_id: str
    user_query: str

    # Routing
    target_agent: Optional[AgentTarget]

    # Sub-agent data slots (each agent writes only to its own slot)
    yelp_data: Optional[dict]
    sec_data: Optional[dict]
    fred_data: Optional[dict]

    # Append-only tracking — use the `add` reducer so concurrent updates
    # from different nodes concatenate instead of overwriting
    tool_calls_made: Annotated[list[dict], add]
    supervisor_log: Annotated[list[str], add]
    errors: Annotated[list[dict], add]

    # Reserved for Day 6 synthesis
    final_memo: Optional[str]

    # --- Human-in-the-loop (HITL) ---
    # When any sub-agent's confidence is below the threshold (0.7), the graph
    # pauses for human review. These fields carry the review state.

    # The reviewer's decision, set by the CLI when it resumes the paused graph.
    hitl_decision: Optional[Literal["approve", "reject", "retry"]]

    # Optional free-text hint the reviewer supplies on a "retry" decision; the
    # supervisor reads it to inform re-routing (Option Y / informed retry).
    hitl_retry_hint: Optional[str]

    # Append-only record of every sub-agent that has ever triggered HITL, e.g.
    # {"agent": "yelp", "confidence": 0.45, "reason": "..."}. The `add` reducer
    # accumulates these across iterations (same pattern as supervisor_log).
    # check_confidence_node dedupes against this so an agent triggers HITL once.
    hitl_low_confidence_agents: Annotated[list[dict], add]

    # Transient routing signal (last-write-wins, NOT append): set by
    # check_confidence_node each pass to tell _route_from_check_confidence
    # whether this pass flagged a NEW low-confidence agent. Not part of the
    # durable HITL record — it exists purely to drive the conditional edge.
    hitl_pending: Optional[bool]


def new_state(user_query: str) -> AgentState:
    """Create a fresh state for a new query.

    The query_id is a UUID hex string used to correlate this query across
    logs, the audit trail, and any HITL approval flow.
    """
    return AgentState(
        query_id=uuid.uuid4().hex,
        user_query=user_query,
        target_agent=None,
        yelp_data=None,
        sec_data=None,
        fred_data=None,
        tool_calls_made=[],
        supervisor_log=[],
        errors=[],
        final_memo=None,
        hitl_decision=None,
        hitl_retry_hint=None,
        hitl_low_confidence_agents=[],
        hitl_pending=None,
    )
