"""LLM-as-judge scorer for MarketPulse evals.

Reads a results JSON file produced by evals/runner.py and scores each query
across three binary (pass/fail) dimensions using Claude Sonnet 4.5 as the
judge — the same model family used for synthesis:

  - faithfulness          : are the memo's findings grounded in the data the
                            sub-agents actually gathered (no hallucinated
                            numbers/facts)?
  - citation_correctness  : is every finding's citation a valid
                            '<agent>:<tool>' that names a tool actually called?
  - confidence_calibration: does the memo honestly surface low sub-agent
                            confidence rather than overclaiming?

The scoring is SEMANTIC, not literal: the golden_memo is a reference example,
not ground truth to diff against. The judge evaluates the actual_memo against
the query and the data the agent gathered.

Usage:
    uv run python -m evals.scorer                       # score latest results
    uv run python -m evals.scorer --results <path.json> # score a specific file

Outputs a scored JSON file at evals/scored/scored_<timestamp>.json and prints
a summary. Judge calls are non-deterministic across runs (temperature=0 and
constrained prompts reduce, but do not eliminate, variance) — this is a
snapshot eval, not a CI gate, so no retries or N-shot averaging are applied.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import structlog
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

# Load .env so ANTHROPIC_API_KEY is available to the judge LLM. The runner gets
# this for free via agent/__init__.py's load_dotenv(); the scorer never imports
# the agent package, so it loads the env itself (same project convention).
load_dotenv()

log = structlog.get_logger("evals.scorer")

RESULTS_DIR = Path("evals/results")
SCORED_DIR = Path("evals/scored")

DIMENSIONS = ("faithfulness", "citation_correctness", "confidence_calibration")

# Sub-agent confidence at or above this is "confident"; below it MUST surface
# as a caveat or in the confidence_summary. Mirrors the agent's HITL threshold.
_LOW_CONFIDENCE_THRESHOLD = 0.7


class ScoreVerdict(BaseModel):
    """One judge's binary verdict on a single dimension."""

    passed: bool
    # No hard max_length: the dimension prompts ask the judge for a concise
    # (<= 300 char) reason, but the judge occasionally exceeds that on
    # multi-finding memos. A structured-output ValidationError on a purely
    # explanatory field would abort the entire eval, so we keep this tolerant
    # and rely on the prompt (and max_tokens) to bound length.
    reason: str


# Module-level judge LLM. Sonnet 4.5 (per synthesis), temperature=0 for the
# most stable judgments we can get from an LLM judge.
_judge_llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0,
    max_tokens=1024,
)
_structured_judge = _judge_llm.with_structured_output(ScoreVerdict)


async def _run_judge(system_prompt: str, user_content: str) -> ScoreVerdict:
    """Invoke the judge LLM and return its structured verdict.

    No retries: if the call fails we let the exception surface so judge
    variance/failures are debuggable rather than silently swallowed.
    """
    return await _structured_judge.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]
    )


def _findings(actual_memo: dict | None) -> list[dict]:
    """Return the memo's findings list, defensively."""
    if not isinstance(actual_memo, dict):
        return []
    return actual_memo.get("findings") or []


# --------------------------------------------------------------------------
# Dimension judges
# --------------------------------------------------------------------------

_FAITHFULNESS_SYSTEM = """You are a strict evaluator of equity-research memos. \
You judge FAITHFULNESS: whether a memo's findings are grounded in the data the \
agent's tools actually returned.

Rules for your verdict:
- Every specific number in a finding (e.g. '5,721 reviews', '3.9%') must be \
traceable to the content of a tool's output.
- Every named entity (business, FRED series, SEC filing) in a finding must \
appear in tool output content.
- A finding must NOT introduce numbers or facts that the tools did not return.

SCOPE — what FAITHFULNESS does NOT judge (this is a rubric boundary, not \
leniency): this dimension judges ONLY whether the findings that are PRESENT are \
grounded. It does NOT judge coverage or completeness. A memo that omits a \
finding for part of the query is NOT a faithfulness failure — for example, when \
a sub-agent ran but returned no usable data and therefore no finding cites it. \
A missing, absent, or omitted finding is OUT OF SCOPE here. Do NOT FAIL because \
a query component lacks a finding; whether all parts of the query were covered \
is a separate concern judged elsewhere. Judge only the grounding of the \
findings that DO appear.

PASS only if ALL findings that appear are grounded in the provided tool \
outputs. FAIL only if a finding that appears contains a hallucinated specific \
number or fact (NOT because some expected finding is missing). Give a concise \
reason (<= 300 chars) citing the specific finding if you FAIL."""

_CITATION_SYSTEM = """You are a strict evaluator of equity-research memos. You \
judge CITATION CORRECTNESS: whether every finding's citation is well-formed and \
names a tool that was actually called.

Rules for your verdict:
- A citation must have the shape '<agent>:<tool_name>' (a single colon, both \
sides non-empty).
- The <agent> portion must be one of: yelp, sec, fred.
- The <tool_name> portion must match the 'tool' field of a tool call that was \
actually made for this query (see the provided tool_calls).

PASS only if EVERY finding's citation is valid AND names a real tool call. FAIL \
if any citation is malformed, uses an unknown agent, or cites a tool that was \
not called. Give a concise reason (<= 300 chars) naming the offending citation \
if you FAIL."""

_CALIBRATION_SYSTEM = """You are a strict evaluator of equity-research memos. \
You judge CONFIDENCE CALIBRATION: whether the memo honestly reflects the \
sub-agents' reported confidence scores.

Rules for your verdict:
- If ANY sub-agent confidence is below 0.7, the memo's caveats OR \
confidence_summary MUST acknowledge that uncertainty (e.g. name the agent, or \
note the limitation that drove the low score). A 0.5 'judgment_failed' \
fallback counts as low confidence and must surface.
- The confidence_summary must NOT claim higher overall confidence than the \
sub-agent scores justify.
- A sub-agent confidence of null means that agent did not run; it does not \
require a caveat.

PASS only if all low confidence is properly surfaced and nothing is \
overclaimed. FAIL if low confidence is hidden or the summary overclaims. Give a \
concise reason (<= 300 chars)."""


async def judge_faithfulness(
    query: str, actual_memo: dict | None, tool_calls: list[dict]
) -> ScoreVerdict:
    """Judge whether the memo's findings are grounded in actual tool output."""
    findings = _findings(actual_memo)
    # Only the fields the judge needs to trace numbers/entities to their source.
    tool_evidence = [
        {
            "agent": c.get("agent"),
            "tool": c.get("tool"),
            "input": c.get("input"),
            "output_preview": c.get("output_preview"),
        }
        for c in tool_calls
    ]
    user_content = (
        f"USER QUERY:\n{query}\n\n"
        f"MEMO FINDINGS (headline / detail / citation):\n"
        f"{json.dumps(findings, indent=2, default=str)}\n\n"
        f"TOOL CALLS THE AGENT ACTUALLY MADE (with output previews):\n"
        f"{json.dumps(tool_evidence, indent=2, default=str)}\n\n"
        "Are all findings grounded in the tool outputs above? Decide PASS/FAIL."
    )
    return await _run_judge(_FAITHFULNESS_SYSTEM, user_content)


async def judge_citation(
    actual_memo: dict | None, tool_calls: list[dict]
) -> ScoreVerdict:
    """Judge whether every finding cites a real, well-formed '<agent>:<tool>'."""
    findings = _findings(actual_memo)
    citations = [
        {"headline": f.get("headline"), "citation": f.get("citation")}
        for f in findings
    ]
    called_tools = [
        {"agent": c.get("agent"), "tool": c.get("tool")} for c in tool_calls
    ]
    user_content = (
        f"FINDING CITATIONS:\n{json.dumps(citations, indent=2, default=str)}\n\n"
        f"TOOLS ACTUALLY CALLED (agent + tool name):\n"
        f"{json.dumps(called_tools, indent=2, default=str)}\n\n"
        "Is every citation well-formed and matched to a real tool call? "
        "Decide PASS/FAIL."
    )
    return await _run_judge(_CITATION_SYSTEM, user_content)


async def judge_confidence_calibration(
    actual_memo: dict | None, confidences: dict
) -> ScoreVerdict:
    """Judge whether the memo honestly surfaces low sub-agent confidence."""
    caveats = actual_memo.get("caveats") if isinstance(actual_memo, dict) else None
    confidence_summary = (
        actual_memo.get("confidence_summary") if isinstance(actual_memo, dict) else None
    )
    low = {
        a: c
        for a, c in (confidences or {}).items()
        if isinstance(c, (int, float)) and c < _LOW_CONFIDENCE_THRESHOLD
    }
    user_content = (
        f"SUB-AGENT CONFIDENCE SCORES (null = agent did not run):\n"
        f"{json.dumps(confidences, indent=2, default=str)}\n\n"
        f"Sub-agents BELOW the 0.7 threshold that MUST be surfaced: "
        f"{json.dumps(low, default=str) or '{}'}\n\n"
        f"MEMO CAVEATS:\n{json.dumps(caveats or [], indent=2, default=str)}\n\n"
        f"MEMO CONFIDENCE SUMMARY:\n{confidence_summary or '(none)'}\n\n"
        "Does the memo honestly surface low confidence without overclaiming? "
        "Decide PASS/FAIL."
    )
    return await _run_judge(_CALIBRATION_SYSTEM, user_content)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def _find_latest_results() -> Path:
    """Return the most recently modified eval_*.json in evals/results/."""
    candidates = list(RESULTS_DIR.glob("eval_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No results files found in {RESULTS_DIR}. Run evals.runner first."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


async def _score_one(result: dict) -> dict:
    """Score a single result row across the three dimensions.

    Skipped (run_error) rows return None dimensions. A row that completed but
    produced no memo is scored as a failure on all dimensions (a missing memo
    cannot be faithful, cited, or calibrated).
    """
    name = result.get("name")
    query = result.get("query", "")
    run_error = result.get("run_metadata", {}).get("run_error")

    if run_error is not None:
        log.info("scorer_skip_run_error", name=name, error=run_error)
        return {
            "name": name,
            "query": query,
            "dimensions": {dim: None for dim in DIMENSIONS},
            "overall_passed": None,
            "skipped": True,
        }

    actual_memo = result.get("actual_memo")
    tool_calls = result.get("tool_calls") or []
    confidences = result.get("sub_agent_confidences") or {}

    if not actual_memo:
        reason = "No memo was produced for this run (nothing to score)."
        dims = {dim: {"passed": False, "reason": reason} for dim in DIMENSIONS}
        return {
            "name": name,
            "query": query,
            "dimensions": dims,
            "overall_passed": False,
            "skipped": False,
        }

    faithfulness = await judge_faithfulness(query, actual_memo, tool_calls)
    citation = await judge_citation(actual_memo, tool_calls)
    calibration = await judge_confidence_calibration(actual_memo, confidences)

    dims = {
        "faithfulness": faithfulness.model_dump(),
        "citation_correctness": citation.model_dump(),
        "confidence_calibration": calibration.model_dump(),
    }
    overall_passed = all(d["passed"] for d in dims.values())
    return {
        "name": name,
        "query": query,
        "dimensions": dims,
        "overall_passed": overall_passed,
        "skipped": False,
    }


def _print_summary(
    run_id: str, source_path: Path, scored: list[dict], agg: dict
) -> None:
    """Print the human-readable scoring summary."""
    bar = "=" * 60
    pad = 23  # widest dimension label ('confidence_calibration') + breathing room

    print(bar)
    print(f"EVAL SCORES (run_id: {run_id}, source: {source_path})")
    print(bar)

    for row in scored:
        print(f"\n{row['name']}:")
        if row["skipped"]:
            err = "run_error"
            print(f"  {'(skipped)':<{pad}}{err}")
            continue
        for dim in DIMENSIONS:
            verdict = row["dimensions"][dim]
            status = "PASS" if verdict["passed"] else "FAIL"
            print(f"  {dim:<{pad}}{status}  {verdict['reason']}")
        overall = "PASS" if row["overall_passed"] else "FAIL"
        print(f"  {'overall':<{pad}}{overall}")

    print(f"\n{bar}")
    print(
        f"AGGREGATE: {agg['passed_checks']}/{agg['total_checks']} "
        f"dimension checks passed ({agg['pass_rate_pct']}%)"
    )
    print(f"queries with overall PASS: {agg['n_passed']}/{agg['n_scored']}")
    print(f"queries skipped (run_error): {agg['n_skipped']}")
    print(bar)


async def score_all(results_path: Path | None = None) -> dict:
    """Score every query in a runner results file, print and persist the report.

    Returns the scored report dict (also written to evals/scored/).
    """
    source_path = Path(results_path) if results_path else _find_latest_results()
    data = json.loads(source_path.read_text(encoding="utf-8"))
    run_id = data.get("run_id")
    results = data.get("results", [])

    log.info("scorer_start", source=str(source_path), n_queries=len(results))

    scored = [await _score_one(result) for result in results]

    # Aggregate across non-skipped queries (3 checks each).
    n_skipped = sum(1 for r in scored if r["skipped"])
    n_scored = len(scored) - n_skipped
    total_checks = n_scored * len(DIMENSIONS)
    passed_checks = sum(
        1
        for r in scored
        if not r["skipped"]
        for v in r["dimensions"].values()
        if v and v["passed"]
    )
    n_passed = sum(1 for r in scored if r["overall_passed"] is True)
    pass_rate = (passed_checks / total_checks) if total_checks else 0.0

    agg = {
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "pass_rate": pass_rate,
        "pass_rate_pct": round(pass_rate * 100),
        "n_passed": n_passed,
        "n_scored": n_scored,
        "n_skipped": n_skipped,
    }

    _print_summary(run_id, source_path, scored, agg)

    # Strip the internal 'skipped' helper flag from the persisted rows; the
    # null dimensions already encode "skipped" in the output schema.
    out_results = [
        {k: v for k, v in row.items() if k != "skipped"} for row in scored
    ]
    output = {
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "source_results_path": str(source_path),
        "source_run_id": run_id,
        "n_queries": len(results),
        "n_passed": n_passed,
        "n_skipped": n_skipped,
        "aggregate_pass_rate": pass_rate,
        "results": out_results,
    }

    SCORED_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = SCORED_DIR / f"scored_{timestamp}.json"
    output_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nScored report written to: {output_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Score MarketPulse eval results.")
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="Path to a runner results JSON. Default: most recent in "
        "evals/results/.",
    )
    args = parser.parse_args()

    asyncio.run(score_all(results_path=args.results))


if __name__ == "__main__":
    main()
