"""Structured memo models produced by the synthesis node.

Day 6 replaces the Day 4 ``synthesize_stub_node`` with a real synthesis step
backed by Claude Sonnet 4.5. The synthesis LLM consumes the gathered data
(``yelp_data`` / ``sec_data`` / ``fred_data`` on ``AgentState``) and emits a
``Memo`` via ``with_structured_output`` — a typed object, not a markdown blob.

This module defines ONLY the memo shape. The synthesis logic (the LLM call and
its integration into the graph) lands in [DAY 6.2] and [DAY 6.3]. There are no
graph/state imports, no LLM calls, and no DB calls here — purely data
definitions.

Why structured rather than free-form:
  - Predictable shape across runs.
  - Parseable for downstream consumers (the Streamlit UI in Week 3, the eval
    suite in Day 7).
  - Self-validating via Pydantic constraints (length limits, required fields).

Two invariants the synthesis LLM must respect, encoded here:
  - Every finding carries a citation identifying the sub-agent/tool that
    produced its data. This is what makes a finding "faithful" rather than
    hallucinated; Day 7 evals score citation accuracy.
  - Low-confidence work is first-class. A sub-agent that scored low confidence
    must surface in ``caveats`` — the LLM either reports the finding with a
    caveat or omits it and notes the gap. It cannot silently drop it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

ALLOWED_DATA_SOURCES = {"yelp", "sec", "fred"}


class Finding(BaseModel):
    """One finding in the memo.

    Each finding has a headline (the conclusion) and detail (the supporting
    explanation). Citation identifies which sub-agent and tool produced the
    data backing this finding.
    """

    headline: str = Field(
        min_length=1,
        max_length=200,
        description="One-sentence conclusion this finding supports.",
    )
    detail: str = Field(
        min_length=1,
        max_length=1000,
        description="The supporting explanation, citing specific data points.",
    )
    citation: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "Which sub-agent and tool produced the data backing this "
            "finding. Format: '<agent>:<tool_name>' (e.g., "
            "'yelp:find_businesses_by_category')."
        ),
    )

    @field_validator("citation")
    @classmethod
    def citation_has_colon(cls, value: str) -> str:
        """Require a ``<agent>:<tool_name>`` shape via a ':' separator.

        We deliberately do NOT enforce that ``<agent>`` is one of the known
        actors — the LLM may cite something slightly different (e.g.
        'yelp_agent:find_businesses'). We only require the colon shape so the
        citation isn't an unstructured free-form string. Both sides of the
        colon must be non-empty.
        """
        if ":" not in value:
            raise ValueError(
                "citation must be of the form '<agent>:<tool_name>' "
                f"(missing ':' separator): {value!r}"
            )
        agent, _, tool = value.partition(":")
        if not agent.strip() or not tool.strip():
            raise ValueError(
                "citation must have a non-empty agent and tool around the "
                f"':' separator: {value!r}"
            )
        return value


class Memo(BaseModel):
    """The structured memo produced by synthesis.

    Replaces the Day 4 stub. Captures: a 1-3 sentence executive summary, a
    list of concrete findings (each with citation), the data sources used,
    caveats about confidence or coverage, and a brief confidence summary tying
    everything back to the sub-agent scores.
    """

    executive_summary: str = Field(
        min_length=100,
        max_length=500,
        description=(
            "1-3 sentence overview of what this memo concludes. "
            "MUST be specific and data-grounded, not vague."
        ),
    )
    findings: list[Finding] = Field(
        min_length=1,
        max_length=10,
        description=(
            "Concrete findings, each with citation. Minimum 1, maximum 10. "
            "Each finding must cite which sub-agent/tool produced the data."
        ),
    )
    data_sources_used: list[str] = Field(
        min_length=1,
        description=(
            "Which sub-agents contributed data to this memo. "
            "Values: 'yelp', 'sec', 'fred'."
        ),
    )
    caveats: list[str] = Field(
        default_factory=list,
        description=(
            "Limitations of the analysis. MUST include any sub-agent that "
            "scored confidence < 0.7. Also include data freshness warnings "
            "(e.g., Yelp 2022 snapshot) when relevant. Each caveat <= 300 "
            "chars."
        ),
    )
    confidence_summary: str = Field(
        min_length=10,
        max_length=400,
        description=(
            "Brief paragraph noting the confidence scores of the underlying "
            "sub-agents and any reasons for uncertainty. Echo the judge's "
            "reasoning where relevant."
        ),
    )

    @field_validator("data_sources_used")
    @classmethod
    def data_sources_are_known(cls, value: list[str]) -> list[str]:
        """Each data source must be one of {'yelp', 'sec', 'fred'}."""
        invalid = [v for v in value if v not in ALLOWED_DATA_SOURCES]
        if invalid:
            allowed = ", ".join(sorted(ALLOWED_DATA_SOURCES))
            raise ValueError(
                f"data_sources_used contains unknown value(s) {invalid!r}; "
                f"allowed values are: {allowed}"
            )
        return value

    @field_validator("caveats")
    @classmethod
    def caveats_within_length(cls, value: list[str]) -> list[str]:
        """Enforce a per-element length cap of 300 chars.

        Pydantic's ``max_length`` on a ``list[str]`` constrains the list
        length, not the length of each element — so we enforce element-level
        length here.
        """
        for index, caveat in enumerate(value):
            if len(caveat) > 300:
                raise ValueError(
                    f"caveats[{index}] exceeds 300 chars ({len(caveat)} chars)."
                )
        return value


EXAMPLE_MEMO = Memo(
    executive_summary=(
        "Reading Terminal Market in Philadelphia is the most-reviewed "
        "coffee/tea business in the dataset with 5,721 reviews and a 4.5-star "
        "average. Recent review velocity is steady but the dataset is from 2022 "
        "so trends may have shifted since."
    ),
    findings=[
        Finding(
            headline=(
                "Reading Terminal Market has the highest review count among "
                "Philadelphia coffee businesses."
            ),
            detail=(
                "Among Philadelphia businesses tagged 'Coffee & Tea' with 50+ "
                "reviews, Reading Terminal Market leads with 5,721 reviews. The "
                "Franklin Fountain (2,062) and Cafe La Maude (1,485) are the "
                "next two."
            ),
            citation="yelp:find_businesses_by_category",
        ),
    ],
    data_sources_used=["yelp"],
    caveats=[
        "Yelp dataset is a 2022 snapshot; current state may differ.",
        "Single sub-agent ran; no SEC or macro context applied.",
    ],
    confidence_summary=(
        "Yelp confidence was 0.45 because the query also asked about review "
        "velocity trends, which were not gathered. The business identification "
        "portion is high-confidence."
    ),
)


if __name__ == "__main__":
    from pydantic import ValidationError

    # Happy path: the example memo constructs and serializes cleanly.
    print("EXAMPLE_MEMO:")
    print(EXAMPLE_MEMO.model_dump_json(indent=2))

    # Constraint check: too-short executive_summary must raise.
    try:
        Memo(
            executive_summary="too short",
            findings=EXAMPLE_MEMO.findings,
            data_sources_used=["yelp"],
            confidence_summary=EXAMPLE_MEMO.confidence_summary,
        )
    except ValidationError as e:
        print("\nValidation caught (executive_summary):", e.errors()[0]["msg"])

    # Constraint check: a citation without a ':' separator must raise.
    try:
        Finding(
            headline="A finding with a malformed citation.",
            detail="Some supporting detail goes here.",
            citation="yelp find_businesses_by_category",
        )
    except ValidationError as e:
        print("Validation caught (citation):", e.errors()[0]["msg"])

    # Constraint check: an unknown data source must raise.
    try:
        Memo(
            executive_summary=EXAMPLE_MEMO.executive_summary,
            findings=EXAMPLE_MEMO.findings,
            data_sources_used=["bloomberg"],
            confidence_summary=EXAMPLE_MEMO.confidence_summary,
        )
    except ValidationError as e:
        print("Validation caught (data_sources_used):", e.errors()[0]["msg"])

    # Constraint check: a caveat over 300 chars must raise.
    try:
        Memo(
            executive_summary=EXAMPLE_MEMO.executive_summary,
            findings=EXAMPLE_MEMO.findings,
            data_sources_used=["yelp"],
            caveats=["x" * 301],
            confidence_summary=EXAMPLE_MEMO.confidence_summary,
        )
    except ValidationError as e:
        print("Validation caught (caveats):", e.errors()[0]["msg"])
