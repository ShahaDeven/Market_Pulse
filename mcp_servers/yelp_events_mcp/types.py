"""Pydantic v2 models for yelp-events MCP tool I/O.

These are the typed result shapes returned by the tools in ``tools.py``.
FastMCP serializes them to JSON for the agent. Per project convention, all data
crossing the MCP boundary is a Pydantic model (never a bare dict).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---- get_review_velocity -------------------------------------------------

class WeeklyBucket(BaseModel):
    """One week of review activity for a business."""

    week: str  # ISO date (Monday) of the week bucket, "YYYY-MM-DD"
    count: int
    avg_stars: float


class BaselineComparison(BaseModel):
    """Current window vs the equally-sized prior window, in reviews/week."""

    current_avg_per_week: float
    prior_avg_per_week: float
    pct_change: float


class ReviewVelocityResult(BaseModel):
    business_id: str
    business_name: Optional[str] = None
    weeks_requested: int
    weekly_buckets: list[WeeklyBucket] = Field(default_factory=list)
    baseline_comparison: Optional[BaselineComparison] = None
    confidence: Literal["high", "low"]


# ---- get_rating_delta ----------------------------------------------------

class RatingDeltaResult(BaseModel):
    business_id: str
    business_name: Optional[str] = None
    window_weeks: int
    current_avg_stars: float
    current_review_count: int
    prior_avg_stars: float
    prior_review_count: int
    delta: float  # current_avg_stars - prior_avg_stars (signed)
    pct_delta: float
    confidence: Literal["high", "low"]


# ---- find_businesses_by_category ----------------------------------------

class BusinessSearchHit(BaseModel):
    business_id: str
    name: str
    stars: float
    review_count: int


class BusinessSearchResult(BaseModel):
    city: str
    category: str
    results: list[BusinessSearchHit] = Field(default_factory=list)
    total_found: int  # count of all matches before `limit` was applied
