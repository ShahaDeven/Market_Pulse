"""Integration tests for the yelp-events MCP tools.

These hit the real Postgres (DATABASE_URL). They skip gracefully when the
database is unreachable or empty, so they are safe to run in environments
without the dataset loaded. A high-review-count Philadelphia business is
discovered at runtime and reused as the subject for the time-based tools.
"""

from __future__ import annotations

import os

import pytest

from mcp_servers.yelp_events_mcp import tools
from mcp_servers.yelp_events_mcp.db import get_connection
from mcp_servers.yelp_events_mcp.types import (
    BusinessSearchResult,
    RatingDeltaResult,
    ReviewVelocityResult,
)


@pytest.fixture(scope="module")
def db_conn():
    """A live connection, or skip the whole module if unreachable."""
    if not os.environ.get("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set")
    try:
        conn = get_connection()
    except Exception as exc:  # noqa: BLE001 - any connect failure -> skip
        pytest.skip(f"Postgres unreachable: {exc}")
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def philly_business_id(db_conn) -> str:
    """The most-reviewed Philadelphia business."""
    with db_conn.cursor() as cur:
        cur.execute(
            """
            SELECT business_id
            FROM businesses
            WHERE LOWER(city) = LOWER(%s)
              AND review_count > 0
            ORDER BY review_count DESC
            LIMIT 1
            """,
            ("Philadelphia",),
        )
        row = cur.fetchone()
    if row is None:
        pytest.skip("No Philadelphia businesses with reviews found")
    return row[0]


def test_get_review_velocity(philly_business_id):
    result = tools.get_review_velocity(philly_business_id, weeks=12)
    assert isinstance(result, ReviewVelocityResult)
    assert result.business_id == philly_business_id
    assert result.weeks_requested == 12
    assert result.confidence in ("high", "low")
    # Buckets should be ordered and well-formed.
    weeks = [b.week for b in result.weekly_buckets]
    assert weeks == sorted(weeks)
    for b in result.weekly_buckets:
        assert b.count >= 1
        assert 1.0 <= b.avg_stars <= 5.0


def test_get_rating_delta(philly_business_id):
    result = tools.get_rating_delta(philly_business_id, window_weeks=12)
    assert isinstance(result, RatingDeltaResult)
    assert result.business_id == philly_business_id
    assert result.window_weeks == 12
    assert result.confidence in ("high", "low")
    # delta must be internally consistent with the two averages.
    assert result.delta == pytest.approx(
        result.current_avg_stars - result.prior_avg_stars, abs=1e-3
    )


def test_find_businesses_by_category():
    result = tools.find_businesses_by_category(
        city="Philadelphia", category="Restaurants", min_reviews=50, limit=10
    )
    assert isinstance(result, BusinessSearchResult)
    assert result.city == "Philadelphia"
    assert result.category == "Restaurants"
    assert len(result.results) <= 10
    assert result.total_found >= len(result.results)
    # Results must respect the filters and ordering.
    counts = [hit.review_count for hit in result.results]
    assert counts == sorted(counts, reverse=True)
    for hit in result.results:
        assert hit.review_count >= 50


def test_unknown_business_returns_empty_not_error():
    result = tools.get_review_velocity("does-not-exist-000000", weeks=12)
    assert isinstance(result, ReviewVelocityResult)
    assert result.weekly_buckets == []
    assert result.baseline_comparison is None
    assert result.confidence == "low"


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        tools.get_review_velocity("abc", weeks=0)
    with pytest.raises(ValueError):
        tools.get_rating_delta("", window_weeks=12)
    with pytest.raises(ValueError):
        tools.find_businesses_by_category(city="X", category="Y", limit=0)
