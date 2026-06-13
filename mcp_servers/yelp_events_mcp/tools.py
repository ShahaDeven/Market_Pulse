"""MCP tool implementations for the yelp-events server.

Three read-only tools over the local Yelp Open Dataset in Postgres:

  - get_review_velocity         — weekly review counts + velocity vs baseline
  - get_rating_delta            — avg-star change between two windows
  - find_businesses_by_category — category/city search

IMPORTANT — time windows are anchored to the data, not wall-clock time.
The Yelp Open Dataset is static and ends in 2022, so "last N weeks" relative to
``now()`` would always be empty. For the two time-based tools we therefore
anchor each window to the MOST RECENT review for the given business
(``MAX(reviews.date)``): the "current" window is the N weeks ending at that
business's latest review, and the "prior" window is the N weeks before that.
This makes velocity/delta meaningful on frozen historical data. Reviews exist
only for Philadelphia and Tampa (ADR-007).
"""

from __future__ import annotations

from psycopg.rows import dict_row

from .db import get_connection
from .types import (
    BaselineComparison,
    BusinessSearchHit,
    BusinessSearchResult,
    RatingDeltaResult,
    ReviewVelocityResult,
    WeeklyBucket,
)

# A review window with fewer than this many reviews in the comparison/prior
# window is statistically too thin to trust — we drop confidence to "low".
MIN_REVIEWS_FOR_CONFIDENCE = 10


def _clean_business_id(business_id: str) -> str:
    """Validate and normalize a business_id. Raises ValueError if malformed."""
    if not isinstance(business_id, str) or not business_id.strip():
        raise ValueError("business_id must be a non-empty string")
    return business_id.strip()


def get_review_velocity(business_id: str, weeks: int = 12) -> ReviewVelocityResult:
    """Weekly review volume for a business over the last N weeks of its data,
    with a velocity comparison against the equally-sized prior window.

    Windows are anchored to the business's most recent review (the dataset is
    static), not to today's date. Returns per-week review counts and average
    star ratings, plus a baseline comparison of reviews-per-week (current vs
    prior window). If the prior window has fewer than 10 reviews the comparison
    is omitted and confidence is "low". Unknown businesses return an empty
    result rather than an error.

    NOTE: The Yelp Open Dataset is a 2022 snapshot. The 'weeks' parameter
    measures from the most recent review for the specific business, not
    from today. The window is per-business deterministic regardless of
    query time.

    Args:
        business_id: Yelp business id (22-char string).
        weeks: window length in weeks (must be > 0; default 12).
    """
    business_id = _clean_business_id(business_id)
    if weeks <= 0:
        raise ValueError("weeks must be greater than 0")

    conn = get_connection()
    try:
        with conn.cursor(row_factory=dict_row) as cur:
            # Business name + anchor (latest review date) in one shot.
            cur.execute(
                """
                SELECT b.name AS business_name, MAX(r.date) AS anchor
                FROM businesses b
                LEFT JOIN reviews r ON r.business_id = b.business_id
                WHERE b.business_id = %(bid)s
                GROUP BY b.name
                """,
                {"bid": business_id},
            )
            head = cur.fetchone()

            # Unknown business, or a known business with zero reviews: empty.
            if head is None or head["anchor"] is None:
                return ReviewVelocityResult(
                    business_id=business_id,
                    business_name=head["business_name"] if head else None,
                    weeks_requested=weeks,
                    weekly_buckets=[],
                    baseline_comparison=None,
                    confidence="low",
                )

            params = {"bid": business_id, "anchor": head["anchor"], "w": weeks,
                      "w2": weeks * 2}

            # Weekly buckets over the current window. Uses idx_reviews_business_id_date.
            cur.execute(
                """
                SELECT to_char(date_trunc('week', date), 'YYYY-MM-DD') AS week,
                       COUNT(*) AS count,
                       AVG(stars)::float AS avg_stars
                FROM reviews
                WHERE business_id = %(bid)s
                  AND date >  %(anchor)s - make_interval(weeks => %(w)s)
                  AND date <= %(anchor)s
                GROUP BY 1
                ORDER BY 1
                """,
                params,
            )
            buckets = [
                WeeklyBucket(week=r["week"], count=r["count"],
                             avg_stars=round(r["avg_stars"], 3))
                for r in cur.fetchall()
            ]

            # Prior window total, for the baseline comparison.
            cur.execute(
                """
                SELECT COUNT(*) AS prior_total
                FROM reviews
                WHERE business_id = %(bid)s
                  AND date >  %(anchor)s - make_interval(weeks => %(w2)s)
                  AND date <= %(anchor)s - make_interval(weeks => %(w)s)
                """,
                params,
            )
            prior_total = cur.fetchone()["prior_total"]
    finally:
        conn.close()

    current_total = sum(b.count for b in buckets)

    if prior_total < MIN_REVIEWS_FOR_CONFIDENCE:
        baseline = None
        confidence = "low"
    else:
        current_per_week = current_total / weeks
        prior_per_week = prior_total / weeks
        pct_change = (current_per_week - prior_per_week) / prior_per_week * 100
        baseline = BaselineComparison(
            current_avg_per_week=round(current_per_week, 3),
            prior_avg_per_week=round(prior_per_week, 3),
            pct_change=round(pct_change, 2),
        )
        confidence = "high"

    return ReviewVelocityResult(
        business_id=business_id,
        business_name=head["business_name"],
        weeks_requested=weeks,
        weekly_buckets=buckets,
        baseline_comparison=baseline,
        confidence=confidence,
    )


def get_rating_delta(business_id: str, window_weeks: int = 12) -> RatingDeltaResult:
    """Change in average star rating between two consecutive windows.

    Compares the average review stars in the current window (the N weeks ending
    at the business's most recent review) against the prior window (the N weeks
    before that). Windows are anchored to the data, not today's date. Confidence
    is "low" if either window has fewer than 10 reviews. Unknown businesses
    return a zeroed result rather than an error.

    NOTE: The Yelp Open Dataset is a 2022 snapshot. The 'window_weeks'
    parameter measures from the most recent review for the specific
    business, not from today. The window is per-business deterministic
    regardless of query time.

    Args:
        business_id: Yelp business id (22-char string).
        window_weeks: window length in weeks (must be > 0; default 12).
    """
    business_id = _clean_business_id(business_id)
    if window_weeks <= 0:
        raise ValueError("window_weeks must be greater than 0")

    conn = get_connection()
    try:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT b.name AS business_name, MAX(r.date) AS anchor
                FROM businesses b
                LEFT JOIN reviews r ON r.business_id = b.business_id
                WHERE b.business_id = %(bid)s
                GROUP BY b.name
                """,
                {"bid": business_id},
            )
            head = cur.fetchone()

            if head is None or head["anchor"] is None:
                return RatingDeltaResult(
                    business_id=business_id,
                    business_name=head["business_name"] if head else None,
                    window_weeks=window_weeks,
                    current_avg_stars=0.0, current_review_count=0,
                    prior_avg_stars=0.0, prior_review_count=0,
                    delta=0.0, pct_delta=0.0, confidence="low",
                )

            params = {"bid": business_id, "anchor": head["anchor"],
                      "w": window_weeks, "w2": window_weeks * 2}

            # Both windows in a single pass via FILTER aggregates.
            cur.execute(
                """
                SELECT
                  AVG(stars) FILTER (
                    WHERE date >  %(anchor)s - make_interval(weeks => %(w)s)
                      AND date <= %(anchor)s)::float                 AS current_avg,
                  COUNT(*) FILTER (
                    WHERE date >  %(anchor)s - make_interval(weeks => %(w)s)
                      AND date <= %(anchor)s)                        AS current_count,
                  AVG(stars) FILTER (
                    WHERE date >  %(anchor)s - make_interval(weeks => %(w2)s)
                      AND date <= %(anchor)s - make_interval(weeks => %(w)s))::float
                                                                     AS prior_avg,
                  COUNT(*) FILTER (
                    WHERE date >  %(anchor)s - make_interval(weeks => %(w2)s)
                      AND date <= %(anchor)s - make_interval(weeks => %(w)s))
                                                                     AS prior_count
                FROM reviews
                WHERE business_id = %(bid)s
                """,
                params,
            )
            row = cur.fetchone()
    finally:
        conn.close()

    current_avg = row["current_avg"] or 0.0
    prior_avg = row["prior_avg"] or 0.0
    current_count = row["current_count"]
    prior_count = row["prior_count"]

    delta = current_avg - prior_avg
    pct_delta = (delta / prior_avg * 100) if prior_avg else 0.0

    confidence = (
        "high"
        if current_count >= MIN_REVIEWS_FOR_CONFIDENCE
        and prior_count >= MIN_REVIEWS_FOR_CONFIDENCE
        else "low"
    )

    return RatingDeltaResult(
        business_id=business_id,
        business_name=head["business_name"],
        window_weeks=window_weeks,
        current_avg_stars=round(current_avg, 3),
        current_review_count=current_count,
        prior_avg_stars=round(prior_avg, 3),
        prior_review_count=prior_count,
        delta=round(delta, 3),
        pct_delta=round(pct_delta, 2),
        confidence=confidence,
    )


def find_businesses_by_category(
    city: str,
    category: str,
    min_reviews: int = 50,
    limit: int = 20,
) -> BusinessSearchResult:
    """Find businesses in a city that carry a given Yelp category.

    Matches city case-insensitively and requires the business's normalized
    categories_array column to contain `category` (exact element match).
    The categories_array column was populated by Migration 002 from Yelp's
    comma-separated categories string (see ADR-009). Filters to businesses
    with at least `min_reviews` reviews and returns the top `limit` ordered
    by review_count descending.

    Args:
        city: city name (case-insensitive), e.g. "Philadelphia".
        category: exact Yelp category string, e.g. "Coffee & Tea".
        min_reviews: minimum review_count to include (must be >= 0; default 50).
        limit: max rows to return (must be > 0; default 20).
    """
    if not isinstance(city, str) or not city.strip():
        raise ValueError("city must be a non-empty string")
    if not isinstance(category, str) or not category.strip():
        raise ValueError("category must be a non-empty string")
    if min_reviews < 0:
        raise ValueError("min_reviews must be >= 0")
    if limit <= 0:
        raise ValueError("limit must be greater than 0")

    city = city.strip()
    category = category.strip()

    conn = get_connection()
    try:
        with conn.cursor(row_factory=dict_row) as cur:
            # categories @> '["Category"]' uses idx_businesses_categories_gin.
            # COUNT(*) OVER() gives the pre-limit total in the same query.
            cur.execute(
                """
                SELECT business_id,
                       name,
                       stars::float AS stars,
                       review_count,
                       COUNT(*) OVER() AS total_found
                FROM businesses
                WHERE LOWER(city) = LOWER(%(city)s)
                  AND %(cat)s = ANY(categories_array)
                  AND review_count >= %(minr)s
                ORDER BY review_count DESC
                LIMIT %(lim)s
                """,
                {"city": city, "cat": category,
                 "minr": min_reviews, "lim": limit},
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    total_found = rows[0]["total_found"] if rows else 0
    hits = [
        BusinessSearchHit(
            business_id=r["business_id"],
            name=r["name"],
            stars=r["stars"],
            review_count=r["review_count"],
        )
        for r in rows
    ]

    return BusinessSearchResult(
        city=city,
        category=category,
        results=hits,
        total_found=total_found,
    )
