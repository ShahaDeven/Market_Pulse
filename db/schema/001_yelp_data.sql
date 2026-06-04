-- ============================================================================
-- 001_yelp_data.sql — Yelp Open Dataset tables
-- ============================================================================
--
-- Schema for the 3 Yelp Open Dataset files we load: business, review, tip.
-- (The dataset also ships user.json and checkin.json, which we do not load.)
--
-- Source: Yelp Open Dataset Documentation (Last Updated: July 7, 2023)
--   https://www.yelp.com/dataset/documentation/main
--
-- Two quirks worth knowing before you read further:
--
--   1. "postal code" space in the source JSON.
--      business.json names the field "postal code" WITH A SPACE, not an
--      underscore. The Python loader renames it to postal_code before insert,
--      so the column here is postal_code. Don't be surprised by the mismatch
--      between source key and column name.
--
--   2. Unenforced foreign keys.
--      reviews.business_id and tips.business_id logically reference
--      businesses.business_id, and reviews.user_id / tips.user_id reference
--      user.json (which we don't load). We deliberately declare NO FK
--      constraints: the loader bulk-inserts files in an arbitrary order, and
--      enforced FKs would fail on load-order / missing-parent issues. Referential
--      integrity is the loader's responsibility, not the database's.
--
-- All string fields use TEXT (not VARCHAR). All date fields use
-- TIMESTAMP WITH TIME ZONE (source is YYYY-MM-DD strings; the loader parses).
-- All statements are idempotent (IF NOT EXISTS) so re-running init is safe.
-- ============================================================================


-- ----------------------------------------------------------------------------
-- businesses  (from business.json)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS businesses (
    business_id   TEXT PRIMARY KEY,          -- 22-char unique string
    name          TEXT,
    address       TEXT,                      -- full street address
    city          TEXT,
    state         TEXT,                      -- 2-char state/province code
    -- Source JSON key is "postal code" (with a space); loader renames to this.
    postal_code   TEXT,
    latitude      DOUBLE PRECISION,
    longitude     DOUBLE PRECISION,
    -- Business rating is a float rounded to half-stars by Yelp (e.g. 4.5).
    stars         REAL,
    review_count  INTEGER,
    -- is_open is an integer flag in the source (0 or 1), NOT a boolean.
    is_open       SMALLINT,
    -- Variable nested object; may contain nested objects (e.g. BusinessParking).
    attributes    JSONB,
    -- Array of category strings, e.g. ["Mexican", "Burgers", "Gastropubs"].
    -- Stored as JSONB (not a delimited string) and indexed with GIN below.
    categories    JSONB,
    -- Object mapping day name -> hours range string, e.g. {"Monday": "9:0-17:0"}.
    hours         JSONB
);


-- ----------------------------------------------------------------------------
-- reviews  (from review.json)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reviews (
    review_id     TEXT PRIMARY KEY,
    -- References user.json (not loaded). No FK constraint.
    user_id       TEXT,
    -- Logical FK to businesses.business_id. NOT enforced — see header note (2).
    business_id   TEXT NOT NULL,
    -- Review rating is an integer 1-5 (unlike businesses.stars, which is REAL).
    stars         SMALLINT,
    date          TIMESTAMP WITH TIME ZONE,
    text          TEXT,
    useful        INTEGER,
    funny         INTEGER,
    cool          INTEGER
);


-- ----------------------------------------------------------------------------
-- tips  (from tip.json)
-- ----------------------------------------------------------------------------
-- tip.json has no natural primary key, so we synthesize one with BIGSERIAL.
CREATE TABLE IF NOT EXISTS tips (
    id                BIGSERIAL PRIMARY KEY,
    text              TEXT,
    date              TIMESTAMP WITH TIME ZONE,
    compliment_count  INTEGER,
    -- Logical FK to businesses.business_id. NOT enforced — see header note (2).
    business_id       TEXT,
    -- References user.json (not loaded). No FK constraint.
    user_id           TEXT
);


-- ============================================================================
-- Indexes  (all IF NOT EXISTS for idempotency)
-- ============================================================================

-- businesses: geographic queries (state then city).
CREATE INDEX IF NOT EXISTS idx_businesses_state_city
    ON businesses (state, city);

-- businesses: category lookups via the @> containment operator.
CREATE INDEX IF NOT EXISTS idx_businesses_categories_gin
    ON businesses USING GIN (categories);

-- businesses: attribute lookups via the @> containment operator.
CREATE INDEX IF NOT EXISTS idx_businesses_attributes_gin
    ON businesses USING GIN (attributes);

-- reviews: most queries filter by business.
CREATE INDEX IF NOT EXISTS idx_reviews_business_id
    ON reviews (business_id);

-- reviews: time-window queries.
CREATE INDEX IF NOT EXISTS idx_reviews_date
    ON reviews (date);

-- reviews: composite (business_id, date) — the primary index for review
-- velocity / rating-delta queries over a time window for a given business.
CREATE INDEX IF NOT EXISTS idx_reviews_business_id_date
    ON reviews (business_id, date);

-- tips: time-window queries scoped to a business.
CREATE INDEX IF NOT EXISTS idx_tips_business_id_date
    ON tips (business_id, date);

-- tips: joining with businesses.
CREATE INDEX IF NOT EXISTS idx_tips_business_id
    ON tips (business_id);
