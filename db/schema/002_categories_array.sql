-- Migration 002: Normalize categories to TEXT[] for working containment queries
--
-- The Yelp Open Dataset stores categories as a comma-separated JSONB string
-- (e.g., '"Restaurants, Food, Bakeries"'), not as a JSON array as the docs
-- claim. Containment queries against the original column return 0 results
-- because string @> array is always false in Postgres JSONB.
--
-- This migration adds a categories_array TEXT[] column populated by splitting
-- the comma-separated string. The original categories column is preserved
-- as source of truth.
--
-- Idempotent and safe to re-run. See ADR-009.

ALTER TABLE businesses ADD COLUMN IF NOT EXISTS categories_array TEXT[];

UPDATE businesses
SET categories_array = string_to_array(
    regexp_replace(categories #>> '{}', ',\s*', ',', 'g'),
    ','
)
WHERE categories IS NOT NULL
  AND jsonb_typeof(categories) = 'string'
  AND categories_array IS NULL;

CREATE INDEX IF NOT EXISTS idx_businesses_categories_array
ON businesses USING GIN (categories_array);