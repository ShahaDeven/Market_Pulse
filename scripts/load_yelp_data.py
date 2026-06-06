#!/usr/bin/env python3
"""Load the Yelp Open Dataset into Postgres via COPY FROM.

Supports the three tables we ingest from the dataset:
  - businesses (from yelp_academic_dataset_business.json)
  - tips       (from yelp_academic_dataset_tip.json)
  - reviews    (from yelp_academic_dataset_review.json)

With --table=all (the default) they load smaller-to-larger
(businesses -> tips -> reviews) so each can be sanity-checked before the long
reviews step.

Tips and reviews can be scoped to one or more cities with --cities; in that
mode only rows belonging to businesses in those cities are inserted (the
businesses table is always loaded in full), keeping the subset referentially
consistent. This is the recommended path on HDD storage, where the full
~7M-review load risks OOM / stuck COPY transactions.

See scripts/README.md for how to acquire and extract the dataset, and
db/schema/001_yelp_data.sql for the target table definitions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Iterator, Optional

import psycopg
import structlog
from dotenv import load_dotenv
from tqdm import tqdm

# ----------------------------------------------------------------------------
# structlog: JSON output, per project convention (never print()).
# ----------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger()

# If more than this fraction of records fail to parse, we assume schema drift
# (the source file changed shape) and flag it with a distinct exit code.
SKIP_RATIO_ABORT = 0.01

# Default source file per table (relative to the repo root / CWD).
DEFAULT_FILES = {
    "businesses": "data/yelp/yelp_academic_dataset_business.json",
    "reviews": "data/yelp/yelp_academic_dataset_review.json",
    "tips": "data/yelp/yelp_academic_dataset_tip.json",
}

VALID_TABLES = ("businesses", "reviews", "tips")

# Column order MUST match the COPY target list and the table definitions in
# db/schema/001_yelp_data.sql.
BUSINESS_COLUMNS = (
    "business_id",
    "name",
    "address",
    "city",
    "state",
    "postal_code",
    "latitude",
    "longitude",
    "stars",
    "review_count",
    "is_open",
    "attributes",
    "categories",
    "hours",
)

REVIEW_COLUMNS = (
    "review_id",
    "user_id",
    "business_id",
    "stars",
    "date",
    "text",
    "useful",
    "funny",
    "cool",
)

# NOTE: the tips table has an `id BIGSERIAL PRIMARY KEY` that Postgres
# auto-generates — it is deliberately NOT in this COPY column list.
TIP_COLUMNS = (
    "text",
    "date",
    "compliment_count",
    "business_id",
    "user_id",
)


@dataclass
class LoadResult:
    """Outcome of loading one table, used for the final run summary."""

    table: str
    loaded: int
    skipped: int
    drift: bool  # True if the skipped ratio exceeded SKIP_RATIO_ABORT


def escape_for_copy(value: object) -> str:
    """Escape a value for Postgres COPY FROM with tab delimiter."""
    if value is None:
        return r"\N"
    s = str(value)
    s = s.replace("\\", "\\\\")  # backslash first
    s = s.replace("\t", "\\t")
    s = s.replace("\n", "\\n")
    s = s.replace("\r", "\\r")
    return s


def _jsonb(value: Any) -> Optional[str]:
    """Serialize a JSONB-bound value to a JSON string, or None if missing.

    `attributes` (nested object), `categories` (array of strings), and `hours`
    (day -> range object) are all passed through json.dumps as-is. Any of them
    may be missing/None for a given business, which becomes a SQL NULL.
    """
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _validate_date(value: Any) -> Optional[str]:
    """Return a validated date string, None if absent, or raise if unparseable.

    The source "date" fields are ISO strings ("YYYY-MM-DD" or
    "YYYY-MM-DD HH:MM:SS"). We validate with datetime.fromisoformat and then
    hand the original string to COPY, letting Postgres cast it to timestamptz.
    Raises ValueError on a present-but-unparseable value so the caller can skip
    the row.
    """
    if value is None or value == "":
        return None
    s = str(value)
    try:
        datetime.fromisoformat(s)
    except ValueError as exc:
        raise ValueError(f"unparseable date: {s!r}") from exc
    return s


def _to_int(value: Any) -> Optional[int]:
    """Cast a possibly float-formatted numeric to int; None passes through.

    Yelp's data occasionally stores integer columns as floats ("3.0") despite
    the documentation. int(float(...)) handles both "3" and "3.0". Returns None
    for a missing value (-> SQL NULL); raises ValueError on a non-numeric value
    so the caller can skip the row (ValueError is what the loaders already
    catch).
    """
    if value is None:
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"non-numeric int value: {value!r}") from exc


def build_business_row(record: dict[str, Any]) -> str:
    """Build a single tab-separated COPY line for the businesses table.

    Raises ValueError if the record lacks a business_id (the primary key), so
    the caller can count it as a skipped record rather than fail the load.
    """
    business_id = record.get("business_id")
    if not business_id:
        raise ValueError("record missing business_id")

    fields: list[object] = [
        business_id,
        record.get("name"),
        record.get("address"),
        record.get("city"),
        record.get("state"),
        # NOTE: source JSON key is "postal code" WITH A SPACE, not an
        # underscore. record["postal_code"] would KeyError on every row.
        record.get("postal code"),
        record.get("latitude"),
        record.get("longitude"),
        record.get("stars"),  # businesses.stars is REAL — "3.0" is fine as-is
        _to_int(record.get("review_count")),
        _to_int(record.get("is_open")),
        _jsonb(record.get("attributes")),
        _jsonb(record.get("categories")),
        _jsonb(record.get("hours")),
    ]
    return "\t".join(escape_for_copy(f) for f in fields) + "\n"


def build_review_tuple(record: dict[str, Any]) -> tuple[Any, ...]:
    """Build a typed value tuple for a reviews COPY row (order = REVIEW_COLUMNS).

    For use with psycopg 3's copy.write_row, which handles type adaptation and
    escaping — no manual tab-escaping needed. Raises ValueError (caller skips
    the row) when review_id, date, or stars is missing/unparseable. Missing/null
    text becomes an empty string, not NULL.
    """
    review_id = record.get("review_id")
    if not review_id:
        raise ValueError("record missing review_id")

    date = _validate_date(record.get("date"))
    if date is None:
        raise ValueError("record missing date")

    # Yelp docs say review stars are integers, but some rows store them as
    # floats ("3.0"), which Postgres SMALLINT rejects. Cast explicitly.
    stars_raw = record.get("stars")
    if stars_raw is None:
        raise ValueError("record missing stars")
    try:
        stars = int(float(stars_raw))  # handles both 3 and 3.0
    except (ValueError, TypeError) as exc:
        raise ValueError(f"unparseable stars: {stars_raw!r}") from exc

    text = record.get("text")
    if text is None:
        text = ""  # store empty string, not NULL

    return (
        review_id,
        record.get("user_id"),
        record.get("business_id"),
        stars,
        date,
        text,
        _to_int(record.get("useful")),
        _to_int(record.get("funny")),
        _to_int(record.get("cool")),
    )


def build_tip_tuple(record: dict[str, Any]) -> tuple[Any, ...]:
    """Build a typed value tuple for a tips COPY row (order = TIP_COLUMNS).

    The id column is BIGSERIAL and omitted (Postgres generates it). A missing
    date becomes SQL NULL; a present-but-unparseable date raises (caller skips).
    A missing/null text becomes an empty string.
    """
    date = _validate_date(record.get("date"))

    text = record.get("text")
    if text is None:
        text = ""  # store empty string, not NULL

    return (
        text,
        date,
        _to_int(record.get("compliment_count")),
        record.get("business_id"),
        record.get("user_id"),
    )


def count_lines(path: str) -> int:
    """Count newlines in a file (cheap pre-pass to size the progress bar)."""
    total = 0
    with open(path, "rb") as fh:
        for _ in fh:
            total += 1
    return total


def iter_json_lines(path: str) -> Iterator[str]:
    """Yield raw lines from a JSON-lines file (one object per line)."""
    # The Yelp files are JSON-LINES: one object per line, NOT a JSON array.
    # We read line by line and json.loads each — json.load() on the whole
    # ~5 GB review file would OOM.
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            yield line


def confirm_truncate(table: str, force: bool) -> None:
    """Ask for confirmation before TRUNCATE unless --force was passed."""
    if force:
        return
    answer = input(
        f"This will TRUNCATE the '{table}' table (all rows deleted). "
        "Continue? [y/N] "
    ).strip().lower()
    if answer not in ("y", "yes"):
        log.info("aborted_by_user", table=table)
        sys.exit(0)


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as H:MM:SS."""
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def load_businesses(
    conn: psycopg.Connection,
    file_path: str,
    force: bool,
) -> LoadResult:
    """Truncate and bulk-load the businesses table."""
    total_lines = count_lines(file_path)
    log.info("start", table="businesses", file=file_path, total_lines=total_lines)

    confirm_truncate("businesses", force)

    loaded = 0
    skipped = 0
    with conn.cursor() as cur:
        # TRUNCATE is far faster than DELETE for emptying a large table and
        # resets it cleanly for an idempotent re-load.
        log.info("truncate", table="businesses")
        cur.execute("TRUNCATE TABLE businesses")

        copy_sql = (
            f"COPY businesses ({', '.join(BUSINESS_COLUMNS)}) FROM STDIN"
        )
        log.info("copy_start", table="businesses")

        with cur.copy(copy_sql) as copy:
            for line in tqdm(
                iter_json_lines(file_path),
                total=total_lines,
                unit="rows",
                desc="businesses",
            ):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    row = build_business_row(record)
                except (json.JSONDecodeError, ValueError) as exc:
                    skipped += 1
                    log.warning("record_skipped", table="businesses", error=str(exc))
                    continue
                copy.write(row)
                loaded += 1

        log.info("copy_complete", table="businesses", loaded=loaded, skipped=skipped)

        processed = loaded + skipped
        drift = bool(processed) and (skipped / processed) > SKIP_RATIO_ABORT
        if drift:
            log.error(
                "skip_ratio_exceeded",
                table="businesses",
                skipped=skipped,
                processed=processed,
                ratio=round(skipped / processed, 4),
                threshold=SKIP_RATIO_ABORT,
                hint="likely schema drift — the business.json field names may "
                "have changed; inspect a sample line against build_business_row",
            )

        cur.execute("SELECT COUNT(*) FROM businesses")
        row_count = cur.fetchone()[0]
        log.info("row_count", table="businesses", count=row_count)

    conn.commit()
    return LoadResult("businesses", loaded, skipped, drift)


def fetch_business_ids_for_cities(
    conn: psycopg.Connection, cities: list[str]
) -> frozenset[str]:
    """Return business_ids whose city matches (case-insensitively) any input.

    Yelp city capitalization is inconsistent ("Philadelphia" vs "philadelphia"),
    so we LOWER() both the column and the supplied names. The result is a
    frozenset for O(1) membership tests during the stream.
    """
    lowered = [c.lower() for c in cities]
    with conn.cursor() as cur:
        cur.execute(
            "SELECT business_id FROM businesses WHERE LOWER(city) = ANY(%s)",
            (lowered,),
        )
        return frozenset(row[0] for row in cur)


def _stream_filtered_copy(
    conn: psycopg.Connection,
    *,
    table: str,
    columns: tuple[str, ...],
    file_path: str,
    tuple_builder: Callable[[dict[str, Any]], tuple[Any, ...]],
    force: bool,
    cities: Optional[list[str]],
) -> LoadResult:
    """Stream a JSON-lines file into `table` via psycopg 3 COPY, row by row.

    Each row goes straight to the COPY stream with copy.write_row — there is NO
    Python-side list/string accumulation — so peak memory stays bounded
    regardless of row count. This is the fix for the prior OOM / stuck COPY on
    the full national review set.

    When `cities` is provided, the matching business_id set is built ONCE up
    front and only rows whose business_id is in it are inserted, keeping
    tips/reviews referentially consistent with the fully loaded businesses table.
    """
    matched_ids: Optional[frozenset[str]] = None
    if cities is not None:
        matched_ids = fetch_business_ids_for_cities(conn, cities)
        log.info(
            "city_filter",
            table=table,
            cities=cities,
            businesses_matched=len(matched_ids),
        )

    log.info("start", table=table, file=file_path)
    confirm_truncate(table, force)

    loaded = 0
    lines_read = 0
    skipped_city = 0
    skipped_errors = 0
    start = time.monotonic()

    with conn.cursor() as cur:
        log.info("truncate", table=table)
        # RESTART IDENTITY resets the tips BIGSERIAL; harmless for reviews.
        cur.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY")

        copy_sql = f"COPY {table} ({', '.join(columns)}) FROM STDIN"
        log.info("copy_start", table=table)

        with cur.copy(copy_sql) as copy:
            for line in tqdm(iter_json_lines(file_path), unit="rows", desc=table):
                line = line.strip()
                if not line:
                    continue
                lines_read += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    skipped_errors += 1
                    log.warning("row_skipped", table=table, error=str(exc))
                    continue

                if (
                    matched_ids is not None
                    and record.get("business_id") not in matched_ids
                ):
                    skipped_city += 1
                    continue

                try:
                    copy.write_row(tuple_builder(record))
                    loaded += 1
                except Exception as exc:  # noqa: BLE001 — per-row skip
                    skipped_errors += 1
                    log.warning("row_skipped", table=table, error=str(exc))

    elapsed = time.monotonic() - start

    # Drift check applies only to rows that passed the city filter — i.e. genuine
    # parse/build errors, not expected city mismatches.
    considered = loaded + skipped_errors
    drift = bool(considered) and (skipped_errors / considered) > SKIP_RATIO_ABORT
    if drift:
        log.error(
            "skip_ratio_exceeded",
            table=table,
            skipped_errors=skipped_errors,
            considered=considered,
            ratio=round(skipped_errors / considered, 4),
            threshold=SKIP_RATIO_ABORT,
            hint="likely schema drift — inspect a sample line against the row "
            "builder for this table",
        )

    log.info(
        "load_complete",
        table=table,
        businesses_matched=(len(matched_ids) if matched_ids is not None else None),
        lines_read=lines_read,
        loaded=loaded,
        skipped_city_filter=skipped_city,
        skipped_errors=skipped_errors,
        elapsed_seconds=round(elapsed, 1),
    )

    conn.commit()
    return LoadResult(table, loaded, skipped_city + skipped_errors, drift)


def load_reviews(
    conn: psycopg.Connection,
    file_path: str,
    force: bool,
    cities: Optional[list[str]],
) -> LoadResult:
    """Stream-load the reviews table, optionally scoped to --cities."""
    return _stream_filtered_copy(
        conn,
        table="reviews",
        columns=REVIEW_COLUMNS,
        file_path=file_path,
        tuple_builder=build_review_tuple,
        force=force,
        cities=cities,
    )


def load_tips(
    conn: psycopg.Connection,
    file_path: str,
    force: bool,
    cities: Optional[list[str]],
) -> LoadResult:
    """Stream-load the tips table, optionally scoped to --cities."""
    return _stream_filtered_copy(
        conn,
        table="tips",
        columns=TIP_COLUMNS,
        file_path=file_path,
        tuple_builder=build_tip_tuple,
        force=force,
        cities=cities,
    )


def resolve_tables(args: argparse.Namespace) -> list[str]:
    """Resolve the ordered list of tables to load from the CLI args.

    --tables-only (comma-separated) overrides --table. --table=all expands to
    businesses, tips, reviews (smaller-to-larger).
    """
    if args.tables_only:
        tables = [t.strip() for t in args.tables_only.split(",") if t.strip()]
        invalid = [t for t in tables if t not in VALID_TABLES]
        if invalid:
            raise ValueError(
                f"unknown table(s) in --tables-only: {', '.join(invalid)}; "
                f"valid: {', '.join(VALID_TABLES)}"
            )
        return tables
    if args.table == "all":
        return ["businesses", "tips", "reviews"]
    return [args.table]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load the Yelp Open Dataset into Postgres."
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Override the source JSON-lines file. Only honored when a single "
        "table is selected; ignored for multi-table runs (per-table defaults "
        "are used).",
    )
    parser.add_argument(
        "--table",
        default="all",
        choices=["businesses", "reviews", "tips", "all"],
        help="Which table to load. 'all' loads businesses, tips, then reviews.",
    )
    parser.add_argument(
        "--tables-only",
        default=None,
        help="Comma-separated subset of tables to load (e.g. 'reviews,tips'). "
        "Overrides --table. Useful for re-running just the failed tables.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the TRUNCATE confirmation prompt(s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Deprecated no-op: tips/reviews now stream row-by-row via "
        "copy.write_row, so memory is bounded without batching. Retained so "
        "existing invocations don't break.",
    )
    parser.add_argument(
        "--cities",
        default=None,
        help="Comma-separated city names to scope tips/reviews to "
        "(case-insensitive, e.g. 'Philadelphia,Tampa'). The businesses table is "
        "always loaded in full. Default: no filter (load all tips/reviews).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    load_dotenv()

    conninfo = os.environ.get("DATABASE_URL")
    if not conninfo:
        log.error("missing_env", var="DATABASE_URL", hint="set it in .env")
        return 1

    try:
        tables = resolve_tables(args)
    except ValueError as exc:
        log.error("invalid_tables", error=str(exc))
        return 1

    # --file only makes sense for a single-table run.
    if args.file and len(tables) > 1:
        log.warning(
            "file_arg_ignored",
            file=args.file,
            reason="multiple tables selected; using per-table default files",
        )

    # Parse --cities once; None means "no filter" (original behavior).
    cities: Optional[list[str]] = None
    if args.cities:
        cities = [c.strip() for c in args.cities.split(",") if c.strip()]
        if not any(t in ("tips", "reviews") for t in tables):
            log.warning(
                "cities_arg_ignored",
                cities=cities,
                reason="--cities only affects tips/reviews; none selected",
            )

    results: list[LoadResult] = []
    error_any = False
    overall_start = time.monotonic()

    try:
        with psycopg.connect(conninfo) as conn:
            for table in tables:
                file_path = (
                    args.file
                    if (args.file and len(tables) == 1)
                    else DEFAULT_FILES[table]
                )
                if not os.path.isfile(file_path):
                    log.error(
                        "file_not_found",
                        table=table,
                        file=file_path,
                        hint="see scripts/README.md",
                    )
                    error_any = True
                    continue

                if table == "businesses":
                    results.append(load_businesses(conn, file_path, args.force))
                elif table == "tips":
                    results.append(load_tips(conn, file_path, args.force, cities))
                elif table == "reviews":
                    results.append(
                        load_reviews(conn, file_path, args.force, cities)
                    )
    except Exception as exc:  # noqa: BLE001 — top-level guard, log and exit 1
        log.error("load_failed", error=str(exc), error_type=type(exc).__name__)
        return 1

    # Final summary: per-table loaded/skipped + total elapsed.
    log.info(
        "summary",
        tables={
            r.table: {"loaded": r.loaded, "skipped": r.skipped, "drift": r.drift}
            for r in results
        },
        total_elapsed=_fmt_duration(time.monotonic() - overall_start),
    )

    if error_any:
        return 1
    if any(r.drift for r in results):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
