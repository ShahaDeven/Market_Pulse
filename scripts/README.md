# Yelp Open Dataset Loader

## Overview

This script loads the Yelp Open Dataset into the local Postgres instance for
use by the `yelp-events-mcp` server. **The data is NOT included in the repo** —
the files are gitignored and large (~10 GB extracted), so each user must
download them independently.

## Prerequisites

- **Docker Compose is running and Postgres is up** — `docker compose ps` shows
  `marketpulse-postgres` as healthy.
- **The `db/schema/001_yelp_data.sql` migration has run** — verify with:

  ```bash
  docker compose exec postgres psql -U marketpulse -d marketpulse -c "\dt"
  ```

- **Python 3.11+ with `uv` installed.**
- **~15 GB free disk space** (10 GB extracted data + Postgres footprint after
  load).

## Step 1: Download the Yelp Open Dataset

- **URL:** https://business.yelp.com/data/resources/open-dataset/
- **License:** Academic use only (Yelp Dataset Terms of Use, 2023-07-07)
- **File:** `yelp_dataset.tar` (~4.35 GB compressed)
- **Accept the license terms** on Yelp's site before downloading.

## Step 2: Extract

Use Windows native `tar` (**NOT WinRAR** — it produces a broken single-file
extraction):

```powershell
mkdir data\yelp
tar -xvf yelp_dataset.tar -C data\yelp
```

On macOS/Linux:

```bash
mkdir -p data/yelp
tar -xvf yelp_dataset.tar -C data/yelp
```

## Step 3: Verify the expected files exist

After extraction, `data/yelp/` should contain:

| File | Approximate Size | Used by loader |
|---|---|---|
| `yelp_academic_dataset_business.json` | ~120 MB | **Yes** (businesses table) |
| `yelp_academic_dataset_review.json` | ~5.3 GB | **Yes** (reviews table) |
| `yelp_academic_dataset_tip.json` | ~250 MB | **Yes** (tips table) |
| `yelp_academic_dataset_user.json` | ~3.4 GB | No (skip) |
| `yelp_academic_dataset_checkin.json` | ~400 MB | No (skip) |
| `Dataset_User_Agreement.pdf` | <1 MB | No |

> You can delete `user.json` and `checkin.json` after extraction to save disk
> space — we don't load them.

## Step 4: Run the loader

Load everything (businesses → tips → reviews) in one go:

```powershell
uv run python scripts/load_yelp_data.py --force
```

`--force` skips the per-table TRUNCATE confirmation prompts. Useful flags:

| Flag | Purpose |
|---|---|
| `--table businesses\|reviews\|tips\|all` | Load one table (default: `all`). |
| `--tables-only reviews,tips` | Load a comma-separated subset; overrides `--table`. |
| `--cities "Philadelphia,Tampa"` | Scope tips/reviews to one or more cities (see below). |
| `--force` | Skip TRUNCATE confirmation prompts. |
| `--file <path>` | Override the source file (single-table runs only). |

Each loader is idempotent: it TRUNCATEs its table first, so re-running is safe.

### Loading a city subset (HDD users)

The full dataset is ~7M reviews. Loading all of it is only recommended on
**SSD/NVMe** storage — on a spinning HDD it can run for hours and has been
observed to hit out-of-memory / stuck COPY transactions.

If you're on an HDD (or just want a faster, smaller working set), scope the
**tips and reviews** to specific cities with `--cities`:

```powershell
uv run python scripts/load_yelp_data.py --tables-only=tips,reviews --force --cities "Philadelphia"
```

How the filter behaves:

- The **businesses table is always loaded in full** (~150K rows is cheap, and
  tips/reviews need it to resolve the city filter). Run a full load or at least
  the businesses table **before** a filtered tips/reviews run.
- **tips** and **reviews** are inserted only for businesses located in the
  chosen cities, so there are **no orphaned records** — the subset stays
  referentially consistent with `businesses`.
- City matching is **case-insensitive** (`"Philadelphia"` == `"philadelphia"`),
  and whitespace around commas is stripped. Pass multiple cities comma-separated:
  `--cities "Philadelphia, Tampa, Tucson"`.
- Philadelphia alone is ~1.2M reviews — a much more manageable load than the
  full ~7M.

## License caveats

- **Academic/educational use only** per Yelp Dataset Terms (2023-07-07).
- **No commercial use** without a separate Yelp Data Licensing agreement
  (https://business.yelp.com/data/).
- **No redistribution** of the dataset itself.
- **All derivative works remain Yelp's property.**
- **License terminates 12 months after first access** — re-download if you
  return to the project after that.
- **Data must be deleted upon termination.**

## Troubleshooting

- **`tar -xvf` produces a single file with no extension instead of 5 JSON
  files:** you used WinRAR or another tool that doesn't handle tar archives
  correctly. Re-extract using Windows native `tar`.
