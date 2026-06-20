# MarketPulse Reviewer UI

Streamlit-based UI for browsing memos produced by the MarketPulse agent.

## Running

From the project root:

    uv run streamlit run reviewer_ui/app.py

Opens in your browser at http://localhost:8501

## Requirements

- Postgres running (`docker compose up postgres -d`)
- migrations applied including `004_memos.sql`
- `DATABASE_URL` set in `.env`

## Features (Chunk 1)

- Browse past memos with filters (data sources, outcome)
- Expand any memo to see findings, citations, caveats

## Coming in later chunks

- Submit new queries from the UI (Chunk 2)
- Handle HITL interrupts in the UI with approve/reject/retry buttons (Chunk 3)
