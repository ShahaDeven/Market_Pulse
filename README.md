# MarketPulse

MarketPulse is a multi-source equity-research agent that drafts company-outlook
memos by triangulating three kinds of evidence — SEC filings (EDGAR), macro
indicators (FRED), and consumer-sentiment signals (the Yelp Open Dataset) — using
a LangGraph supervisor that routes each evidence channel to a scoped sub-agent.
Every memo carries a confidence score from a 5-axis rubric; high-confidence memos
auto-publish to an append-only, hash-chained audit log, while low-confidence ones
are paused via a durable checkpoint and routed to a Streamlit reviewer UI for
human approve/edit/reject before resuming. See
[ARCHITECTURE.md](./ARCHITECTURE.md) for the design rationale and
[CLAUDE.md](./CLAUDE.md) for the working context and conventions.
