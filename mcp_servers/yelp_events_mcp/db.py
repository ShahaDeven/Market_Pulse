"""Postgres connection helper for the yelp-events MCP server.

Reads ``DATABASE_URL`` from the environment (loaded from ``.env`` via
python-dotenv) and returns a fresh psycopg 3 connection per call. The caller
owns the connection lifecycle and MUST close it — use a ``try/finally``.

A connection-per-call model is deliberate: this MCP server is launched as an
ephemeral subprocess that handles a handful of tool calls and exits, so a
connection pool would add complexity with no payoff. See ARCHITECTURE.md ADR-002.
"""

from __future__ import annotations

import os

import psycopg
from dotenv import load_dotenv

# Load .env once at import time. Safe to call repeatedly; it does not override
# variables already present in the real environment.
load_dotenv()


def get_connection() -> psycopg.Connection:
    """Open and return a new psycopg 3 connection to the MarketPulse Postgres.

    The caller is responsible for closing the returned connection.

    Raises:
        RuntimeError: if ``DATABASE_URL`` is not set in the environment.
    """
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env and set it. "
            "Remember the host port is 5433, not 5432."
        )
    return psycopg.connect(dsn)
