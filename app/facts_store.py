"""Persisted store for the pre-extracted datapoints (FTE + sustainability goals).

Kept separate from the vector store on purpose: the vector store answers
"which passages are relevant?", while this answers "what did we pull out of each
report?". It's a plain SQLite file on disk, so it survives a restart and powers
the dashboard.

One row per report. FTE is flat columns (single value); the variable-length
sustainability goals are stored as a JSON blob.
"""
import json
import sqlite3
from typing import TYPE_CHECKING

from app.config import settings

if TYPE_CHECKING:
    from app.extraction import ReportFacts


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.sqlite_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS report_facts (
                source                TEXT PRIMARY KEY,
                company               TEXT NOT NULL,
                fte_value             TEXT,
                fte_quote             TEXT,
                fte_pages             TEXT,   -- JSON list of page numbers
                sustainability_goals  TEXT,   -- JSON list of {goal, quote, pages}
                extracted_at          TEXT
            )
            """
        )


def save_facts(source: str, company: str, facts: "ReportFacts") -> None:
    """Insert or replace the facts row for a report (idempotent on re-ingest)."""
    with _conn() as c:
        c.execute(
            """
            INSERT INTO report_facts
                (source, company, fte_value, fte_quote, fte_pages,
                 sustainability_goals, extracted_at)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(source) DO UPDATE SET
                company              = excluded.company,
                fte_value            = excluded.fte_value,
                fte_quote            = excluded.fte_quote,
                fte_pages            = excluded.fte_pages,
                sustainability_goals = excluded.sustainability_goals,
                extracted_at         = excluded.extracted_at
            """,
            (
                source,
                company,
                facts.fte.value,
                facts.fte.quote,
                json.dumps(facts.fte.pages),
                json.dumps([g.model_dump() for g in facts.sustainability_goals]),
            ),
        )


def _row_to_dict(r: sqlite3.Row) -> dict:
    return {
        "source": r["source"],
        "company": r["company"],
        "fte": {
            "value": r["fte_value"],
            "quote": r["fte_quote"],
            "pages": json.loads(r["fte_pages"] or "[]"),
        },
        "sustainability_goals": json.loads(r["sustainability_goals"] or "[]"),
        "extracted_at": r["extracted_at"],
    }


def get_all_facts() -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM report_facts ORDER BY company").fetchall()
    return [_row_to_dict(r) for r in rows]


# Ensure the table exists as soon as the module is imported.
init_db()
