"""Hop 1 — PDF parsing.

Turns a PDF file into a list of page-tagged text blocks. We use two libraries
on purpose:

* PyMuPDF (`fitz`) for the bulk prose — fast, and it gives us the page number
  for every block, which we need for citations later.
* pdfplumber for tables — annual reports hide the important numbers (spend,
  FTE, financials) in tables, and a text-only parser flattens those into soup.
  pdfplumber reconstructs rows/columns so a label stays attached to its value.

The page number is kept throughout so the final answer can cite a source page.
"""
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber


@dataclass
class ParsedPage:
    page: int          # 1-based page number, used for citations
    text: str          # prose + flattened tables for this page


def _flatten_table(table: list[list]) -> str:
    """Turn pdfplumber's list-of-rows into readable 'cell | cell | cell' lines
    so the row/column relationship survives into the embedding."""
    lines = []
    for row in table:
        cells = [(cell or "").strip() for cell in row]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def parse_pdf(path: str | Path) -> list[ParsedPage]:
    path = Path(path)
    pages: dict[int, str] = {}

    # --- prose via PyMuPDF ---
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            pages[i] = page.get_text().strip()

    # --- tables via pdfplumber, appended to the same page ---
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            if not tables:
                continue
            table_text = "\n\n".join(_flatten_table(t) for t in tables)
            existing = pages.get(i, "")
            pages[i] = (existing + "\n\n[TABLES]\n" + table_text).strip()

    # Drop empty pages (covers, blank dividers) and return in page order.
    return [ParsedPage(page=i, text=pages[i]) for i in sorted(pages) if pages[i]]
