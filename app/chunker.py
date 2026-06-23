"""Hop 2 — chunking.

Splits each page's text into overlapping character windows. Two reasons:

* Embeddings work best on focused passages, not whole pages.
* The LLM later receives only the retrieved chunks, so smaller = more precise.

The overlap means a sentence split across a boundary (e.g. "we invested $2,300"
/ "million in adaptation") still appears intact in at least one chunk, so the
number isn't lost. Each chunk keeps its company / source / page metadata so we
can cite it and filter by company at query time.
"""
from dataclasses import dataclass

from app.parser import ParsedPage


@dataclass
class Chunk:
    id: str            # stable, human-readable id e.g. "shell_2024.pdf_p42_c3"
    text: str
    company: str
    source: str        # original filename
    page: int


def _split(text: str, size: int, overlap: int) -> list[str]:
    """Character windows with overlap. Simple and easy to explain; good enough
    for digital annual reports."""
    if len(text) <= size:
        return [text]
    pieces, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        pieces.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap   # step back so windows overlap
    return pieces


def chunk_pages(
    pages: list[ParsedPage],
    company: str,
    source: str,
    size: int,
    overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        for j, piece in enumerate(_split(page.text, size, overlap)):
            chunks.append(
                Chunk(
                    id=f"{source}_p{page.page}_c{j}",
                    text=piece,
                    company=company,
                    source=source,
                    page=page.page,
                )
            )
    return chunks
