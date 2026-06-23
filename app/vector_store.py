"""Hop 4 — the vector store (ChromaDB, persistent).

Stores three things per chunk: the embedding (for similarity search), the
original text (so we can show a verbatim quote), and metadata (company, source,
page — for citations and per-company filtering).

`PersistentClient` writes to disk under data/chroma/, so the index survives a
restart with no extra code — that's the "data persists" requirement.

We pass our own embeddings in (computed in embeddings.py) rather than letting
Chroma embed for us, so the embedding model is explicit and identical at ingest
and query time.
"""
import logging

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.chunker import Chunk
from app.config import settings

# ChromaDB's telemetry has a known posthog signature bug that spams stderr even
# when disabled via Settings. Silence its logger — telemetry is off regardless.
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

_collection = None


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        _collection = client.get_or_create_collection(
            name="report_chunks",
            metadata={"hnsw:space": "cosine"},  # cosine fits OpenAI embeddings
        )
    return _collection


def add_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    get_collection().add(
        ids=[c.id for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[
            {"company": c.company, "source": c.source, "page": c.page}
            for c in chunks
        ],
    )


def delete_source(source: str) -> None:
    """Remove all chunks for a file so re-ingesting it doesn't create duplicates
    (makes ingestion idempotent)."""
    get_collection().delete(where={"source": source})


def query(embedding: list[float], k: int, company: str | None = None) -> dict:
    where = {"company": company} if company else None
    return get_collection().query(
        query_embeddings=[embedding],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )


def keyword_search(
    pattern: str, company: str | None = None, limit: int = 5
) -> list[tuple[str, dict]]:
    """Exact-ish keyword retrieval: scan stored chunks for a regex match on the
    whitespace-normalized text. Complements vector search — dense numeric tables
    (e.g. an FTE total) embed poorly and can be missed by similarity alone, but
    still contain the literal phrase we're after. Returns (document, metadata)."""
    import re

    where = {"company": company} if company else None
    data = get_collection().get(where=where, include=["documents", "metadatas"])
    rx = re.compile(pattern, re.I)
    out: list[tuple[str, dict]] = []
    for doc, meta in zip(data["documents"], data["metadatas"]):
        if rx.search(" ".join(doc.split())):
            out.append((doc, meta))
            if len(out) >= limit:
                break
    return out


def count() -> int:
    return get_collection().count()


def list_sources() -> list[dict]:
    """Distinct (company, source) pairs currently in the store."""
    data = get_collection().get(include=["metadatas"])
    seen = {}
    for meta in data["metadatas"]:
        seen[meta["source"]] = meta["company"]
    return [{"source": s, "company": c} for s, c in seen.items()]
