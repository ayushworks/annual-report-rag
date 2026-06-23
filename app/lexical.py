"""Lexical (BM25) search — the keyword half of hybrid retrieval.

Vector search matches meaning but misses exact terms and dense numeric tables
(e.g. an FTE total, or a taxonomy row literally labelled "climate change
adaptation"). BM25 scores those by term overlap, so the two together cover each
other's blind spots.

The index is built over every stored chunk and cached. It rebuilds when the
collection changes (new chunk count) or when `invalidate()` is called after an
ingest.
"""
import re

from rank_bm25 import BM25Okapi

from app import vector_store

_TOKEN = re.compile(r"[a-z0-9]+")
_cache: dict | None = None


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def invalidate() -> None:
    """Drop the cached index (call after ingest changes the corpus)."""
    global _cache
    _cache = None


def _get_index() -> dict:
    global _cache
    count = vector_store.count()
    if _cache is None or _cache["count"] != count:
        data = vector_store.get_collection().get(include=["documents", "metadatas"])
        docs = data["documents"]
        bm25 = BM25Okapi([_tokenize(d) for d in docs]) if docs else None
        _cache = {
            "count": count,
            "bm25": bm25,
            "ids": data["ids"],
            "docs": docs,
            "metas": data["metadatas"],
        }
    return _cache


def bm25_search(query: str, k: int, company: str | None = None) -> list[tuple]:
    """Return up to k (id, document, metadata) tuples ranked by BM25 score."""
    idx = _get_index()
    if not idx["bm25"]:
        return []
    scores = idx["bm25"].get_scores(_tokenize(query))
    order = sorted(range(len(idx["docs"])), key=lambda i: scores[i], reverse=True)
    out = []
    for i in order:
        if scores[i] <= 0:
            break  # ranked list is sorted; nothing useful left
        if company and idx["metas"][i]["company"] != company:
            continue
        out.append((idx["ids"][i], idx["docs"][i], idx["metas"][i]))
        if len(out) >= k:
            break
    return out
