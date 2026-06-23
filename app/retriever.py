"""Retrieval.

Two entry points:

* `retrieve` — plain vector similarity. Used by the pre-extraction pass.
* `hybrid_retrieve` — the full pipeline for chat: vector + BM25 run in parallel,
  fused with Reciprocal Rank Fusion, then reranked by the LLM. This is what
  rescues specific, table-bound figures with a vocabulary gap (the SHELL
  "climate change adaptation" case), which pure vector search misses.

Both return RetrievedChunk objects carrying the page number and verbatim text
needed for citations.
"""
from dataclasses import dataclass

from app import lexical, reranker, vector_store
from app.config import settings
from app.embeddings import embed_query


@dataclass
class RetrievedChunk:
    text: str
    company: str
    source: str
    page: int
    distance: float


def retrieve(query: str, k: int, company: str | None = None) -> list[RetrievedChunk]:
    res = vector_store.query(embed_query(query), k=k, company=company)
    # Chroma nests results one level deep (one list per query); we sent one query.
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    return [
        RetrievedChunk(
            text=doc,
            company=meta["company"],
            source=meta["source"],
            page=meta["page"],
            distance=dist,
        )
        for doc, meta, dist in zip(docs, metas, dists)
    ]


def _vector_candidates(query: str, n: int, company: str | None) -> list[tuple]:
    res = vector_store.query(embed_query(query), k=n, company=company)
    return list(zip(res["ids"][0], res["documents"][0], res["metadatas"][0]))


def _rrf(ranked_lists: list[list[tuple]], k: int = 60) -> list[tuple]:
    """Reciprocal Rank Fusion. Each list is ordered (id, doc, meta) tuples; an
    item's fused score is the sum of 1/(k + rank) across the lists it appears in.
    No weights to tune — items ranked highly by either retriever bubble up."""
    scores: dict[str, float] = {}
    items: dict[str, tuple] = {}
    for lst in ranked_lists:
        for rank, (cid, doc, meta) in enumerate(lst):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            items[cid] = (cid, doc, meta)
    ordered = sorted(scores, key=lambda c: scores[c], reverse=True)
    return [items[c] for c in ordered]


def hybrid_retrieve(query: str, k: int, company: str | None = None) -> list[RetrievedChunk]:
    n = settings.rerank_candidates
    vector_hits = _vector_candidates(query, n, company)
    lexical_hits = lexical.bm25_search(query, n, company)

    fused = _rrf([vector_hits, lexical_hits])[:n]
    top = reranker.rerank(query, fused, top_k=k)

    return [
        RetrievedChunk(
            text=doc,
            company=meta["company"],
            source=meta["source"],
            page=meta["page"],
            distance=0.0,  # rank-based pipeline; distance isn't meaningful here
        )
        for _id, doc, meta in top
    ]
