"""Hop 3 — embeddings.

Turns chunk text into vectors via OpenAI's embedding model. The same function
is used at ingest time (embed every chunk) and at query time (embed the user's
question), which guarantees both live in the same vector space.

Calls are batched because the API accepts many inputs per request — far fewer
round-trips than one call per chunk.
"""
from app.config import settings
from app.llm import get_client


def embed_texts(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    client = get_client()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=settings.embedding_model, input=batch)
        vectors.extend(item.embedding for item in resp.data)
    return vectors


def embed_query(text: str) -> list[float]:
    """Convenience wrapper for the single-string query case."""
    return embed_texts([text])[0]
