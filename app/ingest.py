"""The ingestion pipeline — orchestrates the four hops in order:

    PDF ──parse──► pages ──chunk──► chunks ──embed──► vectors ──store──► Chroma

This is the single entrypoint the API (and the bundled-report loader) will call.
"""
from pathlib import Path

from app import facts_store, lexical, vector_store
from app.chunker import chunk_pages
from app.config import settings
from app.embeddings import embed_texts
from app.extraction import extract_facts
from app.parser import parse_pdf


def ingest_pdf(path: str | Path, company: str) -> dict:
    path = Path(path)
    source = path.name

    # Idempotent: clear any existing chunks for this file first.
    vector_store.delete_source(source)

    # Hop 1 + 2: parse to page text, then split into chunks.
    pages = parse_pdf(path)
    chunks = chunk_pages(
        pages,
        company=company,
        source=source,
        size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    if not chunks:
        raise ValueError(f"No extractable text found in {source} (is it a scanned PDF?)")

    # Hop 3 + 4: embed the chunk text, then store vectors + text + metadata.
    embeddings = embed_texts([c.text for c in chunks])
    vector_store.add_chunks(chunks, embeddings)
    lexical.invalidate()  # the BM25 index must rebuild now the corpus changed

    # Step 3: pre-extract FTE + sustainability goals and persist them.
    # Runs now because retrieval (which extraction relies on) needs the chunks
    # to already be in the store.
    facts = extract_facts(company, source)
    facts_store.save_facts(source, company, facts)

    return {
        "source": source,
        "company": company,
        "pages": len(pages),
        "chunks": len(chunks),
        "facts": {
            "fte": facts.fte.value,
            "sustainability_goals": len(facts.sustainability_goals),
        },
    }
