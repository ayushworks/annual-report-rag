"""Step 4 — grounded chat answers with citations.

The query-time reverse of ingestion:

    question → embed → retrieve top-k chunks → prompt the LLM with those chunks
             → answer grounded ONLY in them, citing page numbers

The grounding contract (in the system prompt) is what makes answers trustworthy
for the litmus-test question ("how much did SHELL spend on climate adaptation"):
the model must use only the retrieved excerpts, cite the page, and admit when
the answer isn't there rather than inventing one.

Returns the answer text plus the source chunks it was given, so the frontend can
show the verbatim evidence behind every answer.
"""
from dataclasses import asdict

from app.config import settings
from app.llm import get_client
from app.retriever import RetrievedChunk, hybrid_retrieve

_SYSTEM = (
    "You answer questions about company annual reports using ONLY the numbered "
    "excerpts provided. Rules:\n"
    "- Base every fact strictly on the excerpts. Never use outside knowledge or "
    "invent figures.\n"
    "- Cite the page after each fact, e.g. (p.227), using the page numbers given.\n"
    "- Quote figures verbatim as they appear (keep currencies, units, years).\n"
    "- If the excerpts contain a direct answer, give it plainly.\n"
    "- If there is no exact match but the excerpts contain closely related figures "
    "(the report uses different wording, or reports an adjacent/broader number), "
    "present those as the closest available evidence: say plainly that it is not "
    "an exact match, state what each figure actually measures, and cite each one. "
    "A cited, qualified answer is better than refusing.\n"
    "- Only when nothing in the excerpts is relevant at all, say: \"I couldn't find "
    "that in the reports.\"\n"
    "- Be concise and direct; do not pad."
)


def _format_sources(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        text = " ".join(c.text.split())  # collapse whitespace for a tidy prompt
        blocks.append(f"[Source {i} | {c.company}, page {c.page}]\n{text}")
    return "\n\n".join(blocks)


def answer(question: str, company: str | None = None) -> dict:
    chunks = hybrid_retrieve(question, k=settings.top_k, company=company)

    if not chunks:
        return {
            "answer": "I couldn't find that in the reports. No documents are "
                      "ingested yet, or none matched your question.",
            "sources": [],
        }

    user = (
        f"Question: {question}\n\n"
        f"Excerpts:\n\n{_format_sources(chunks)}\n\n"
        "Answer the question using only these excerpts, with page citations."
    )

    client = get_client()
    completion = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    answer_text = completion.choices[0].message.content

    return {
        "answer": answer_text,
        "sources": [asdict(c) for c in chunks],
    }
