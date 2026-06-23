"""LLM reranker (RankGPT-style).

After hybrid retrieval + RRF gives us a pool of candidate chunks, a cross-
attention reranker decides which few actually answer the question. We use the
LLM for this (reuses the OpenAI key, no extra dependency): it reads the question
and the numbered candidates and returns the most relevant passage numbers,
best first.

This is the step that rescues the hard cases — a candidate that ranked mid-pack
because it's a dense table can still be pulled to the top once the model reads
it against the question.
"""
from pydantic import BaseModel, Field

from app.config import settings
from app.llm import get_client

_SYSTEM = (
    "You are a precise reranker for a retrieval system over company annual "
    "reports. Given a question and numbered passages, return the passage numbers "
    "ordered from MOST to LEAST relevant for answering the question. Judge by "
    "whether a passage actually contains the answer (the specific figure, the "
    "exact topic asked about) — not just related themes. Omit clearly off-topic "
    "passages."
)


class _Ranking(BaseModel):
    ranked_indices: list[int] = Field(
        description="Passage numbers from most to least relevant. Omit irrelevant ones."
    )


def rerank(query: str, candidates: list[tuple], top_k: int) -> list[tuple]:
    """candidates: list of (id, document, metadata). Returns the top_k reranked."""
    if len(candidates) <= 1:
        return candidates[:top_k]

    lines = []
    for i, (_id, doc, meta) in enumerate(candidates):
        text = " ".join(doc.split())[:400]
        lines.append(f"[{i}] (page {meta['page']}) {text}")
    user = (
        f"Question: {query}\n\nPassages:\n\n" + "\n\n".join(lines) +
        f"\n\nReturn the passage numbers most relevant to the question, best first "
        f"(up to {top_k})."
    )

    completion = get_client().beta.chat.completions.parse(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format=_Ranking,
        temperature=0,
    )
    order = completion.choices[0].message.parsed.ranked_indices

    picked, seen = [], set()
    for idx in order:
        if 0 <= idx < len(candidates) and idx not in seen:
            seen.add(idx)
            picked.append(candidates[idx])
        if len(picked) >= top_k:
            break
    # If the model returned fewer than top_k, top up from the fused order.
    for i, cand in enumerate(candidates):
        if len(picked) >= top_k:
            break
        if i not in seen:
            picked.append(cand)
            seen.add(i)
    return picked
