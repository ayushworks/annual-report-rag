"""Step 3 — the pre-extraction pass.

Runs automatically at the end of ingestion. Instead of dumping whole pages at
the LLM, it reuses our retriever to find the passages most relevant to (a) FTE /
headcount and (b) sustainability goals, then asks the LLM to extract structured
facts from *only those excerpts*. Cheaper and more accurate than scanning the
whole document, and it makes the retriever earn its keep twice.

Output uses OpenAI structured outputs (a Pydantic schema), so we get typed,
validated data back — including a verbatim quote and page numbers for evidence.
"""
from pydantic import BaseModel, Field

from app import vector_store
from app.config import settings
from app.llm import get_client
from app.retriever import retrieve

# Strong phrases that mark an authoritative employee-count table. Used for the
# keyword half of the hybrid retrieval below.
_FTE_KEYWORD = r"full[\s-]*time\s+equivalent|total\s+headcount"


# --- The schema the LLM must return -------------------------------------------
class FteFact(BaseModel):
    value: str = Field(description="The FTE / employee count for the most recent year, "
                                   "e.g. '43,200 (worldwide, 2025)'. 'not found' if absent.")
    quote: str = Field(description="Verbatim sentence/row from the excerpts supporting it. "
                                   "Empty string if not found.")
    pages: list[int] = Field(description="Page number(s) the evidence came from.")


class SustainabilityGoal(BaseModel):
    goal: str = Field(description="One concrete sustainability goal or target.")
    quote: str = Field(description="Verbatim supporting text from the excerpts.")
    pages: list[int] = Field(description="Page number(s) the evidence came from.")


class ReportFacts(BaseModel):
    fte: FteFact
    sustainability_goals: list[SustainabilityGoal]


# --- Retrieval queries used to gather context ---------------------------------
_FTE_QUERY = (
    "total group consolidated number of employees full-time equivalents FTE "
    "headcount workforce average total employees"
)
_SUSTAINABILITY_QUERY = (
    "sustainability goals targets net zero carbon emissions reduction climate ambition"
)

_SYSTEM = (
    "You extract structured facts from excerpts of a company's annual report. "
    "Use ONLY the provided excerpts — never outside knowledge. Quote evidence "
    "verbatim. If a value is not in the excerpts, set its value to 'not found', "
    "quote to '', and pages to []. Always report the page number(s) you used."
)


def _gather_context(company: str) -> str:
    """Retrieve and de-duplicate the chunks relevant to both fields.

    FTE is a single number, so a few chunks suffice. Sustainability goals are a
    many-item answer, so we widen recall there to avoid missing goals."""
    by_page: dict[tuple[int, str], tuple[int, str]] = {}

    # Semantic half: vector similarity for both fields.
    for query, k in ((_FTE_QUERY, 6), (_SUSTAINABILITY_QUERY, 10)):
        for chunk in retrieve(query, k=k, company=company):
            key = (chunk.page, chunk.text[:50])
            by_page[key] = (chunk.page, chunk.text)

    # Keyword half: pull the literal FTE/headcount tables that vector search
    # misses because they are mostly numbers.
    for doc, meta in vector_store.keyword_search(_FTE_KEYWORD, company, limit=5):
        key = (meta["page"], doc[:50])
        by_page[key] = (meta["page"], doc)

    parts = [f"[page {page}] {text}" for page, text in sorted(by_page.values())]
    return "\n\n".join(parts)


def extract_facts(company: str, source: str) -> ReportFacts:
    context = _gather_context(company)
    user = (
        f"Company: {company}\n\n"
        f"Excerpts from the annual report:\n\n{context}\n\n"
        "Extract:\n"
        "1. Total employees for the GROUP as a whole, most recent year, on a "
        "FULL-TIME-EQUIVALENT (FTE) basis. Rules, in order:\n"
        "   a. It must be the consolidated group-wide total. If employees are "
        "broken down by country, segment, or legal entity, use the 'Total' row "
        "or group figure, NEVER a single country/entity row.\n"
        "   b. If the report reports both a 'headcount' (number of people) and "
        "an FTE / full-time-equivalent figure, you MUST report the FTE one, "
        "even when the headcount number is larger or more prominent.\n"
        "2. The company's sustainability goals / targets. List EVERY distinct "
        "goal or target you find as a SEPARATE item — do not merge them. "
        "Companies typically have several (e.g. separate targets for scope 1/2 "
        "emissions, scope 3, net-zero dates, energy, circularity). Keep each "
        "target with its own specific figure and deadline."
    )
    client = get_client()
    completion = client.beta.chat.completions.parse(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format=ReportFacts,
        temperature=0,  # deterministic, reproducible extraction
    )
    return completion.choices[0].message.parsed
