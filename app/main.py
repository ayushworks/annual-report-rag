"""FastAPI application — HTTP layer over the RAG building blocks.

These routes are thin wrappers: all the real work lives in the tested modules
(ingest, chat, facts_store, vector_store). The static frontend is mounted last
so it never shadows the /api routes.
"""
import shutil

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import facts_store, vector_store
from app.chat import answer
from app.config import ROOT_DIR, settings
from app.ingest import ingest_pdf

app = FastAPI(title="Annual Report RAG", version="0.1.0")

STATIC_DIR = ROOT_DIR / "app" / "static"


# --- Request/response models --------------------------------------------------
class ChatRequest(BaseModel):
    question: str
    company: str | None = None   # optional filter to one company's report


# --- API routes ---------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "openai_key_set": bool(settings.openai_api_key),
        "chunks_indexed": vector_store.count(),
    }


@app.get("/api/reports")
def reports() -> list[dict]:
    """Reports currently in the store — drives the company filter in the UI."""
    return vector_store.list_sources()


@app.get("/api/facts")
def facts() -> list[dict]:
    """Pre-extracted FTE + sustainability goals for the dashboard."""
    return facts_store.get_all_facts()


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    return answer(req.question, company=req.company)


@app.post("/api/upload")
def upload(file: UploadFile, company: str = Form(...)) -> dict:
    """Accept a PDF, save it to reports/, and ingest it (parse→chunk→embed→store
    →extract). Ingestion runs in the foreground and the result is returned."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    dest = settings.reports_dir / file.filename
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        result = ingest_pdf(dest, company=company.strip())
    except Exception as exc:  # surface ingestion errors as a clean 400
        raise HTTPException(status_code=400, detail=f"Ingestion failed: {exc}")
    return result


# --- Frontend (mounted last) --------------------------------------------------
@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
