"""Central configuration, loaded from environment / .env file.

Keeping all settings in one typed object makes the app easy to reason about
and means there are no magic strings scattered across the codebase.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = the directory that contains this `app/` package.
ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- OpenAI ---
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # --- Storage (everything on disk so it survives a restart) ---
    data_dir: Path = ROOT_DIR / "data"          # chroma + sqlite live here
    reports_dir: Path = ROOT_DIR / "reports"    # source PDFs

    # --- Retrieval ---
    chunk_size: int = 1000        # characters per chunk
    chunk_overlap: int = 150      # characters shared between adjacent chunks
    top_k: int = 8                # chunks retrieved per question (wider net so
                                  # precise figures in tables aren't crowded out
                                  # by prose mentions)
    rerank_candidates: int = 30   # size of the fused pool sent to the reranker

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "facts.db"


settings = Settings()

# Make sure the storage folders exist on startup.
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.reports_dir.mkdir(parents=True, exist_ok=True)
