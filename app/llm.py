"""Shared OpenAI client.

One lazily-built client used by both embeddings and extraction/chat, so the API
key is read in exactly one place and importing these modules never requires a
key (only actually calling the API does).
"""
from openai import OpenAI

from app.config import settings

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client
