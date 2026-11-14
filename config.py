"""Central configuration for IntelliChat AI.

Every setting is read from environment variables (local `.env` file) or from
Streamlit Secrets (Streamlit Community Cloud). Model IDs live here so they can
be changed without touching application logic.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv(override=True)


def _read(name: str, default: str = "") -> str:
    """Read a setting from the environment first, then from Streamlit Secrets."""
    value = os.getenv(name)
    if value:
        return value.strip()
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:  # no secrets file, or not running inside Streamlit
        pass
    return default


def _read_int(name: str, default: int) -> int:
    raw = _read(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _read_float(name: str, default: float) -> float:
    raw = _read(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # --- model -----------------------------------------------------------
    groq_api_key: str
    gemini_api_key: str
    chat_model: str
    model_options: tuple[str, ...]
    embedding_model: str
    # NOTE: sql/schema.sql declares vector(768). Change both together.
    embedding_dim: int
    # --- Supabase ---------------------------------------------------------
    database_url: str
    supabase_url: str
    supabase_anon_key: str
    supabase_service_key: str
    storage_bucket: str
    # --- Optional integrations -------------------------------------------
    tavily_api_key: str
    langsmith_api_key: str
    langsmith_project: str
    # --- Limits and tuning -----------------------------------------------
    max_upload_mb: int
    chunk_size: int
    chunk_overlap: int
    embed_batch_size: int
    embed_concurrency: int
    doc_concurrency: int
    min_similarity: float
    db_pool_size: int
    summary_trigger: int      # summarise when the thread has more messages than this
    keep_recent: int          # messages kept verbatim after summarising
    max_context_messages: int  # hard cap on messages sent to the model
    auto_memory: bool
    memory_top_k: int
    memory_min_similarity: float
    max_memories: int

    def missing(self) -> list[str]:
        """Names of required settings that are not configured."""
        required = {
            "GEMINI_API_KEY": self.gemini_api_key,
            "GROQ_API_KEY": self.groq_api_key,
            "DATABASE_URL": self.database_url,
            "SUPABASE_URL": self.supabase_url,
            "SUPABASE_ANON_KEY": self.supabase_anon_key,
            "SUPABASE_SERVICE_KEY": self.supabase_service_key,
        }
        return [name for name, value in required.items() if not value]


def load_settings() -> Settings:
    chat_model = _read("CHAT_MODEL", "openai/gpt-oss-20b")
    extra = [m.strip() for m in _read("CHAT_MODEL_OPTIONS").split(",") if m.strip()]
    options = tuple(dict.fromkeys([chat_model, *extra]))
    return Settings(
        gemini_api_key=_read("GEMINI_API_KEY") or _read("GOOGLE_API_KEY"),
        groq_api_key=_read("GROQ_API_KEY"),
        chat_model=chat_model,
        model_options=options,
        embedding_model=_read("EMBEDDING_MODEL", "gemini-embedding-001"),
        embedding_dim=_read_int("EMBEDDING_DIM", 768),
        database_url=_read("DATABASE_URL"),
        supabase_url=_read("SUPABASE_URL"),
        supabase_anon_key=_read("SUPABASE_ANON_KEY"),
        supabase_service_key=_read("SUPABASE_SERVICE_KEY"),
        storage_bucket=_read("STORAGE_BUCKET", "documents"),
        tavily_api_key=_read("TAVILY_API_KEY"),
        langsmith_api_key=_read("LANGSMITH_API_KEY"),
        langsmith_project=_read("LANGSMITH_PROJECT", "intellichat-ai"),
        max_upload_mb=_read_int("MAX_UPLOAD_MB", 20),
        chunk_size=_read_int("CHUNK_SIZE", 1000),
        chunk_overlap=_read_int("CHUNK_OVERLAP", 150),
        embed_batch_size=_read_int("EMBED_BATCH_SIZE", 50),
        embed_concurrency=_read_int("EMBED_CONCURRENCY", 3),
        doc_concurrency=_read_int("DOC_CONCURRENCY", 3),
        min_similarity=_read_float("MIN_SIMILARITY", 0.40),
        db_pool_size=_read_int("DB_POOL_SIZE", 5),
        summary_trigger=_read_int("SUMMARY_TRIGGER", 24),
        keep_recent=_read_int("KEEP_RECENT", 8),
        max_context_messages=_read_int("MAX_CONTEXT_MESSAGES", 30),
        auto_memory=_read("AUTO_MEMORY", "true").lower() in ("1", "true", "yes", "on"),
        memory_top_k=_read_int("MEMORY_TOP_K", 5),
        memory_min_similarity=_read_float("MEMORY_MIN_SIMILARITY", 0.45),
        max_memories=_read_int("MAX_MEMORIES", 200),
    )


settings = load_settings()

# LangSmith tracing is optional: only switched on when a key is provided.
if settings.langsmith_api_key:
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)

# Chat modes: key -> label shown in the UI.
# General Chat covers everyday chat, coding help, study help and tool use: the model
# decides per message (see graph.py). Document Chat answers from uploaded documents.
MODES: dict[str, str] = {
    "general": "General Chat",
    "document": "Document Chat",
}

RESPONSE_LENGTHS = ("Concise", "Balanced", "Detailed")

LANGUAGES = (
    "English", "Hindi", "Spanish", "French", "German", "Portuguese",
    "Italian", "Japanese", "Korean", "Chinese", "Arabic", "Russian",
)

DEFAULT_PREFERENCES: dict = {
    "model": "",          # empty string means: use settings.chat_model (env / secrets)
    "temperature": 0.5,
    "max_tokens": 8192,
    "response_length": "Balanced",
    "language": "English",
    "system_instructions": "",
    "top_k": 5,
    "timezone": "UTC",
}

DOC_EXTENSIONS = ("pdf", "txt", "docx")
IMAGE_EXTENSIONS = ("png", "jpg", "jpeg", "webp")