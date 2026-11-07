"""Document ingestion (extract -> chunk -> embed -> store) and retrieval for RAG.

Concurrency model (all on the background event loop, see chatbot.py):
  * several documents are processed at the same time, capped by DOC_CONCURRENCY;
  * blocking work (PDF parsing, database, Storage) runs in worker threads;
  * Gemini embedding calls are batched and capped by EMBED_CONCURRENCY;
  * one failing document never discards the results of the others.
"""
from __future__ import annotations

import asyncio
import io
import logging
import uuid
from dataclasses import dataclass
from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter

import database as db
from chatbot import get_embeddings
from config import DOC_EXTENSIONS, settings
from utils import (
    file_checksum, file_extension, friendly_error, retry_async, safe_filename, vector_literal,
)

logger = logging.getLogger("intellichat.rag")

MIME_TYPES = {
    "pdf": "application/pdf",
    "txt": "text/plain",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
MAX_CHUNKS_PER_DOCUMENT = 2000


@dataclass
class IngestResult:
    filename: str
    status: str  # "ready", "duplicate" or "failed"
    message: str = ""
    chunks: int = 0


# ---------------------------------------------------------------------------
# Supabase Storage
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _storage():
    """Service-role client; the key stays on the server and is never sent to the browser."""
    from supabase import create_client

    client = create_client(settings.supabase_url, settings.supabase_service_key)
    return client.storage.from_(settings.storage_bucket)


def storage_upload(path: str, data: bytes, mime_type: str) -> None:
    _storage().upload(path=path, file=data, file_options={"content-type": mime_type, "upsert": "true"})


def storage_remove(paths: list[str]) -> None:
    """Best-effort removal of files from Storage."""
    paths = [p for p in paths if p]
    if not paths:
        return
    try:
        _storage().remove(paths)
    except Exception:  # noqa: BLE001 - an orphaned file is harmless; do not block deletion
        logger.warning("Could not remove %d file(s) from Storage", len(paths), exc_info=True)


# ---------------------------------------------------------------------------
# Extraction and chunking (pure, CPU-bound -> run in threads)
# ---------------------------------------------------------------------------
def extract_pages(filename: str, data: bytes) -> list[tuple[int | None, str]]:
    """Return [(page_number_or_None, text)] with empty pages removed."""
    ext = file_extension(filename)
    pages: list[tuple[int | None, str]]
    if ext == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("The PDF is password-protected.")
        pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    elif ext == "docx":
        from docx import Document

        document = Document(io.BytesIO(data))
        parts = [p.text for p in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text.strip() for cell in row.cells))
        pages = [(None, "\n".join(parts))]
    elif ext == "txt":
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = data.decode("latin-1")
        pages = [(None, text)]
    else:
        raise ValueError(f"Unsupported file type '.{ext}'. Use PDF, TXT or DOCX.")
    cleaned = [(n, t.replace("\x00", "").strip()) for n, t in pages]  # NUL bytes break PostgreSQL text
    return [(n, t) for n, t in cleaned if t]


def chunk_pages(pages: list[tuple[int | None, str]]) -> list[tuple[int | None, str]]:
    """Split each page into overlapping chunks, keeping the page number."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size,
                                              chunk_overlap=settings.chunk_overlap)
    chunks: list[tuple[int | None, str]] = []
    for page, text in pages:
        chunks.extend((page, piece) for piece in splitter.split_text(text) if piece.strip())
    return chunks


# ---------------------------------------------------------------------------
# Embedding (batched, concurrency-limited, with retry)
# ---------------------------------------------------------------------------
_embed_semaphore: asyncio.Semaphore | None = None


def _semaphore() -> asyncio.Semaphore:
    """Created lazily on the background loop so every task shares one limit."""
    global _embed_semaphore
    if _embed_semaphore is None:
        _embed_semaphore = asyncio.Semaphore(settings.embed_concurrency)
    return _embed_semaphore


async def embed_texts(texts: list[str]) -> list[list[float]]:
    embedder = get_embeddings("RETRIEVAL_DOCUMENT")
    size = settings.embed_batch_size
    batches = [texts[i:i + size] for i in range(0, len(texts), size)]

    async def one(batch: list[str]) -> list[list[float]]:
        async with _semaphore():
            return await retry_async(lambda: embedder.aembed_documents(batch))

    results = await asyncio.gather(*(one(b) for b in batches), return_exceptions=True)
    for item in results:
        if isinstance(item, BaseException):
            raise item
    return [vector for batch in results for vector in batch]  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
async def _ingest_one(user_id: str, conversation_id: str | None, filename: str, data: bytes) -> IngestResult:
    ext = file_extension(filename)
    if ext not in DOC_EXTENSIONS:
        return IngestResult(filename, "failed", "Unsupported file type (use PDF, TXT or DOCX).")
    if not data:
        return IngestResult(filename, "failed", "The file is empty.")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        return IngestResult(filename, "failed", f"The file is larger than {settings.max_upload_mb} MB.")

    checksum = file_checksum(data)
    duplicate = await asyncio.to_thread(db.find_duplicate, user_id, conversation_id, checksum)
    if duplicate:
        return IngestResult(filename, "duplicate", f"Already uploaded as '{duplicate['filename']}'.")

    doc_id = str(uuid.uuid4())
    storage_path = f"{user_id}/{doc_id}/{safe_filename(filename)}"
    mime = MIME_TYPES[ext]
    created = await asyncio.to_thread(db.create_document, doc_id, user_id, conversation_id, filename,
                                      mime, len(data), checksum, storage_path)
    if not created:  # another upload of the same file won the race
        return IngestResult(filename, "duplicate", "This file is already being processed or uploaded.")

    try:
        pages = await asyncio.to_thread(extract_pages, filename, data)
        chunks = await asyncio.to_thread(chunk_pages, pages) if pages else []
        if not chunks:
            raise ValueError("No readable text found (a scanned PDF needs OCR, which is not supported).")
        if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
            raise ValueError(f"The document is too large ({len(chunks)} chunks; limit {MAX_CHUNKS_PER_DOCUMENT}).")
        # Uploading the original file and creating embeddings are independent: run them together.
        _, vectors = await asyncio.gather(
            asyncio.to_thread(storage_upload, storage_path, data, mime),
            embed_texts([text for _, text in chunks]),
        )
        rows = [(i, page, text, vector_literal(vec)) for i, ((page, text), vec) in enumerate(zip(chunks, vectors))]
        await asyncio.to_thread(db.insert_chunks, user_id, doc_id, rows)
    except BaseException as exc:
        # Roll back so the file can be uploaded again; also covers cancellation.
        await asyncio.shield(asyncio.to_thread(_cleanup_failed, user_id, doc_id, storage_path))
        if isinstance(exc, Exception):
            logger.warning("Ingest failed for a document: %s", type(exc).__name__)
            return IngestResult(filename, "failed", friendly_error(exc))
        raise
    return IngestResult(filename, "ready", f"{len(chunks)} chunks indexed.", len(chunks))


def _cleanup_failed(user_id: str, doc_id: str, storage_path: str) -> None:
    db.delete_document(user_id, doc_id)
    storage_remove([storage_path])


async def ingest_files(user_id: str, conversation_id: str | None,
                       files: list[tuple[str, bytes]], emit=None) -> list[IngestResult]:
    """Process several files concurrently. `emit(("progress", done, total, message))` reports progress."""
    gate = asyncio.Semaphore(settings.doc_concurrency)
    total, finished = len(files), 0

    async def worker(filename: str, data: bytes) -> IngestResult:
        nonlocal finished
        async with gate:
            try:
                result = await _ingest_one(user_id, conversation_id, filename, data)
            except Exception as exc:  # noqa: BLE001 - keep the other documents going
                result = IngestResult(filename, "failed", friendly_error(exc))
        finished += 1
        if emit:
            emit(("progress", finished, total, f"{filename}: {result.status}"))
        return result

    if emit:
        emit(("progress", 0, total, "Starting…"))
    return list(await asyncio.gather(*(worker(name, data) for name, data in files)))


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
async def retrieve(user_id: str, conversation_id: str | None, query: str, k: int,
                   doc_ids: list[str] | None = None) -> list[dict]:
    """Return the most relevant chunks above the similarity threshold (possibly none)."""
    if not query.strip():
        return []
    if not await asyncio.to_thread(db.has_documents, user_id, conversation_id):
        return []  # nothing to search: skip the embedding call and save quota
    embedder = get_embeddings("RETRIEVAL_QUERY")
    vector = await retry_async(lambda: embedder.aembed_query(query))
    hits = await asyncio.to_thread(db.search_chunks, user_id, conversation_id,
                                   vector_literal(vector), max(1, min(k, 20)), doc_ids)
    return [h for h in hits if h["similarity"] >= settings.min_similarity]


def format_context(hits: list[dict]) -> tuple[str, list[dict]]:
    """Build the numbered excerpt block for the prompt and the citation list for the UI."""
    blocks, sources = [], []
    for n, hit in enumerate(hits, 1):
        where = hit["filename"] + (f", page {hit['page']}" if hit.get("page") else "")
        blocks.append(f"[{n}] ({where})\n{hit['content'][:1500]}")
        sources.append({"n": n, "filename": hit["filename"], "page": hit.get("page"),
                        "similarity": round(float(hit["similarity"]), 3)})
    return "\n\n".join(blocks), sources
