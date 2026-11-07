"""PostgreSQL (Supabase) data access.

Every function takes the authenticated `user_id` and includes it in the SQL, so
one user can never read or modify another user's rows. All queries are
parameterised; related writes happen inside one transaction.
"""
from __future__ import annotations

import uuid
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Sequence

from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from config import DEFAULT_PREFERENCES, settings
from utils import escape_like

SCHEMA_PATH = Path(__file__).parent / "sql" / "schema.sql"
CHECKPOINT_TABLES = ("checkpoints", "checkpoint_blobs", "checkpoint_writes", "checkpoint_migrations")


# ---------------------------------------------------------------------------
# Connection pool
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_pool() -> ConnectionPool:
    """One shared, thread-safe pool for the whole process."""
    pool = ConnectionPool(
        conninfo=settings.database_url,
        min_size=1,
        max_size=settings.db_pool_size,
        # prepare_threshold=None disables server-side prepared statements, which
        # Supabase's pooled connections do not support.
        kwargs={"row_factory": dict_row, "prepare_threshold": None},
        open=False,
    )
    pool.open(wait=True, timeout=30)
    return pool


@contextmanager
def cursor() -> Iterator[Any]:
    """Cursor inside a transaction: commits on success, rolls back on error."""
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            yield cur


def _clean(row: dict | None) -> dict | None:
    """Convert UUID values to strings so they are easy to use as widget keys."""
    if row is None:
        return None
    return {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in row.items()}


def _clean_all(rows: Sequence[dict]) -> list[dict]:
    return [_clean(r) for r in rows]  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------
def ensure_schema() -> None:
    """Run sql/schema.sql if the app tables do not exist yet."""
    with cursor() as cur:
        cur.execute("select to_regclass('public.document_chunks') as t")
        if cur.fetchone()["t"] is not None:
            return
        cur.execute(SCHEMA_PATH.read_text(encoding="utf-8"))


def lock_down_checkpoint_tables() -> None:
    """Enable RLS (with no policies) on LangGraph's tables so the public API cannot read them."""
    with cursor() as cur:
        for table in CHECKPOINT_TABLES:
            cur.execute(sql.SQL("alter table if exists public.{} enable row level security")
                        .format(sql.Identifier(table)))


# ---------------------------------------------------------------------------
# Profile and preferences
# ---------------------------------------------------------------------------
def upsert_profile(user_id: str, email: str | None) -> None:
    with cursor() as cur:
        cur.execute(
            "insert into user_profiles (user_id, email) values (%s, %s) "
            "on conflict (user_id) do update set email = excluded.email",
            (user_id, email),
        )


def get_preferences(user_id: str) -> dict:
    with cursor() as cur:
        cur.execute("select settings from user_preferences where user_id = %s", (user_id,))
        row = cur.fetchone()
    return {**DEFAULT_PREFERENCES, **(row["settings"] if row else {})}


def save_preferences(user_id: str, prefs: dict) -> None:
    with cursor() as cur:
        cur.execute(
            "insert into user_preferences (user_id, settings) values (%s, %s) "
            "on conflict (user_id) do update set settings = excluded.settings, updated_at = now()",
            (user_id, Jsonb(prefs)),
        )


# ---------------------------------------------------------------------------
# Long-term memory (explicitly saved facts only)
# ---------------------------------------------------------------------------
def list_memories(user_id: str) -> list[dict]:
    with cursor() as cur:
        cur.execute("select id, content, created_at from user_memories "
                    "where user_id = %s order by created_at desc limit 50", (user_id,))
        return cur.fetchall()


def add_memory(user_id: str, content: str) -> None:
    content = " ".join(content.split())[:500]
    if not content:
        raise ValueError("Memory text is empty.")
    with cursor() as cur:
        cur.execute("insert into user_memories (user_id, content) values (%s, %s)", (user_id, content))


def delete_memory(user_id: str, memory_id: int) -> None:
    with cursor() as cur:
        cur.execute("delete from user_memories where id = %s and user_id = %s", (memory_id, user_id))


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------
def create_conversation(user_id: str, title: str = "New chat", mode: str = "general") -> dict:
    with cursor() as cur:
        cur.execute(
            "insert into conversations (user_id, title, mode) values (%s, %s, %s) returning *",
            (user_id, title[:120], mode),
        )
        return _clean(cur.fetchone())  # type: ignore[return-value]


def get_conversation(user_id: str, conversation_id: str) -> dict | None:
    with cursor() as cur:
        cur.execute("select * from conversations where id = %s and user_id = %s",
                    (conversation_id, user_id))
        return _clean(cur.fetchone())


def list_conversations(user_id: str, search: str = "", limit: int = 60) -> list[dict]:
    with cursor() as cur:
        if search.strip():
            pattern = f"%{escape_like(search.strip())}%"
            cur.execute(
                "select id, title, mode, updated_at from conversations c "
                "where c.user_id = %s and (c.title ilike %s or exists ("
                "  select 1 from messages m where m.conversation_id = c.id and m.content ilike %s)) "
                "order by c.updated_at desc limit %s",
                (user_id, pattern, pattern, limit),
            )
        else:
            cur.execute(
                "select id, title, mode, updated_at from conversations "
                "where user_id = %s order by updated_at desc limit %s",
                (user_id, limit),
            )
        return _clean_all(cur.fetchall())


def rename_conversation(user_id: str, conversation_id: str, title: str) -> None:
    title = " ".join(title.split())[:120] or "Untitled"
    with cursor() as cur:
        cur.execute("update conversations set title = %s where id = %s and user_id = %s",
                    (title, conversation_id, user_id))


def set_conversation_mode(user_id: str, conversation_id: str, mode: str) -> None:
    with cursor() as cur:
        cur.execute("update conversations set mode = %s where id = %s and user_id = %s",
                    (mode, conversation_id, user_id))


def delete_conversation(user_id: str, conversation_id: str) -> dict:
    """Delete a conversation. Messages, documents and chunks go with it (ON DELETE CASCADE).

    Returns what the caller must clean up outside the database:
    the LangGraph thread id and the Storage paths of attached documents.
    """
    with cursor() as cur:
        cur.execute("select thread_id from conversations where id = %s and user_id = %s",
                    (conversation_id, user_id))
        row = cur.fetchone()
        if row is None:
            return {"thread_ids": [], "storage_paths": []}
        cur.execute("select storage_path from documents "
                    "where conversation_id = %s and user_id = %s and storage_path is not null",
                    (conversation_id, user_id))
        paths = [r["storage_path"] for r in cur.fetchall()]
        cur.execute("delete from conversations where id = %s and user_id = %s",
                    (conversation_id, user_id))
    return {"thread_ids": [str(row["thread_id"])], "storage_paths": paths}


def delete_all_conversations(user_id: str) -> dict:
    """Delete every conversation of the user (knowledge-base documents are kept)."""
    with cursor() as cur:
        cur.execute("select thread_id from conversations where user_id = %s", (user_id,))
        threads = [str(r["thread_id"]) for r in cur.fetchall()]
        cur.execute("select storage_path from documents where user_id = %s "
                    "and conversation_id is not null and storage_path is not null", (user_id,))
        paths = [r["storage_path"] for r in cur.fetchall()]
        cur.execute("delete from conversations where user_id = %s", (user_id,))
    return {"thread_ids": threads, "storage_paths": paths}


def clear_conversation(user_id: str, conversation_id: str) -> str | None:
    """Delete a conversation's messages and start a fresh LangGraph thread.

    Returns the old thread id so its checkpoints can be removed.
    """
    with cursor() as cur:
        cur.execute("select thread_id from conversations where id = %s and user_id = %s",
                    (conversation_id, user_id))
        row = cur.fetchone()
        if row is None:
            return None
        cur.execute("delete from messages where conversation_id = %s and user_id = %s",
                    (conversation_id, user_id))
        cur.execute("update conversations set thread_id = gen_random_uuid(), updated_at = now() "
                    "where id = %s and user_id = %s", (conversation_id, user_id))
    return str(row["thread_id"])


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------
def add_message(user_id: str, conversation_id: str, role: str, content: str, *,
                mode: str | None = None, sources: list | None = None,
                tools_used: list | None = None, tokens_in: int | None = None,
                tokens_out: int | None = None, latency_ms: int | None = None) -> dict:
    """Insert a message, only if the conversation belongs to `user_id`."""
    with cursor() as cur:
        cur.execute(
            "insert into messages (conversation_id, user_id, role, content, mode, sources, "
            "tools_used, tokens_in, tokens_out, latency_ms) "
            "select c.id, c.user_id, %s, %s, %s, %s, %s, %s, %s, %s "
            "from conversations c where c.id = %s and c.user_id = %s returning *",
            (role, content, mode, Jsonb(sources or []), Jsonb(tools_used or []),
             tokens_in, tokens_out, latency_ms, conversation_id, user_id),
        )
        row = cur.fetchone()
        if row is None:
            raise PermissionError("Conversation not found for this user.")
        cur.execute("update conversations set updated_at = now() where id = %s", (conversation_id,))
        return _clean(row)  # type: ignore[return-value]


def list_messages(user_id: str, conversation_id: str) -> list[dict]:
    with cursor() as cur:
        cur.execute("select * from messages where conversation_id = %s and user_id = %s order by id",
                    (conversation_id, user_id))
        return _clean_all(cur.fetchall())


def delete_last_assistant_message(user_id: str, conversation_id: str) -> bool:
    with cursor() as cur:
        cur.execute(
            "delete from messages where id = ("
            "  select id from messages where conversation_id = %s and user_id = %s "
            "  order by id desc limit 1) and role = 'assistant'",
            (conversation_id, user_id),
        )
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Documents and chunks
# ---------------------------------------------------------------------------
def find_duplicate(user_id: str, conversation_id: str | None, checksum: str) -> dict | None:
    """A ready document with the same content already visible in this scope."""
    with cursor() as cur:
        cur.execute(
            "select id, filename, conversation_id from documents "
            "where user_id = %s and checksum = %s and status = 'ready' "
            "and (conversation_id is null or conversation_id = %s) limit 1",
            (user_id, checksum, conversation_id),
        )
        return _clean(cur.fetchone())


def create_document(doc_id: str, user_id: str, conversation_id: str | None, filename: str,
                    mime_type: str, size_bytes: int, checksum: str, storage_path: str) -> bool:
    """Insert a 'processing' document. Returns False if an identical upload already exists."""
    with cursor() as cur:
        # Clear leftovers from a crashed earlier attempt so the file can be re-uploaded.
        cur.execute("delete from documents where user_id = %s and checksum = %s "
                    "and status <> 'ready' and created_at < now() - interval '30 minutes'",
                    (user_id, checksum))
        cur.execute(
            "insert into documents (id, user_id, conversation_id, filename, mime_type, size_bytes, "
            "checksum, storage_path) values (%s, %s, %s, %s, %s, %s, %s, %s) "
            "on conflict do nothing returning id",
            (doc_id, user_id, conversation_id, filename, mime_type, size_bytes, checksum, storage_path),
        )
        return cur.fetchone() is not None


def insert_chunks(user_id: str, doc_id: str, rows: list[tuple[int, int | None, str, str]]) -> None:
    """Batch-insert (chunk_index, page, content, embedding_literal) rows and mark the doc ready."""
    pages = {r[1] for r in rows if r[1] is not None}
    with cursor() as cur:
        cur.execute("select 1 from documents where id = %s and user_id = %s", (doc_id, user_id))
        if cur.fetchone() is None:
            raise PermissionError("Document not found for this user.")
        cur.executemany(
            "insert into document_chunks (document_id, user_id, chunk_index, page, content, embedding) "
            "values (%s, %s, %s, %s, %s, %s::vector)",
            [(doc_id, user_id, idx, page, content, emb) for idx, page, content, emb in rows],
        )
        cur.execute("update documents set status = 'ready', chunk_count = %s, page_count = %s "
                    "where id = %s and user_id = %s",
                    (len(rows), len(pages) or None, doc_id, user_id))


def list_documents(user_id: str, conversation_id: str | None = None, *, everything: bool = False) -> list[dict]:
    """Ready documents. Default: knowledge base + (optionally) one chat's attachments."""
    with cursor() as cur:
        if everything:
            cur.execute(
                "select d.id, d.filename, d.size_bytes, d.page_count, d.chunk_count, d.created_at, "
                "d.conversation_id, c.title as conversation_title "
                "from documents d left join conversations c on c.id = d.conversation_id "
                "where d.user_id = %s and d.status = 'ready' order by d.created_at desc",
                (user_id,),
            )
        else:
            cur.execute(
                "select id, filename, size_bytes, page_count, chunk_count, created_at, conversation_id "
                "from documents where user_id = %s and status = 'ready' "
                "and (conversation_id is null or conversation_id = %s) order by created_at desc",
                (user_id, conversation_id),
            )
        return _clean_all(cur.fetchall())


def has_documents(user_id: str, conversation_id: str | None) -> bool:
    with cursor() as cur:
        cur.execute("select 1 from documents where user_id = %s and status = 'ready' "
                    "and (conversation_id is null or conversation_id = %s) limit 1",
                    (user_id, conversation_id))
        return cur.fetchone() is not None


def delete_document(user_id: str, doc_id: str) -> str | None:
    """Delete a document and (by cascade) its chunks. Returns its Storage path for cleanup."""
    with cursor() as cur:
        cur.execute("delete from documents where id = %s and user_id = %s returning storage_path",
                    (doc_id, user_id))
        row = cur.fetchone()
    return row["storage_path"] if row else None


def search_chunks(user_id: str, conversation_id: str | None, embedding: str, k: int,
                  doc_ids: list[str] | None = None) -> list[dict]:
    """Nearest chunks by cosine similarity, restricted to this user's visible documents."""
    query = (
        "select c.content, c.page, c.chunk_index, d.id as document_id, d.filename, "
        "1 - (c.embedding <=> %s::vector) as similarity "
        "from document_chunks c join documents d on d.id = c.document_id "
        "where c.user_id = %s and d.user_id = %s and d.status = 'ready' "
        "and (d.conversation_id is null or d.conversation_id = %s) "
    )
    params: list[Any] = [embedding, user_id, user_id, conversation_id]
    if doc_ids:
        query += "and c.document_id = any(%s) "
        params.append([uuid.UUID(d) for d in doc_ids])
    query += "order by c.embedding <=> %s::vector limit %s"
    params += [embedding, k]
    with cursor() as cur:
        cur.execute(query, params)
        return _clean_all(cur.fetchall())


# ---------------------------------------------------------------------------
# Monitoring and export
# ---------------------------------------------------------------------------
def usage_summary(user_id: str) -> dict:
    with cursor() as cur:
        cur.execute(
            "select count(*) filter (where role = 'assistant') as replies, "
            "coalesce(sum(tokens_in), 0) as tokens_in, coalesce(sum(tokens_out), 0) as tokens_out, "
            "coalesce(avg(latency_ms) filter (where role = 'assistant'), 0)::int as avg_latency_ms "
            "from messages where user_id = %s", (user_id,))
        summary = cur.fetchone()
        cur.execute(
            "select t.tool, count(*) as uses from messages m, "
            "jsonb_array_elements_text(m.tools_used) as t(tool) "
            "where m.user_id = %s group by t.tool order by uses desc", (user_id,))
        summary["tools"] = cur.fetchall()
    return summary


def export_all(user_id: str) -> dict:
    """Everything the user owns that is worth backing up (no embeddings, no files)."""
    with cursor() as cur:
        cur.execute("select id, title, mode, created_at, updated_at from conversations "
                    "where user_id = %s order by created_at", (user_id,))
        conversations = _clean_all(cur.fetchall())
        cur.execute("select conversation_id, role, content, mode, sources, tools_used, created_at "
                    "from messages where user_id = %s order by id", (user_id,))
        by_conv: dict[str, list] = {}
        for m in cur.fetchall():
            by_conv.setdefault(str(m.pop("conversation_id")), []).append(m)
        cur.execute("select content, created_at from user_memories where user_id = %s", (user_id,))
        memories = cur.fetchall()
        cur.execute("select filename, size_bytes, created_at from documents where user_id = %s", (user_id,))
        documents = cur.fetchall()
    for conv in conversations:
        conv["messages"] = by_conv.get(conv["id"], [])
    return {"conversations": conversations, "memories": memories, "documents": documents,
            "preferences": get_preferences(user_id)}
