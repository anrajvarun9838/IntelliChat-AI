"""Gemini clients, the background asyncio runtime, and turn execution.

Streamlit scripts are synchronous and re-run on every interaction, so they must
never call `asyncio.run()` repeatedly. Instead this module starts ONE background
thread that owns ONE event loop (created once through `st.cache_resource`). The
LangGraph workflow, the async Postgres checkpointer and all async work run on
that loop. The Streamlit script talks to it through a `Job`: work is submitted
with `run_coroutine_threadsafe`, and results/stream chunks come back through a
thread-safe queue that the script can iterate.
"""
from __future__ import annotations

import asyncio
import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Awaitable, Callable

import streamlit as st
from langchain_core.messages import (
    AIMessage, AIMessageChunk, HumanMessage, RemoveMessage, ToolMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.types import Command

import database as db
from config import settings
from utils import setup_logging, text_of

setup_logging()
logger = logging.getLogger("intellichat.chatbot")


# ---------------------------------------------------------------------------
# Gemini clients (cached, shared, contain no per-user data)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=16)
def get_llm(model: str, temperature: float, max_output_tokens: int) -> ChatGoogleGenerativeAI:
    """Chat model client. Cached per (model, temperature, max tokens)."""
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.gemini_api_key,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        max_retries=1,  # the graph's RetryPolicy handles the rest with backoff
    )


@lru_cache(maxsize=4)
def get_embeddings(task_type: str) -> GoogleGenerativeAIEmbeddings:
    """Embedding client. task_type: RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY."""
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.gemini_api_key,
        task_type=task_type,
        output_dimensionality=settings.embedding_dim,
    )


# ---------------------------------------------------------------------------
# Background runtime
# ---------------------------------------------------------------------------
class Runtime:
    """Owns the background event loop, the async checkpoint pool and the compiled graph."""

    def __init__(self) -> None:
        # psycopg's async mode cannot use Windows' default ProactorEventLoop.
        self.loop = asyncio.SelectorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="intellichat-loop", daemon=True)
        self.thread.start()
        self.pool = None
        self.saver = None
        self.graph = None

    def run(self, coro: Awaitable, timeout: float | None = 60):
        """Run a coroutine on the background loop and wait for its result."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)  # type: ignore[arg-type]

    async def _setup(self) -> None:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool

        from graph import build_graph

        self.pool = AsyncConnectionPool(
            conninfo=settings.database_url,
            min_size=1,
            max_size=4,
            kwargs={"autocommit": True, "prepare_threshold": None, "row_factory": dict_row},
            open=False,
        )
        await self.pool.open(wait=True, timeout=30)
        self.saver = AsyncPostgresSaver(self.pool)
        await self.saver.setup()  # creates the checkpoint tables if they do not exist
        self.graph = build_graph(self.saver)

    def start(self) -> None:
        db.ensure_schema()
        self.run(self._setup(), timeout=90)
        db.lock_down_checkpoint_tables()


@st.cache_resource(show_spinner="Starting IntelliChat AI…")
def get_runtime() -> Runtime:
    """Create the runtime once per server process (call from the Streamlit script thread)."""
    runtime = Runtime()
    runtime.start()
    return runtime


# ---------------------------------------------------------------------------
# Jobs: run a coroutine on the loop, stream events back to the Streamlit thread
# ---------------------------------------------------------------------------
class _Done:
    def __init__(self, value: Any = None, error: BaseException | None = None) -> None:
        self.value = value
        self.error = error


class Job:
    """Handle for a coroutine running on the background loop.

    The coroutine receives an `emit(event)` callable. The Streamlit script
    iterates `events()`; afterwards `value` / `error` hold the outcome.
    """

    def __init__(self, runtime: Runtime, fn: Callable[[Callable[[Any], None]], Awaitable[Any]]) -> None:
        self._queue: queue.Queue = queue.Queue()
        self.value: Any = None
        self.error: BaseException | None = None
        self.finished = False
        self._future = asyncio.run_coroutine_threadsafe(self._run(fn), runtime.loop)

    async def _run(self, fn) -> None:
        try:
            self._queue.put(_Done(value=await fn(self._queue.put)))
        except asyncio.CancelledError:
            self._queue.put(_Done(error=RuntimeError("cancelled")))
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI by the caller
            logger.exception("Job failed")
            self._queue.put(_Done(error=exc))

    def events(self):
        """Yield events until the job finishes. Cancels the job if the consumer stops early."""
        try:
            while True:
                item = self._queue.get()
                if isinstance(item, _Done):
                    self.value, self.error, self.finished = item.value, item.error, True
                    return
                yield item
        finally:
            if not self.finished:
                self.cancel()

    def cancel(self) -> None:
        self._future.cancel()


def start_job(fn: Callable[[Callable[[Any], None]], Awaitable[Any]]) -> Job:
    return Job(get_runtime(), fn)


# ---------------------------------------------------------------------------
# Running a chat turn through the graph
# ---------------------------------------------------------------------------
@dataclass
class TurnResult:
    text: str = ""
    sources: list = field(default_factory=list)
    tools_used: list = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    error: str = ""


def _as_list(value: Any) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


async def _run_graph(graph, graph_input, config: dict, emit) -> TurnResult:
    """Stream one graph run. Emits ("text", chunk) and ("status", message) events."""
    result = TurnResult()
    started = time.perf_counter()
    current_message_id = None

    async for mode, payload in graph.astream(graph_input, config, stream_mode=["messages", "updates"]):
        if mode == "messages":
            chunk, meta = payload
            if meta.get("langgraph_node") not in ("respond", "agent") or not isinstance(chunk, AIMessageChunk):
                continue  # ignore summarisation and other internal model calls
            piece = text_of(chunk.content)
            if not piece:
                continue
            if current_message_id and chunk.id != current_message_id and result.text:
                piece = "\n\n" + piece  # new model call after a tool round
            current_message_id = chunk.id
            result.text += piece
            emit(("text", piece))
        elif mode == "updates":
            for node, update in payload.items():
                logger.info("graph node finished: %s", node)  # basic LangGraph tracing
                if not isinstance(update, dict):
                    continue
                if update.get("sources"):
                    result.sources = update["sources"]
                if update.get("error"):
                    result.error = update["error"]
                for message in _as_list(update.get("messages")):
                    if isinstance(message, AIMessage):
                        usage = message.usage_metadata or {}
                        result.tokens_in += int(usage.get("input_tokens", 0) or 0)
                        result.tokens_out += int(usage.get("output_tokens", 0) or 0)
                        for call in message.tool_calls:
                            emit(("status", f"Using tool: {call['name']}…"))
                    elif isinstance(message, ToolMessage) and message.name:
                        if message.name not in result.tools_used:
                            result.tools_used.append(message.name)
                            logger.info("tool executed: %s", message.name)

    result.latency_ms = int((time.perf_counter() - started) * 1000)
    logger.info("turn finished in %d ms (tokens in=%d out=%d, tools=%s)", result.latency_ms,
                result.tokens_in, result.tokens_out, result.tools_used)
    return result


def start_turn(graph_input: dict | Command, config: dict) -> Job:
    """Start streaming a graph run. `graph_input` is {"messages": [...]} or Command(resume=...)."""
    runtime = get_runtime()

    async def fn(emit):
        return await _run_graph(runtime.graph, graph_input, config, emit)

    return Job(runtime, fn)


def build_human_message(text: str, images: list[tuple[str, str]] | None = None) -> HumanMessage:
    """Create the user's message, optionally with images as (mime_type, base64) pairs."""
    if not images:
        return HumanMessage(content=text)
    blocks: list[dict] = [{"type": "text", "text": text or "Describe this image."}]
    for mime, b64 in images:
        blocks.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    return HumanMessage(content=blocks)


def get_pending_approval(config: dict) -> dict | None:
    """Return the payload of a paused human-approval step, if the thread is waiting for one."""
    runtime = get_runtime()

    async def fn():
        state = await runtime.graph.aget_state(config)
        for task in state.tasks:
            for interrupt in getattr(task, "interrupts", ()) or ():
                return interrupt.value
        return None

    return runtime.run(fn(), timeout=30)


def rewind_last_turn(config: dict) -> HumanMessage | None:
    """Remove the last user message and everything after it from the graph state.

    Used by "Regenerate": the same question is then sent again on a clean state.
    """
    runtime = get_runtime()

    async def fn():
        state = await runtime.graph.aget_state(config)
        messages = state.values.get("messages", [])
        index = next((i for i in range(len(messages) - 1, -1, -1)
                      if isinstance(messages[i], HumanMessage)), None)
        if index is None:
            return None
        await runtime.graph.aupdate_state(
            config, {"messages": [RemoveMessage(id=m.id) for m in messages[index:] if m.id]})
        return messages[index]

    last = runtime.run(fn(), timeout=30)
    return HumanMessage(content=last.content) if last is not None else None


def delete_threads(thread_ids: list[str]) -> None:
    """Delete LangGraph checkpoints for the given threads (best effort)."""
    runtime = get_runtime()
    for thread_id in thread_ids:
        try:
            runtime.run(runtime.saver.adelete_thread(thread_id), timeout=30)
        except Exception:  # noqa: BLE001 - leftover checkpoints are harmless
            logger.warning("Could not delete checkpoints for a thread", exc_info=True)


def graph_mermaid() -> str:
    return get_runtime().graph.get_graph().draw_mermaid()
