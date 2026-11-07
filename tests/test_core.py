"""Core tests. They never call Gemini or Supabase.

Run from the project folder:  python -m pytest -q
Tests that need an optional package skip themselves if it is not installed.
"""
import asyncio
import io
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import utils  # noqa: E402


# ---------------------------------------------------------------------------
# utils (pure Python)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("expr,expected", [
    ("2 + 3 * 4", "14"),
    ("(12.5 * 8) / 3", "33.3333333333"),
    ("2 ^ 10", "1024"),
    ("sqrt(49) + 1", "8"),
    ("-5 + 2", "-3"),
    ("10 // 3", "3"),
    ("round(3.14159, 2)", "3.14"),
])
def test_calculator_valid(expr, expected):
    assert utils.safe_eval(expr) == expected


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hi')",
    "open('x')",
    "2 ** 100000",
    "1 / 0",
    "abs",
    "'a' * 3",
    "",
    "x + 1",
    "9" * 300,
])
def test_calculator_rejects_unsafe_or_invalid(expr):
    with pytest.raises(ValueError):
        utils.safe_eval(expr)


def test_text_of_handles_strings_and_blocks():
    assert utils.text_of("hello") == "hello"
    blocks = [{"type": "thinking", "thinking": "hidden"}, {"type": "text", "text": "Hi"}, {"type": "text", "text": "!"}]
    assert utils.text_of(blocks) == "Hi!"
    assert utils.text_of(None) == ""


def test_friendly_error_messages():
    assert "API key" in utils.friendly_error(Exception("400 API key not valid. Please pass a valid API key."))
    assert "rate limit" in utils.friendly_error(Exception("429 RESOURCE_EXHAUSTED")).lower()
    assert "free-tier" in utils.friendly_error(Exception("429 quota exceeded, limit: 0"))
    assert "unavailable" in utils.friendly_error(Exception("404 models/foo is not found for API version v1beta")).lower()


def test_is_transient():
    assert utils.is_transient(Exception("503 UNAVAILABLE"))
    assert utils.is_transient(Exception("429 Too Many Requests"))
    assert not utils.is_transient(Exception("400 API key not valid"))
    assert not utils.is_transient(Exception("429 quota limit: 0"))


def test_retry_async_retries_transient_then_succeeds():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("503 unavailable")
        return "ok"

    result = asyncio.run(utils.retry_async(flaky, attempts=4, base_delay=0.01, max_delay=0.02))
    assert result == "ok" and calls["n"] == 3


def test_retry_async_does_not_retry_permanent_errors():
    calls = {"n": 0}

    async def broken():
        calls["n"] += 1
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        asyncio.run(utils.retry_async(broken, attempts=4, base_delay=0.01))
    assert calls["n"] == 1


def test_file_helpers():
    assert utils.safe_filename("My Report (final) é.pdf") == "My_Report_final_.pdf"
    assert utils.safe_filename("///") == "file"
    assert utils.file_checksum(b"abc") == utils.file_checksum(b"abc") != utils.file_checksum(b"abd")
    assert utils.file_extension("A.B.PDF") == "pdf"
    assert utils.vector_literal([0.5, 1]) == "[0.5000000,1.0000000]"
    assert utils.escape_like("50%_off") == "50\\%\\_off"


def test_group_by_date():
    now = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
    rows = [{"id": i, "updated_at": now - timedelta(days=d)} for i, d in enumerate([0, 1, 3, 20, 90])]
    groups = dict(utils.group_by_date(rows, now=now))
    assert [r["id"] for r in groups["Today"]] == [0]
    assert [r["id"] for r in groups["Yesterday"]] == [1]
    assert [r["id"] for r in groups["Previous 7 days"]] == [2]
    assert [r["id"] for r in groups["Previous 30 days"]] == [3]
    assert [r["id"] for r in groups["Older"]] == [4]


def _messages():
    stamp = datetime(2026, 11, 6, 9, 30, tzinfo=timezone.utc)
    return [{"role": "user", "content": "Hi <b>", "created_at": stamp},
            {"role": "assistant", "content": "Hello & welcome", "created_at": stamp}]


def test_export_txt():
    data = utils.export_txt("My chat", _messages()).decode("utf-8")
    assert "My chat" in data and "You:" in data and "Hello & welcome" in data


def test_export_pdf_is_valid_pdf():
    pytest.importorskip("reportlab")
    assert utils.export_pdf("My chat", _messages()).startswith(b"%PDF")


# ---------------------------------------------------------------------------
# RAG helpers
# ---------------------------------------------------------------------------
def _import_rag():
    for name in ("streamlit", "langchain_google_genai", "langchain_text_splitters", "psycopg_pool"):
        pytest.importorskip(name)
    import rag
    return rag


def test_extract_txt_and_chunking_keeps_pages():
    rag = _import_rag()
    pages = rag.extract_pages("notes.txt", ("Hello world. " * 400).encode("utf-8"))
    assert len(pages) == 1 and pages[0][0] is None
    chunks = rag.chunk_pages(pages)
    assert len(chunks) > 1 and all(len(text) <= 1200 for _, text in chunks)


def test_extract_pdf_reports_page_numbers():
    rag = _import_rag()
    pytest.importorskip("pypdf")
    canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(100, 750, "First page text")
    pdf.showPage()
    pdf.drawString(100, 750, "Second page text")
    pdf.save()
    pages = rag.extract_pages("doc.pdf", buffer.getvalue())
    assert [p for p, _ in pages] == [1, 2]
    assert "Second page" in pages[1][1]


def test_extract_docx():
    rag = _import_rag()
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("Quarterly report")
    buffer = io.BytesIO()
    document.save(buffer)
    pages = rag.extract_pages("r.docx", buffer.getvalue())
    assert "Quarterly report" in pages[0][1]


def test_extract_rejects_unknown_type():
    rag = _import_rag()
    with pytest.raises(ValueError):
        rag.extract_pages("movie.mp4", b"123")


def test_format_context_numbers_sources():
    rag = _import_rag()
    hits = [{"content": "Alpha", "filename": "a.pdf", "page": 3, "similarity": 0.81234},
            {"content": "Beta", "filename": "b.txt", "page": None, "similarity": 0.5}]
    context, sources = rag.format_context(hits)
    assert "[1] (a.pdf, page 3)" in context and "[2] (b.txt)" in context
    assert sources[0] == {"n": 1, "filename": "a.pdf", "page": 3, "similarity": 0.812}
    assert rag.format_context([]) == ("", [])


# ---------------------------------------------------------------------------
# Tools and graph (mocked model: no API calls)
# ---------------------------------------------------------------------------
def _import_graph():
    for name in ("streamlit", "langchain_google_genai", "langgraph", "psycopg_pool", "httpx"):
        pytest.importorskip(name)
    import graph
    return graph


def test_calculator_tool_and_datetime_tool():
    _import_graph()
    import tools
    assert tools.calculator.invoke({"expression": "6 * 7"}) == "42"
    assert tools.calculator.invoke({"expression": "__import__('os')"}).startswith("Calculator error")
    assert "Unknown timezone" in tools.current_datetime.invoke({"timezone_name": "Not/AZone"})


def test_graph_routing_functions():
    graph = _import_graph()
    from langchain_core.messages import AIMessage
    cfg = lambda mode: {"configurable": {"mode": mode}}  # noqa: E731
    assert graph.route_by_mode({}, cfg("document")) == "retrieve"
    assert graph.route_by_mode({}, cfg("agent")) == "agent"
    for mode in ("general", "coding", "study"):
        assert graph.route_by_mode({}, cfg(mode)) == "respond"

    plain = AIMessage(content="done")
    safe_call = AIMessage(content="", tool_calls=[{"name": "calculator", "args": {"expression": "1+1"}, "id": "1"}])
    risky_call = AIMessage(content="", tool_calls=[{"name": "save_memory_fact", "args": {"fact": "x"}, "id": "2"}])
    assert graph.after_agent({"messages": [plain]}) == "__end__"
    assert graph.after_agent({"messages": [safe_call]}) == "tools"
    assert graph.after_agent({"messages": [risky_call]}) == "approval"
    assert graph.after_agent({"messages": [plain], "error": "boom"}) == "handle_error"


def test_recent_messages_starts_on_user_message():
    graph = _import_graph()
    from langchain_core.messages import AIMessage, HumanMessage
    history = [HumanMessage("q1"), AIMessage("a1"), HumanMessage("q2"), AIMessage("a2"), HumanMessage("q3")]
    window = graph.recent_messages(history, 4)
    assert isinstance(window[0], HumanMessage) and window[-1].content == "q3"


def test_chat_graph_runs_and_remembers_per_thread(monkeypatch):
    graph = _import_graph()
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver

    fake = GenericFakeChatModel(messages=iter([AIMessage(content="Hello there"), AIMessage(content="Second answer")]))
    monkeypatch.setattr(graph, "get_llm", lambda *a, **k: fake)
    app = graph.build_graph(InMemorySaver())
    cfg = {"configurable": {"thread_id": "thread-1", "mode": "general", "user_id": "u1", "conversation_id": "c1"}}

    async def scenario():
        first = await app.ainvoke({"messages": [HumanMessage("hi")]}, cfg)
        second = await app.ainvoke({"messages": [HumanMessage("and again?")]}, cfg)
        other = await app.aget_state({"configurable": {"thread_id": "thread-2"}})
        return first, second, other

    first, second, other = asyncio.run(scenario())
    assert first["messages"][-1].content == "Hello there"
    assert len(second["messages"]) == 4  # thread memory kept both turns
    assert not other.values  # another thread is isolated


def test_agent_pauses_for_approval_and_handles_denial(monkeypatch):
    graph = _import_graph()
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    class FakeToolModel(GenericFakeChatModel):
        def bind_tools(self, *args, **kwargs):
            return self

    request = AIMessage(content="", tool_calls=[
        {"name": "save_memory_fact", "args": {"fact": "likes tea"}, "id": "call-1"}])
    fake = FakeToolModel(messages=iter([request, AIMessage(content="Okay, I did not save it.")]))
    monkeypatch.setattr(graph, "get_llm", lambda *a, **k: fake)
    app = graph.build_graph(InMemorySaver())
    cfg = {"configurable": {"thread_id": "t-approval", "mode": "agent", "user_id": "u1"}}

    async def scenario():
        await app.ainvoke({"messages": [HumanMessage("remember I like tea")]}, cfg)
        paused = await app.aget_state(cfg)
        final = await app.ainvoke(Command(resume=False), cfg)
        return paused, final

    paused, final = asyncio.run(scenario())
    assert paused.next == ("approval",)
    assert paused.tasks[0].interrupts[0].value["tools"][0]["name"] == "save_memory_fact"
    assert final["messages"][-1].content == "Okay, I did not save it."
    assert any(getattr(m, "name", "") == "save_memory_fact" and "denied" in m.content for m in final["messages"])
