"""Small, dependency-light helpers: errors, retries, calculator, files, exports.

This module deliberately imports neither Streamlit nor LangChain so it can be
unit-tested without any external service.
"""
from __future__ import annotations

import ast
import asyncio
import base64
import hashlib
import io
import json
import logging
import math
import operator
import random
import re
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Iterable
from xml.sax.saxutils import escape

logger = logging.getLogger("intellichat")


def setup_logging() -> None:
    """Configure application logging once. API keys and document text are never logged."""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


# ---------------------------------------------------------------------------
# Message content helpers
# ---------------------------------------------------------------------------
def text_of(content: Any) -> str:
    """Return only the text of a LangChain message `content`.

    Groq can return a plain string or a list of content blocks (text,
    thinking, images...). Only text blocks are user-visible.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# Errors and retries
# ---------------------------------------------------------------------------
def friendly_error(exc: BaseException) -> str:
    """Translate provider/database exceptions into a short, actionable message."""
    raw = f"{type(exc).__name__}: {exc}"
    low = raw.lower()
    if "api key not valid" in low or "api_key_invalid" in low or "invalid api key" in low:
        return "The Groq API key is invalid. Check GROQ_API_KEY in your .env file or Streamlit Secrets."
    if "permission_denied" in low or ("403" in low and "groq" in low):
        return "The Groq API rejected the request (permission denied). Check that the key's project has the Groq API enabled."
    if "limit: 0" in low:
        return ("This model has no free-tier quota for your project. Pick another model in "
                "Settings, or enable billing in Google AI Studio.")
    if "429" in low or "resource_exhausted" in low or "quota" in low or "rate limit" in low:
        return ("Groq rate limit or quota reached. Wait a minute and try again; "
                "if it keeps happening your daily free-tier quota may be used up.")
    if ("404" in low and "model" in low) or "is not found for api version" in low or "is not supported for" in low:
        return "The selected Groq model is unavailable. Change CHAT_MODEL (or the model in Settings) to a currently supported ID."
    if "503" in low or "unavailable" in low or "overloaded" in low:
        return "Groq is temporarily overloaded. Please try again in a moment."
    if "timeout" in low or "timed out" in low:
        return "The request timed out. Please try again."
    if "graphrecursionerror" in low or "recursion limit" in low:
        return "The agent used too many tool steps without finishing. Try a simpler or more specific request."
    if "psycopg" in low or "operationalerror" in low or ("connection" in low and "database" in low):
        return "Could not reach the database. Check DATABASE_URL and your Supabase project status."
    if "safety" in low or "blocked" in low:
        return "The model declined to answer (safety filter). Try rephrasing your request."
    return f"Something went wrong: {type(exc).__name__}: {str(exc)[:300]}"


def is_transient(exc: BaseException) -> bool:
    """True for errors that are worth retrying (rate limits, overload, network)."""
    low = f"{type(exc).__name__} {exc}".lower()
    if "limit: 0" in low or "api key not valid" in low or "api_key_invalid" in low:
        return False
    markers = (
        "429", "resource_exhausted", "rate limit", "503", "500", "502", "504",
        "unavailable", "overloaded", "timeout", "timed out", "deadline",
        "connection reset", "connection error", "temporarily",
    )
    return any(m in low for m in markers)


async def retry_async(
    func: Callable[[], Awaitable[Any]],
    *,
    attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 20.0,
    should_retry: Callable[[BaseException], bool] = is_transient,
) -> Any:
    """Call `func` and retry transient failures with exponential backoff + jitter."""
    for attempt in range(1, attempts + 1):
        try:
            return await func()
        except Exception as exc:  # noqa: BLE001 - re-raised below when not retryable
            if attempt == attempts or not should_retry(exc):
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1))) * random.uniform(0.7, 1.3)
            logger.warning("Transient error (attempt %d/%d), retrying in %.1fs: %s",
                           attempt, attempts, delay, type(exc).__name__)
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# Safe calculator (never uses eval/exec)
# ---------------------------------------------------------------------------
_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "log": math.log, "log10": math.log10, "log2": math.log2, "exp": math.exp,
    "abs": abs, "round": round, "floor": math.floor, "ceil": math.ceil,
    "min": min, "max": max, "factorial": math.factorial,
}
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("Only numbers are allowed.")
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise ValueError(f"Unknown name: {node.id}")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > 1000:
                raise ValueError("Exponent too large.")
            if isinstance(left, int) and isinstance(right, int) and right > 0 \
                    and abs(left).bit_length() * right > 100_000:
                raise ValueError("Result too large.")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in _FUNCS and not node.keywords and len(node.args) <= 3:
        args = [_eval_node(a) for a in node.args]
        if node.func.id == "factorial" and (len(args) != 1 or not isinstance(args[0], int) or args[0] > 500):
            raise ValueError("factorial needs an integer between 0 and 500.")
        return _FUNCS[node.func.id](*args)
    raise ValueError("Unsupported expression.")


def safe_eval(expression: str) -> str:
    """Evaluate an arithmetic expression safely and return the result as text."""
    expr = (expression or "").strip().replace("^", "**").replace("×", "*").replace("÷", "/")
    if not expr:
        raise ValueError("Empty expression.")
    if len(expr) > 200:
        raise ValueError("Expression is too long (max 200 characters).")
    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval_node(tree)
    except ZeroDivisionError:
        raise ValueError("Division by zero.") from None
    except SyntaxError:
        raise ValueError("Could not parse the expression.") from None
    except OverflowError:
        raise ValueError("Result too large.") from None
    if isinstance(result, float):
        if math.isnan(result) or math.isinf(result):
            raise ValueError("Result is not a finite number.")
        if result.is_integer() and abs(result) < 1e15:
            return str(int(result))
        return f"{result:.12g}"
    return str(result)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
def safe_filename(name: str) -> str:
    """ASCII-only file name safe to use as a Supabase Storage object key."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name or "").strip("._")
    return cleaned[:100] or "file"


def file_checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_extension(name: str) -> str:
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def prepare_image(data: bytes, max_side: int = 1024) -> tuple[str, str]:
    """Downscale an uploaded image and return (mime_type, base64_string).

    Keeping images small matters because they are stored in the LangGraph checkpoint.
    """
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
    return "image/jpeg", base64.b64encode(buffer.getvalue()).decode("ascii")


def vector_literal(values: Iterable[float]) -> str:
    """Format an embedding as a pgvector text literal, e.g. '[0.1,0.2]'."""
    return "[" + ",".join(f"{float(v):.7f}" for v in values) + "]"


# ---------------------------------------------------------------------------
# Dates and grouping
# ---------------------------------------------------------------------------
def _tz(name: str):
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:  # unknown zone or missing tzdata
        return timezone.utc


def format_timestamp(value: datetime | None, tz_name: str = "UTC") -> str:
    if value is None:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(_tz(tz_name)).strftime("%d %b %Y, %H:%M")


def group_by_date(items: list[dict], key: str = "updated_at", tz_name: str = "UTC",
                  now: datetime | None = None) -> list[tuple[str, list[dict]]]:
    """Group rows (already sorted newest first) into Today / Yesterday / ... buckets."""
    tz = _tz(tz_name)
    today = (now or datetime.now(timezone.utc)).astimezone(tz).date()
    labels = ["Today", "Yesterday", "Previous 7 days", "Previous 30 days", "Older"]
    buckets: dict[str, list[dict]] = {label: [] for label in labels}
    for item in items:
        stamp = item[key]
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        age = (today - stamp.astimezone(tz).date()).days
        if age <= 0:
            buckets["Today"].append(item)
        elif age == 1:
            buckets["Yesterday"].append(item)
        elif age <= 7:
            buckets["Previous 7 days"].append(item)
        elif age <= 30:
            buckets["Previous 30 days"].append(item)
        else:
            buckets["Older"].append(item)
    return [(label, buckets[label]) for label in labels if buckets[label]]


def escape_like(term: str) -> str:
    """Escape LIKE wildcards in user-supplied search text."""
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def shorten(text: str, limit: int = 40) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------
def export_txt(title: str, messages: list[dict], tz_name: str = "UTC") -> bytes:
    lines = [title, "=" * len(title), ""]
    for m in messages:
        who = "You" if m["role"] == "user" else "IntelliChat AI"
        lines.append(f"[{format_timestamp(m.get('created_at'), tz_name)}] {who}:")
        lines.append(m["content"])
        lines.append("")
    return "\n".join(lines).encode("utf-8")


def export_pdf(title: str, messages: list[dict], tz_name: str = "UTC") -> bytes:
    """Simple PDF transcript. Built-in PDF fonts only cover Latin text."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    styles = getSampleStyleSheet()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm, title=title)
    story = [Paragraph(escape(title), styles["Title"]), Spacer(1, 12)]
    for m in messages:
        who = "You" if m["role"] == "user" else "IntelliChat AI"
        stamp = format_timestamp(m.get("created_at"), tz_name)
        story.append(Paragraph(f"<b>{escape(who)}</b> <font size=8 color=grey>{escape(stamp)}</font>",
                               styles["Heading4"]))
        body = escape(m["content"]).replace("\n", "<br/>")
        story.append(Paragraph(body, styles["BodyText"]))
        story.append(Spacer(1, 8))
    doc.build(story)
    return buffer.getvalue()


def export_backup_json(data: dict) -> bytes:
    return json.dumps(data, indent=2, default=str, ensure_ascii=False).encode("utf-8")


def parse_memory_ops(text: str) -> list[dict]:
    """Parse the model's JSON list of memory operations; anything invalid is dropped."""
    raw = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return []
    ops: list[dict] = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        kind, fact = item.get("op"), " ".join(str(item.get("text", "")).split())[:200]
        mem_id = item.get("id")
        has_id = isinstance(mem_id, int) and not isinstance(mem_id, bool)
        if kind == "add" and fact:
            ops.append({"op": "add", "text": fact})
        elif kind == "update" and has_id and fact:
            ops.append({"op": "update", "id": mem_id, "text": fact})
        elif kind == "delete" and has_id:
            ops.append({"op": "delete", "id": mem_id})
    return ops