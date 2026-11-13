"""IntelliChat AI - main chat page (Streamlit entry point: `streamlit run app.py`)."""
import html
import logging
import re
import streamlit as st
from langgraph.types import Command

import chatbot
import database as db
import rag
from auth import init_page, logout
from config import DOC_EXTENSIONS, IMAGE_EXTENSIONS, MODES, settings
from utils import (
    export_pdf, export_txt, file_extension, friendly_error, group_by_date,
    prepare_image, shorten,
)

logger = logging.getLogger("intellichat.app")

user = init_page("IntelliChat AI")
uid = user["id"]
prefs = st.session_state["prefs"]
ss = st.session_state
tz = prefs.get("timezone", "UTC")

ss.setdefault("conv_id", None)        # None = a new chat that has not been saved yet
ss.setdefault("new_mode", "general")  # mode used for the next new chat
ss.setdefault("mode_rev", 0)          # bumped when the mode is changed by code, to refresh the selectbox
ss.setdefault("generating", False)    # True while an answer is being generated (input is locked)


def stop_generation() -> None:
    """Called when Stop is clicked: unlock the input and drop the waiting message."""
    ss.generating = False
    ss.pop("queued", None)


CHAT_CSS = """
<style>
/* no avatars / icons */
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessageAvatarCustom"] { display: none !important; }
[data-testid="stChatMessage"] { background: transparent; padding: .3rem 0; gap: 0; }

/* user message: plain text pinned to the far right (no chat_message wrapper) */
.user-row { display: flex; justify-content: flex-end; width: 100%; margin: .5rem 0; }
.user-text { max-width: 75%; text-align: left; line-height: 1.6; word-break: break-word; }

/* assistant message: left side, full width, no bubble */
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
    max-width: 100%; padding: .2rem 0;
}

/* nicer markdown */
[data-testid="stChatMessageContent"] p { line-height: 1.7; margin-bottom: .7rem; }
[data-testid="stChatMessageContent"] h1, [data-testid="stChatMessageContent"] h2,
[data-testid="stChatMessageContent"] h3 { margin: 1rem 0 .5rem 0; letter-spacing: -.01em; }
[data-testid="stChatMessageContent"] ul, [data-testid="stChatMessageContent"] ol { padding-left: 1.4rem; margin-bottom: .7rem; }
[data-testid="stChatMessageContent"] li { margin-bottom: .25rem; }
[data-testid="stChatMessageContent"] pre { border-radius: 12px; }
[data-testid="stChatMessageContent"] code { border-radius: 6px; }
[data-testid="stChatMessageContent"] table { border-collapse: collapse; width: 100%; margin: .6rem 0; }
[data-testid="stChatMessageContent"] th, [data-testid="stChatMessageContent"] td {
    border: 1px solid rgba(128,128,128,.3); padding: .45rem .7rem; }
[data-testid="stChatMessageContent"] th { background: rgba(128,128,128,.12); }
[data-testid="stChatMessageContent"] blockquote {
    border-left: 3px solid #6366f1; margin: .6rem 0; padding: .1rem .9rem; opacity: .9; }
</style>
"""
st.markdown(CHAT_CSS, unsafe_allow_html=True)

def cited_sources(content: str, sources: list[dict]) -> list[dict]:
    """Keep only the sources the answer actually cites, like [1] or [2][3] or [1, 3]."""
    numbers = {int(n) for group in re.findall(r"\[([\d,\s]+)\]", content)
               for n in re.findall(r"\d+", group)}
    return [s for s in sources if s["n"] in numbers]

def render_user(text: str) -> None:
    """Show the user's message as plain text on the right (HTML-escaped, so it is safe)."""
    safe = html.escape(text).replace("\n", "<br>")
    st.markdown(f'<div class="user-row"><div class="user-text">{safe}</div></div>', unsafe_allow_html=True)


WELCOME_HTML = """
<style>
.welcome { display: flex; align-items: center; justify-content: center;
           min-height: 55vh; text-align: center; padding: 0 1rem; }
.welcome-text {
    font-size: clamp(1.9rem, 5vw, 3.2rem); font-weight: 800; letter-spacing: -.02em; line-height: 1.2;
    background: linear-gradient(90deg, #6366f1, #ec4899);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
</style>
<div class="welcome"><div class="welcome-text">How can I help you today?</div></div>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def delete_chat(conversation_id: str) -> None:
    """Delete a chat plus its LangGraph checkpoints and stored files."""
    removed = db.delete_conversation(uid, conversation_id)
    chatbot.delete_threads(removed["thread_ids"])
    rag.storage_remove(removed["storage_paths"])
    if ss.conv_id == conversation_id:
        ss.conv_id = None
    ss.pop("pending_approval", None)


def make_config(conversation: dict, mode: str, doc_ids: list[str]) -> dict:
    """LangGraph run config: thread id for memory plus per-turn settings."""
    return {
        "configurable": {
            "thread_id": conversation["thread_id"],
            "user_id": uid,
            "conversation_id": conversation["id"],
            "mode": mode,
            "model": settings.chat_model,  
            "temperature": prefs["temperature"],
            "max_tokens": prefs["max_tokens"],
            "language": prefs["language"],
            "response_length": prefs["response_length"],
            "system_instructions": prefs["system_instructions"],
            "top_k": prefs["top_k"],
            "doc_ids": doc_ids,
            "memories": [m["content"] for m in db.list_memories(uid)],
            "timezone": tz,
        },
        "recursion_limit": 14,
        "run_name": "intellichat-turn",
        "tags": ["intellichat", mode],
        "metadata": {"mode": mode},
    }


def ingest_attachments(conversation_id: str, files) -> None:
    """Index documents attached in the chat box, with live progress."""
    payload = [(f.name, f.getvalue()) for f in files]
    job = chatbot.start_job(lambda emit: rag.ingest_files(uid, conversation_id, payload, emit))
    bar = st.progress(0.0, text="Processing documents…")
    for event in job.events():
        if event[0] == "progress":
            _, done, total, message = event
            bar.progress(done / max(total, 1), text=f"{done}/{total} · {message}")
    bar.empty()
    if job.error:
        raise job.error
    for result in job.value:
        icon = {"ready": "✅", "duplicate": "ℹ️", "failed": "⚠️"}[result.status]
        st.toast(f"{result.filename}: {result.message}", icon=icon)


def run_turn(conversation: dict, mode: str, graph_input, config: dict) -> None:
    """Stream one assistant answer, save it, then rerun to show the final layout."""
    cid = conversation["id"]
    job = chatbot.start_turn(graph_input, config)
    with st.chat_message("assistant"):
        stop_slot = st.empty()
        stop_slot.button("Stop generating", icon=":material/stop_circle:", key="stop_btn",
                         on_click=stop_generation)  # a click reruns, stops and unlocks the input
        status = st.empty()
        status.caption("Thinking…")
        collected: list[str] = []

        def stream():
            try:
                for kind, value in job.events():
                    if kind == "text":
                        status.empty()
                        collected.append(value)
                        yield value
                    elif kind == "status":
                        status.caption(value)
            finally:
                partial = "".join(collected)
                if not job.finished and partial:  # interrupted by "Stop" or a closed tab
                    db.add_message(uid, cid, "assistant", partial + "\n\n*(generation stopped)*", mode=mode)

        st.write_stream(stream())
        stop_slot.empty()
        status.empty()

    if job.error is not None:
        ss.last_error = friendly_error(job.error)
    else:
        result = job.value
        if result.error:
            ss.last_error = result.error
        elif result.text.strip():
            db.add_message(uid, cid, "assistant", result.text, mode=mode, sources=result.sources,
                           tools_used=result.tools_used, tokens_in=result.tokens_in or None,
                           tokens_out=result.tokens_out or None, latency_ms=result.latency_ms)
        pending = chatbot.get_pending_approval(config)
        ss.pending_approval = {"conversation_id": cid, "payload": pending} if pending else None
        if not result.text.strip() and not result.error and not pending:
            ss.last_error = "The model returned no text. Please try again."
    st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("IntelliChat AI")
    if st.button("New chat", icon=":material/add:", use_container_width=True, type="primary"):
        ss.conv_id = None
        ss.pop("pending_approval", None)
        st.rerun()
    search = st.text_input("Search", placeholder="Search conversations", label_visibility="collapsed")

    conversations = db.list_conversations(uid, search)
    if not conversations:
        st.caption("No conversations yet." if not search else "No matches.")
    for label, items in group_by_date(conversations, tz_name=tz):
        st.caption(label)
        for conv in items:
            row = st.columns([6, 1])
            if row[0].button(shorten(conv["title"], 28), key=f"open_{conv['id']}", use_container_width=True,
                             type="primary" if conv["id"] == ss.conv_id else "secondary"):
                ss.conv_id = conv["id"]
                ss.pop("pending_approval", None)
                st.rerun()
            with row[1].popover("⋯"):
                new_title = st.text_input("Rename", value=conv["title"], key=f"rename_{conv['id']}")
                if st.button("Save name", key=f"save_{conv['id']}"):
                    db.rename_conversation(uid, conv["id"], new_title)
                    st.rerun()
                if st.button("Delete chat", key=f"del_{conv['id']}", icon=":material/delete:"):
                    delete_chat(conv["id"])
                    st.rerun()

    st.divider()
    if ss.conv_id and st.button("Clear this conversation", icon=":material/cleaning_services:",
                                use_container_width=True):
        old_thread = db.clear_conversation(uid, ss.conv_id)
        if old_thread:
            chatbot.delete_threads([old_thread])
        ss.pop("pending_approval", None)
        st.rerun()
    st.page_link("pages/1_Knowledge_Base.py", label="Knowledge Base", icon=":material/library_books:")
    st.page_link("pages/2_Settings.py", label="Settings", icon=":material/settings:")
    st.caption(user["email"])
    if st.button("Log out", icon=":material/logout:", use_container_width=True):
        logout()

# ---------------------------------------------------------------------------
# Load the active conversation
# ---------------------------------------------------------------------------
conversation = db.get_conversation(uid, ss.conv_id) if ss.conv_id else None
if ss.conv_id and conversation is None:  # deleted elsewhere
    ss.conv_id = None
messages = db.list_messages(uid, conversation["id"]) if conversation else []
current_mode = conversation["mode"] if conversation else ss.new_mode
if current_mode not in MODES:
    current_mode = "general"  # chats saved earlier with a mode that no longer exists

# ---------------------------------------------------------------------------
# Header: mode
# ---------------------------------------------------------------------------
head = st.columns([5, 1])
mode_keys = list(MODES)
selected_mode = head[1].selectbox(
    "Mode", mode_keys, index=mode_keys.index(current_mode), format_func=MODES.get,
    key=f"mode_{ss.conv_id or 'new'}_{ss.mode_rev}", label_visibility="collapsed")
if selected_mode != current_mode:
    if conversation:
        db.set_conversation_mode(uid, conversation["id"], selected_mode)
    else:
        ss.new_mode = selected_mode
    current_mode = selected_mode
mode = current_mode

doc_ids: list[str] = []
if mode == "document":
    available = db.list_documents(uid, conversation["id"] if conversation else None)
    if available:
        names = {d["id"]: d["filename"] for d in available}
        doc_ids = st.multiselect("Search in (leave empty for all documents)", list(names), format_func=names.get,
                                 key=f"docs_{ss.conv_id or 'new'}")
    else:
        st.info("No documents yet. Attach PDF, TXT or DOCX files with the button in the message box, "
                "or add them on the Knowledge Base page.")

# ---------------------------------------------------------------------------
# Approval check (when returning to a chat that is waiting for a decision)
# ---------------------------------------------------------------------------
if conversation and ss.get("approval_checked") != conversation["id"]:
    try:
        pending = chatbot.get_pending_approval(make_config(conversation, mode, doc_ids))
    except Exception:  # noqa: BLE001 - not critical for rendering
        logger.warning("Could not check pending approval", exc_info=True)
        pending = None
    ss.pending_approval = {"conversation_id": conversation["id"], "payload": pending} if pending else None
    ss.approval_checked = conversation["id"]
pending_approval = ss.get("pending_approval")
if pending_approval and (not conversation or pending_approval["conversation_id"] != conversation["id"]):
    pending_approval = None

# ---------------------------------------------------------------------------
# Message history
# ---------------------------------------------------------------------------
welcome = st.empty()  # placeholder, so the message can vanish the moment a chat starts
if not messages:
    welcome.markdown(WELCOME_HTML, unsafe_allow_html=True)

last_assistant_id = max((m["id"] for m in messages if m["role"] == "assistant"), default=None)
for m in messages:
    if m["role"] == "user":
        render_user(m["content"])
        continue
    with st.chat_message("assistant"):
        st.markdown(m["content"])
        if m.get("tools_used"):
            st.caption("🛠 " + ", ".join(m["tools_used"]))
        shown_sources = cited_sources(m["content"], m.get("sources") or [])
        if shown_sources:
            with st.expander(f"Sources ({len(shown_sources)})"):
                for s in shown_sources:
                    page = f", page {s['page']}" if s.get("page") else ""
                    st.markdown(f"**[{s['n']}]** {s['filename']}{page}")
        actions = st.columns([1, 1, 14])
        with actions[0].popover(" ", icon=":material/content_copy:", help="Copy"):
            st.code(m["content"], language="markdown", wrap_lines=True)  # click the copy icon on the code box
            st.download_button("Download .md", m["content"], file_name=f"response-{m['id']}.md",
                               mime="text/markdown", key=f"dl_{m['id']}")
        if m["id"] == last_assistant_id and not pending_approval and not ss.generating:
            if actions[1].button(" ", icon=":material/refresh:", help="Regenerate", key=f"regen_{m['id']}"):
                ss.regenerate = True
                ss.generating = True
                st.rerun()

if conversation and messages:
    with st.sidebar.popover("Export chat", icon=":material/download:", use_container_width=True):
        st.download_button("Download as TXT", export_txt(conversation["title"], messages, tz),
                           file_name="conversation.txt", mime="text/plain", key="export_txt")
        if st.button("Prepare PDF", key="prep_pdf"):
            ss.pdf_bytes = export_pdf(conversation["title"], messages, tz)
        if ss.get("pdf_bytes"):
            st.download_button("Download as PDF", ss.pdf_bytes, file_name="conversation.pdf",
                               mime="application/pdf", key="export_pdf")

if ss.get("last_error"):
    st.error(ss.pop("last_error"))

# ---------------------------------------------------------------------------
# Human approval card
# ---------------------------------------------------------------------------
if pending_approval:
    with st.chat_message("assistant"):
        st.warning("**Approval needed.** The assistant wants to perform this action:")
        for tool in pending_approval["payload"].get("tools", []):
            st.code(f"{tool['name']}({tool['args']})", language="python")
        approve_col, deny_col, _ = st.columns([1, 1, 4])
        if approve_col.button("Approve", type="primary", key="approve_btn"):
            ss.approval_decision = True
            ss.generating = True
            st.rerun()
        if deny_col.button("Deny", key="deny_btn"):
            ss.approval_decision = False
            ss.generating = True
            st.rerun()

# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------
generating = ss.generating
submitted = st.chat_input(
    "Ask IntelliChat AI… ",
    accept_file="multiple", file_type=list(DOC_EXTENSIONS + IMAGE_EXTENSIONS),
    disabled=bool(pending_approval) or generating)  # locked while an answer is being generated

if submitted and not generating:
    # remember the message and rerun, so the next run shows the input locked
    ss.queued = {"text": (submitted.text or "").strip(), "files": list(submitted.files or [])}
    ss.generating = True
    st.rerun()

text, files = "", []
if generating:
    queued = ss.pop("queued", None)
    if queued:
        text, files = queued["text"], queued["files"]
elif ss.get("pending_prompt"):
    text = ss.pop("pending_prompt")

if text or files:
    welcome.empty()  # hide the welcome message as soon as a message is sent

try:
    if ss.get("approval_decision") is not None and conversation:
        decision = ss.pop("approval_decision")
        ss.pending_approval = None
        run_turn(conversation, mode, Command(resume=decision), make_config(conversation, mode, doc_ids))

    elif ss.pop("regenerate", False) and conversation and messages:
        last_user = next((m for m in reversed(messages) if m["role"] == "user"), None)
        if last_user:
            cfg = make_config(conversation, mode, doc_ids)
            db.delete_last_assistant_message(uid, conversation["id"])
            human = chatbot.rewind_last_turn(cfg) or chatbot.build_human_message(last_user["content"])
            run_turn(conversation, mode, {"messages": [human]}, cfg)

    elif text or files:
        if conversation is None:
            conversation = db.create_conversation(uid, shorten(text, 50) if text else "New chat", mode)
            ss.conv_id = conversation["id"]
        doc_files = [f for f in files if file_extension(f.name) in DOC_EXTENSIONS]
        image_files = [f for f in files if file_extension(f.name) in IMAGE_EXTENSIONS]

        if doc_files:
            ingest_attachments(conversation["id"], doc_files)
            if mode != "document":  # attached documents are meant to be asked about
                mode = "document"
                db.set_conversation_mode(uid, conversation["id"], mode)
                ss.mode_rev += 1
                st.toast("Switched to Document Chat so you can ask about the attached files.", icon="📄")

        if text or image_files:
            images = [prepare_image(f.getvalue()) for f in image_files]
            shown = text + "".join(f"\n\n📎 *Image: {f.name}*" for f in image_files)
            db.add_message(uid, conversation["id"], "user", shown or text, mode=mode)
            render_user(shown)
            human = chatbot.build_human_message(text, images)
            run_turn(conversation, mode, {"messages": [human]}, make_config(conversation, mode, doc_ids))
        else:
            st.rerun()
except PermissionError:
    st.error("That conversation is not available.")
except Exception as exc:  # noqa: BLE001 - any failure is shown, never silently ignored
    logger.exception("Turn failed")
    st.error(friendly_error(exc))
finally:
    ss.generating = False  # unlock after the answer finishes, fails, or is stopped