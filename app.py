"""IntelliChat AI - main chat page (Streamlit entry point: `streamlit run app.py`)."""
import logging

import streamlit as st
from langgraph.types import Command

import chatbot
import database as db
import rag
from auth import init_page, logout
from config import DOC_EXTENSIONS, IMAGE_EXTENSIONS, MODES, settings
from utils import (
    export_pdf, export_txt, file_extension, format_timestamp, friendly_error, group_by_date,
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

EXAMPLES = {
    "general": ["Explain how vaccines train the immune system", "Plan a 3-day study schedule for exams"],
    "coding": ["Write a Python function to merge overlapping intervals", "Explain async/await with an example"],
    "study": ["Teach me how gradient descent works", "Quiz me on the French Revolution"],
    "agent": ["What is 18% of 2,450 plus 12 squared?", "What's the weather in Lucknow and the USD to INR rate?"],
    "document": ["Summarise the main points of my document", "What does the document say about deadlines?"],
}


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
            "model": prefs["model"] if prefs["model"] in settings.model_options else settings.chat_model,
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
        stop_slot.button("Stop generating", icon=":material/stop_circle:", key="stop_btn")  # a click reruns & stops
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
    st.title("💬 IntelliChat AI")
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

# ---------------------------------------------------------------------------
# Header: mode + model info
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
if not messages:
    st.markdown("## How can I help you today?")
    st.caption(f"Current mode: **{MODES[mode]}**. Pick another mode above, or just start typing.")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES[mode]):
        if cols[i % 2].button(example, key=f"ex_{i}", use_container_width=True):
            ss.pending_prompt = example
            st.rerun()

last_assistant_id = max((m["id"] for m in messages if m["role"] == "assistant"), default=None)
for m in messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        meta = [format_timestamp(m["created_at"], tz)]
        if m["role"] == "assistant":
            if m.get("tokens_out"):
                meta.append(f"{(m.get('tokens_in') or 0) + m['tokens_out']:,} tokens")
            if m.get("latency_ms"):
                meta.append(f"{m['latency_ms'] / 1000:.1f} s")
        st.caption(" · ".join(meta))
        if m["role"] != "assistant":
            continue
        if m.get("tools_used"):
            st.caption("🛠 Tools used: " + ", ".join(m["tools_used"]))
        if m.get("sources"):
            with st.expander(f"Sources ({len(m['sources'])})"):
                for s in m["sources"]:
                    page = f", page {s['page']}" if s.get("page") else ""
                    st.markdown(f"**[{s['n']}]** {s['filename']}{page} · similarity {s['similarity']}")
        actions = st.columns([2, 2, 6])
        with actions[0].popover("Copy / download", icon=":material/content_copy:"):
            st.code(m["content"], language="markdown", wrap_lines=True)  # hover the code box to copy
            st.download_button("Download .md", m["content"], file_name=f"response-{m['id']}.md",
                               mime="text/markdown", key=f"dl_{m['id']}")
        if m["id"] == last_assistant_id and not pending_approval:
            if actions[1].button("Regenerate", icon=":material/refresh:", key=f"regen_{m['id']}"):
                ss.regenerate = True
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
            st.rerun()
        if deny_col.button("Deny", key="deny_btn"):
            ss.approval_decision = False
            st.rerun()

# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------
submitted = st.chat_input(
    "Message IntelliChat AI…  (📎 to attach PDF, TXT, DOCX or images)",
    accept_file="multiple", file_type=list(DOC_EXTENSIONS + IMAGE_EXTENSIONS),
    disabled=bool(pending_approval))

text, files = "", []
if submitted:
    text, files = (submitted.text or "").strip(), list(submitted.files or [])
elif ss.get("pending_prompt"):
    text = ss.pop("pending_prompt")

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
            with st.chat_message("user"):
                st.markdown(shown)
            human = chatbot.build_human_message(text, images)
            run_turn(conversation, mode, {"messages": [human]}, make_config(conversation, mode, doc_ids))
        else:
            st.rerun()
except PermissionError:
    st.error("That conversation is not available.")
except Exception as exc:  # noqa: BLE001 - any failure is shown, never silently ignored
    logger.exception("Turn failed")
    st.error(friendly_error(exc))
