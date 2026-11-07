"""Knowledge Base: upload, list and delete documents used by Document Chat."""
import streamlit as st

import chatbot
import database as db
import rag
from auth import init_page
from config import DOC_EXTENSIONS, settings
from utils import format_timestamp, friendly_error, shorten

user = init_page("Knowledge Base · IntelliChat AI")
uid = user["id"]
tz = st.session_state["prefs"].get("timezone", "UTC")

st.title("📚 Knowledge Base")
st.caption("Documents added here are available in every chat. Use **Document Chat** mode to ask questions about them.")


def run_ingest(files) -> list[rag.IngestResult]:
    """Process uploaded files in the background and show live progress."""
    payload = [(f.name, f.getvalue()) for f in files]
    job = chatbot.start_job(lambda emit: rag.ingest_files(uid, None, payload, emit))
    bar = st.progress(0.0, text="Starting…")
    for event in job.events():
        if event[0] == "progress":
            _, done, total, message = event
            bar.progress(done / max(total, 1), text=f"{done}/{total} · {message}")
    bar.empty()
    if job.error:
        raise job.error
    return job.value


with st.form("upload_form", clear_on_submit=True):
    uploads = st.file_uploader("Upload PDF, TXT or DOCX files", type=list(DOC_EXTENSIONS),
                               accept_multiple_files=True,
                               help=f"Up to {settings.max_upload_mb} MB per file. Duplicates are detected automatically.")
    go = st.form_submit_button("Process documents", type="primary")

if go:
    if not uploads:
        st.warning("Choose at least one file first.")
    else:
        try:
            for result in run_ingest(uploads):
                if result.status == "ready":
                    st.success(f"**{result.filename}** — {result.message}")
                elif result.status == "duplicate":
                    st.info(f"**{result.filename}** — {result.message}")
                else:
                    st.error(f"**{result.filename}** — {result.message}")
        except Exception as exc:  # noqa: BLE001
            st.error(friendly_error(exc))

st.divider()
documents = db.list_documents(uid, everything=True)
knowledge = [d for d in documents if d["conversation_id"] is None]
attachments = [d for d in documents if d["conversation_id"] is not None]


def show_documents(items: list[dict], prefix: str) -> None:
    for doc in items:
        cols = st.columns([5, 2, 2, 1])
        cols[0].markdown(f"**{doc['filename']}**")
        details = f"{doc['chunk_count']} chunks" + (f" · {doc['page_count']} pages" if doc.get("page_count") else "")
        if doc.get("conversation_title"):
            details += f" · chat: {shorten(doc['conversation_title'], 24)}"
        cols[1].caption(details)
        cols[2].caption(format_timestamp(doc["created_at"], tz))
        if cols[3].button("", icon=":material/delete:", key=f"{prefix}_{doc['id']}", help="Delete document"):
            # Removes the row, its vectors (cascade) and the stored file.
            rag.storage_remove([db.delete_document(uid, doc["id"])])
            st.rerun()


st.subheader(f"Knowledge base ({len(knowledge)})")
if knowledge:
    show_documents(knowledge, "kb")
else:
    st.info("No documents yet. Upload some above.")

if attachments:
    with st.expander(f"Chat attachments ({len(attachments)})"):
        st.caption("Files attached inside a chat. They are deleted automatically with that chat.")
        show_documents(attachments, "att")
