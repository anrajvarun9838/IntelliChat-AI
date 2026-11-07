"""Settings: response options, saved memory, data management, monitoring, workflow view."""
import streamlit as st

import chatbot
import database as db
import rag
from auth import init_page, logout
from config import LANGUAGES, RESPONSE_LENGTHS, settings
from graph import GRAPH_EXPLANATION
from utils import export_backup_json

user = init_page("Settings · IntelliChat AI")
uid = user["id"]
prefs = st.session_state["prefs"]

st.title("⚙️ Settings")
st.caption(f"Signed in as {user['email']}")

# --------------------------------------------------------------------------
# Preferences
# --------------------------------------------------------------------------
st.subheader("Chat preferences")
with st.form("prefs_form"):
    c1, c2 = st.columns(2)
    if len(settings.model_options) > 1:
        current = prefs["model"] if prefs["model"] in settings.model_options else settings.chat_model
        model = c1.selectbox("Model", settings.model_options, index=settings.model_options.index(current))
    else:
        model = settings.chat_model
        c1.text_input("Model", value=model, disabled=True,
                      help="Change CHAT_MODEL in your .env / Streamlit Secrets, or add more with CHAT_MODEL_OPTIONS.")
    length = c2.selectbox("Response length", RESPONSE_LENGTHS, index=RESPONSE_LENGTHS.index(prefs["response_length"]))
    temperature = c1.slider("Temperature", 0.0, 2.0, float(prefs["temperature"]), 0.1,
                            help="Google recommends leaving Gemini 3 models at 1.0; lower values can cause repetition.")
    max_tokens = c2.slider("Max output tokens", 1024, 16384, int(prefs["max_tokens"]), 1024,
                           help="Includes the model's internal reasoning tokens, so very low values can cut answers short.")
    language = c1.selectbox("Reply language", LANGUAGES,
                            index=LANGUAGES.index(prefs["language"]) if prefs["language"] in LANGUAGES else 0)
    top_k = c2.slider("Document retrieval top-k", 1, 15, int(prefs["top_k"]),
                      help="How many document excerpts are retrieved per question.")
    try:
        from zoneinfo import available_timezones
        zones = sorted(available_timezones())
    except Exception:  # noqa: BLE001
        zones = ["UTC"]
    timezone_name = c1.selectbox("Time zone (for timestamps)", zones,
                                 index=zones.index(prefs["timezone"]) if prefs["timezone"] in zones else 0)
    instructions = st.text_area("Custom system instructions", value=prefs["system_instructions"], max_chars=2000,
                                placeholder="e.g. Explain things like I'm a beginner. Use metric units.")
    if st.form_submit_button("Save preferences", type="primary"):
        prefs.update(model=model, response_length=length, temperature=temperature, max_tokens=max_tokens,
                     language=language, top_k=top_k, timezone=timezone_name, system_instructions=instructions)
        db.save_preferences(uid, prefs)
        st.success("Preferences saved.")

st.caption("Theme: open the ⋮ menu (top right) → Settings → Theme to switch between light and dark.")

# --------------------------------------------------------------------------
# Long-term memory
# --------------------------------------------------------------------------
st.subheader("Saved memory")
st.caption("Only facts you add here (or approve in AI Agent mode) are remembered across chats. Nothing is saved automatically.")
with st.form("memory_form", clear_on_submit=True):
    new_fact = st.text_input("Add a fact or preference", max_chars=300,
                             placeholder="e.g. I'm preparing for GATE 2027")
    if st.form_submit_button("Save to memory") and new_fact.strip():
        db.add_memory(uid, new_fact)
        st.rerun()
memories = db.list_memories(uid)
if not memories:
    st.info("Nothing saved yet.")
for memory in memories:
    cols = st.columns([8, 1])
    cols[0].write(memory["content"])
    if cols[1].button("", icon=":material/delete:", key=f"mem_{memory['id']}", help="Forget this"):
        db.delete_memory(uid, memory["id"])
        st.rerun()

# --------------------------------------------------------------------------
# Data management
# --------------------------------------------------------------------------
st.subheader("Your data")
st.download_button("Download backup (JSON)", data=export_backup_json(db.export_all(uid)),
                   file_name="intellichat-backup.json", mime="application/json",
                   help="Conversations, messages, memory, document list and preferences. Embeddings and files are not included.")
with st.expander("Delete all chat history"):
    st.warning("This permanently deletes every conversation, its messages, graph memory and chat attachments. "
               "Knowledge-base documents are kept.")
    confirm = st.checkbox("I understand this cannot be undone")
    if st.button("Delete all chats", type="primary", disabled=not confirm):
        removed = db.delete_all_conversations(uid)
        chatbot.delete_threads(removed["thread_ids"])
        rag.storage_remove(removed["storage_paths"])
        st.session_state.pop("conv_id", None)
        st.success("All chat history deleted.")

# --------------------------------------------------------------------------
# Monitoring
# --------------------------------------------------------------------------
st.subheader("Usage")
usage = db.usage_summary(uid)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Assistant replies", usage["replies"])
m2.metric("Tokens in", f"{usage['tokens_in']:,}")
m3.metric("Tokens out", f"{usage['tokens_out']:,}")
m4.metric("Avg response time", f"{usage['avg_latency_ms'] / 1000:.1f} s")
if usage["tools"]:
    st.caption("Tool usage: " + ", ".join(f"{t['tool']} ×{t['uses']}" for t in usage["tools"]))
st.caption("Token counts appear when Gemini reports them. Detailed traces are available if you set LANGSMITH_API_KEY "
           "(note: LangSmith then receives your prompts and answers).")

# --------------------------------------------------------------------------
# Workflow view
# --------------------------------------------------------------------------
with st.expander("How the LangGraph workflow works"):
    st.markdown(GRAPH_EXPLANATION)
    st.caption("Graph definition (Mermaid — paste into https://mermaid.live to see the diagram):")
    st.code(chatbot.graph_mermaid(), language="text")

st.divider()
if st.button("Log out"):
    logout()
