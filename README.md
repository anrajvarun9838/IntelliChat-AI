# IntelliChat AI

A ChatGPT-style Streamlit chatbot with document Q&A (RAG), agent tools, conversation memory and a LangGraph workflow.
Gemini is the AI provider; Supabase provides PostgreSQL, pgvector, Storage and Auth. The same setup is used locally and on Streamlit Community Cloud.

## Honest status (please read)

The code was written and checked **without internet access or any API credentials**. What was verified:

- All files compile (syntax check) and a static check found no unused imports or undefined names.
- 25 pure-Python tests pass (calculator safety, error messages, retries, file helpers, date grouping, exports).

What was **not** run, and may need small fixes on your machine:

- Anything needing Streamlit, LangChain, LangGraph, Supabase or Gemini (they could not be installed offline). The tests for the graph, approval flow and RAG helpers are written but were never executed. Run `python -m pytest -q` first; a failure there points at the exact problem.
- Live Gemini calls, Supabase Auth/Storage/pgvector, the optional tools (Wikipedia, weather, currency, web search), image understanding, and Streamlit Cloud deployment.

Use the manual checklist at the end after setup.

## Features

Implemented: ChatGPT-like UI (sidebar history grouped by date, search, rename, delete, clear, new chat), streaming answers, Stop and Regenerate, copy/download per answer, TXT/PDF export, JSON backup, five modes (General, Document, AI Agent, Coding, Study), PDF/TXT/DOCX upload with pgvector search and citations (file and page), duplicate detection by checksum, Supabase Auth login, per-conversation memory (LangGraph thread + PostgreSQL checkpoints), automatic summarisation of long chats, saved long-term memory (view/delete), tools (calculator, date/time, web search, Wikipedia, weather, currency, save-memory with approval), human approval step, image upload and questions, usage and latency tracking, optional LangSmith.

Not implemented / limitations:

- **Image generation** is not implemented (not verified, so no fake control).
- **Theme**: Streamlit's own theme switcher is used (menu ⋮ → Settings → Theme) because apps cannot reliably change it themselves.
- **Login survives only while the browser tab session lives**; a full page reload asks you to log in again. No password reset UI (use Supabase dashboard).
- Scanned PDFs (no text layer) are rejected; OCR is not included. PDF export of chats only renders Latin text (use TXT for other scripts).
- The embedding size (768) is fixed in `sql/schema.sql` and `config.py`; changing it needs a schema change.
- Gemini model IDs: `gemini-3.8-flash` was confirmed in third-party sources as free-tier eligible (Nov 2026); `gemini-embedding-001` is documented by LangChain. I could not read Google's own page, so confirm both in Google AI Studio. Extra models are not hard-coded; add them with `CHAT_MODEL_OPTIONS` after checking their IDs.

## Project structure

```
app.py                  main chat page (entry point)
pages/1_Knowledge_Base.py   upload/manage documents
pages/2_Settings.py     preferences, memory, backup, usage, workflow view
config.py  utils.py     settings/model IDs; pure helpers
database.py             PostgreSQL access (pool, all queries filtered by user_id)
chatbot.py              Gemini clients, background event loop, streaming jobs
graph.py                LangGraph workflow
rag.py  tools.py        ingestion/retrieval; agent tools
auth.py                 Supabase Auth + page bootstrap
sql/schema.sql          tables, indexes, pgvector, RLS, storage bucket
tests/test_core.py
```

## 1. Supabase setup

1. Create a project at https://supabase.com (free plan is fine). Save the database password.
2. **pgvector / schema**: open *SQL Editor*, paste all of `sql/schema.sql`, click *Run*. It enables the `vector` extension, creates the tables, indexes, Row Level Security policies and a private `documents` Storage bucket. (The app also runs it automatically on first start if tables are missing; running it yourself first gives clearer errors.)
3. **Keys**: *Project Settings → API*: copy the Project URL, the `anon` key and the `service_role` key.
4. **Database URL**: click *Connect* and copy the **Session pooler** string (port 5432). Use it as `DATABASE_URL`. The direct connection is IPv6-only on the free plan and often fails on Streamlit Cloud.
5. **Auth**: *Authentication → Providers → Email* is on by default. For quick testing you may turn off "Confirm email"; otherwise new users must click the email link before logging in.
6. **Storage**: the `documents` bucket is created by the schema (private, 20 MB limit). Files are saved as `<user_id>/<doc_id>/<name>`.

Security model: the app connects as the database owner (bypasses RLS) and adds `user_id = <logged-in user>` to every query. RLS policies on all tables (and RLS with no policies on LangGraph's checkpoint tables) additionally block access through Supabase's public API. The service-role key is only used on the server for Storage; never put it in front-end code or GitHub.

## 2. Gemini API key

Create a key at https://aistudio.google.com/apikey. Free-tier limits are per project and change; the app retries temporary errors with backoff and shows clear messages for invalid keys, exhausted quota and unavailable models.

## 3. Run locally (Windows)

```bat
cd intellchat-ai
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
notepad .env
python -m pytest -q
streamlit run app.py
```

Fill every non-optional value in `.env`. Open http://localhost:8501, sign up, and start chatting.

### Environment variables

| Name | Required | Purpose |
|---|---|---|
| GEMINI_API_KEY | yes | Gemini key |
| DATABASE_URL | yes | Supabase Session pooler connection string |
| SUPABASE_URL / SUPABASE_ANON_KEY / SUPABASE_SERVICE_KEY | yes | Auth and Storage |
| CHAT_MODEL, EMBEDDING_MODEL | no | Defaults `gemini-3.8-flash`, `gemini-embedding-001` |
| CHAT_MODEL_OPTIONS | no | Extra model IDs for the Settings selector |
| TAVILY_API_KEY | no | Better web search; otherwise key-less DuckDuckGo via `ddgs` |
| LANGSMITH_API_KEY | no | Enables LangSmith tracing (it receives prompts and answers) |
| MIN_SIMILARITY | no | Retrieval cut-off (default 0.40); raise if unrelated text is cited, lower if nothing is found |

## 4. Deploy to Streamlit Community Cloud

1. Push to GitHub (never commit `.env`; `.gitignore` already excludes it):
   ```bat
   git init
   git add .
   git commit -m "IntelliChat AI"
   git branch -M main
   git remote add origin https://github.com/YOUR_NAME/intellchat-ai.git
   git push -u origin main
   ```
2. At https://share.streamlit.io choose *New app*, pick the repo, branch `main`, main file `app.py`.
3. *Advanced settings → Secrets*: paste the same values as TOML, with quotes:
   ```toml
   GEMINI_API_KEY = "..."
   DATABASE_URL = "postgresql://postgres.xxxx:PASSWORD@aws-0-REGION.pooler.supabase.com:5432/postgres"
   SUPABASE_URL = "https://xxxx.supabase.co"
   SUPABASE_ANON_KEY = "..."
   SUPABASE_SERVICE_KEY = "..."
   ```
4. Deploy. In Supabase *Authentication → URL Configuration*, set the Site URL to your app URL (used in confirmation emails).

### Troubleshooting

- **Setup screen listing names**: those secrets are missing.
- **"Could not reach the database"**: use the Session pooler URL, check the password (URL-encode special characters), and that the project is not paused.
- **Windows event-loop errors**: the app already uses a selector loop for psycopg; make sure you did not set another loop policy.
- **429 / quota**: wait, or check your free-tier quota. `limit: 0` means the model has no free quota for your project.
- **Nothing found in Document Chat**: lower `MIN_SIMILARITY`, or select the document in the "Search in" box.
- **Dependency conflicts**: install into a fresh virtual environment.

## Architecture notes

**Database**: Supabase PostgreSQL for conversations, messages, documents, preferences and memories; pgvector (`vector(768)`, HNSW cosine index) for chunks; Storage for original files; LangGraph checkpoint tables for graph state. App history and checkpoints are separate: history is what you see, checkpoints are what the model remembers. Documents with no conversation form the knowledge base; documents attached in a chat belong to it and are deleted with it. Deleting a conversation also removes its checkpoints and stored files.

**LangGraph**: `StateGraph` with typed state; nodes prepare, retrieve, respond, agent, approval, tools (ToolNode), summarize, handle_error; conditional edges route by mode and by whether the model asked for tools; `interrupt()` pauses for human approval before `save_memory_fact`; PostgreSQL checkpointer keyed by a per-conversation thread id; RetryPolicy with exponential backoff on temporary errors. See the workflow view in Settings.

**Async and parallelism**: Streamlit scripts are synchronous, so one background thread runs one event loop (created once with `st.cache_resource`). The graph and the async checkpointer run there; the UI receives streamed tokens through a queue (`Job` in `chatbot.py`), so no `asyncio.run()` is ever repeated. Documents are processed concurrently (limit `DOC_CONCURRENCY`); blocking parsing/database/Storage work runs in threads; embeddings are batched and capped by `EMBED_CONCURRENCY`; storage upload and embedding of a document run together; one failed file never discards the others. Independent tool calls run in parallel through ToolNode. Database access uses a connection pool and batch inserts for chunks. Clients are cached; no user data is cached globally.

## Manual test checklist (needs real credentials)

1. Sign up, log in, log out.
2. Send a message; confirm streaming, then restart the app and reopen the chat.
3. Rename, delete, search, and clear a conversation.
4. Press Stop during a long answer; press Regenerate.
5. Upload a PDF, a TXT and a DOCX together; check progress, then re-upload one (should say duplicate).
6. Document Chat: ask something in the file (citations with page), then something unrelated (should say not found).
7. Delete a document; confirm it disappears from Storage and the question no longer finds it.
8. AI Agent: "What is 18% of 2450?", "weather in Lucknow", "USD to INR", a web search, and two tools in one request.
9. Agent: "Remember that I like tea" → approval card → Approve and Deny both behave.
10. A 30+ message chat keeps working (summarisation); follow-ups still use earlier context.
11. Attach an image and ask about it.
12. Two different accounts cannot see each other's chats or documents.
13. Wrong API key shows a clear message.
