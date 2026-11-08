# IntelliChat AI

A ChatGPT-style chatbot built with **Streamlit, LangChain, LangGraph and Google Gemini**, backed by **Supabase** (PostgreSQL, pgvector, Storage and Auth). You can chat normally, ask questions about your own PDF/TXT/DOCX documents (RAG), and let the AI use tools such as a calculator, web search, weather and currency conversion. Conversations, documents and memory are stored per user and survive restarts.

The same architecture runs locally on Windows and on Streamlit Community Cloud.

## Features

- **Two chat modes**
  - **General Chat**: everyday questions, coding help and study help. The model decides from your message whether to answer directly, write code, teach step by step, or call a tool.
  - **Document Chat**: answers from your uploaded documents, with citations (file name and page). The same question can also use tools, for example a document fact plus today's date.
- **Streaming answers** with Markdown and code blocks, a **Stop** button, **Regenerate**, and copy/download for each answer. The input box is locked while an answer is being generated.
- **Chat history** in the sidebar (grouped by date, searchable, rename, delete, clear) and export to TXT/PDF.
- **Memory**: per-conversation memory through LangGraph PostgreSQL checkpoints, automatic summarising of long chats, and long-term facts you choose to save (view and delete them in Settings).
- **Documents**: PDF, TXT and DOCX upload from the chat box or the Knowledge Base page, duplicate detection by checksum, parallel processing, vectors in pgvector, originals in Supabase Storage.
- **Tools**: calculator, date and time, web search, Wikipedia, weather, currency, and save-to-memory (which asks for your approval first).
- **Images**: attach an image and ask about it.
- **Accounts**: email and password login through Supabase Auth; every query is filtered by the logged-in user.
- **Settings**: response length, temperature, max tokens, reply language, time zone, custom instructions, retrieval top-k, saved memory, JSON backup, usage and response-time statistics.
- **Optional** LangSmith tracing (off unless you set a key).

## How the whole project works

### 1.LangGraph workflow
https://mermaid.live/edit#pako:eNpVUttO20AQ_ZXRPrVSQuM4SbFVgaIGlQdAKMlT6wot9iS2uhd3dh0Icf694zUBKu3DzNk5Z64HkdsCRSo2yj7lpSQP60VGmYHVp1-ZWK3ny3Umfn-G4fAC7hmpCWtJ-O2RvlwQOvRQIw19QwY2FarCcXTg3wfK8pAJzRkuM3EM8LKD28LmjUbjW1iuWZTQU4W7XhX1Ixbwt0HnK2sG4FBSXkK93WHuLZ30e6EtGiSpWpizjGTPB40fqCtTwVPlS_DWqreqlutAs3-Y8RFAIkstXLNKKU2h8CEgJ9o8NLN4bUaBbXzd-FNPi3cJ4Ie69nvW-vC3qYxUII17Qs5ywzrKmi34klAWl__rdPVCLhU31c0mlB-aWrN1x-kHQI1xIcRBZYD3wSaqU7G9jJM7fNCoLe0fNjLnUc-7_cm6JruTKijWsnHoYMNFzwOOXf0LNPu3xsMa256FBZf0ES7QVB3YjzJM8tW-Cf97dC2sbjmta7SWVL30K7aqQAKNzvHKHDxibjWChD7qLXkvYmwLV90xXt0tulPsr_M2JLsKzvW7LQZiS1UhUk8NDoRG0rJzxYHjgKdZosZMpNAdXdE8D3OrujVn5sjUWpqf1uoTm2yzLUW6kcqx19SF9Lio5JbkewgabuW7bYwXaTQ6j4KISA_iWaRxNDqbjMZfp0kSx6NkHA_EXqTT5Gw6mUySZBLPklkcxbPjQLyEtNFZNB4n53GSTJPJeTSZzo7_AH45MjU

## Project structure

```
app.py                      main chat page (entry point)
pages/1_Knowledge_Base.py   upload and manage documents
pages/2_Settings.py         preferences, memory, backup, usage
config.py                   settings, model IDs, modes
utils.py                    pure helpers (errors, retries, calculator, exports)
database.py                 PostgreSQL access (pool, user-scoped queries)
chatbot.py                  Gemini clients, background loop, streaming jobs
graph.py                    LangGraph workflow
rag.py                      ingestion and retrieval
tools.py                    agent tools
auth.py                     Supabase Auth and page bootstrap
sql/schema.sql              tables, indexes, pgvector, RLS, storage bucket
tests/test_core.py          automated tests (no API calls)
```

## Tools

| Tool | What it does | Needs a key |
|---|---|---|
| `calculator` | safe arithmetic (no `eval`) | no |
| `current_datetime` | date and time in any time zone | no |
| `web_search` | web results | optional `TAVILY_API_KEY`, otherwise key-less DuckDuckGo |
| `wikipedia_search` | article introduction | no |
| `get_weather` | current weather and 3-day forecast | no |
| `convert_currency` | exchange-rate conversion | no |
| `save_memory_fact` | saves a fact to long-term memory, **only after you approve** | no |

## Database

| Table | Purpose |
|---|---|
| `conversations` | chats, mode, LangGraph thread id |
| `messages` | what you see in the chat (with sources, tools, tokens, response time) |
| `documents` | uploaded files (checksum, storage path, scope) |
| `document_chunks` | text chunks and 768-dimension vectors (HNSW index) |
| `user_preferences`, `user_memories`, `user_profiles` | settings, saved facts, profile |
| `checkpoints*` | LangGraph state, created automatically |

Chat history (what you see) and LangGraph checkpoints (what the model remembers) are separate on purpose. The app connects as the database owner and adds `user_id` to every query; Row Level Security policies protect the same tables if someone reaches them through Supabase's public API.

## Setup

### 1. Supabase

1. Create a project at https://supabase.com.
2. Open **SQL Editor**, paste all of `sql/schema.sql` and run it (the app also runs it on first start if the tables are missing).
3. **Project Settings → API**: copy the project URL, the `anon` key and the `service_role` key.
4. **Connect → Session pooler**: copy that connection string (port 5432) as `DATABASE_URL`. The direct connection is IPv6-only on the free plan and often fails.
5. **Authentication → Providers → Email**: for local testing you may turn off "Confirm email"; otherwise new users must click the link in the confirmation email first.

### 2. Gemini API key

Create a key at https://aistudio.google.com/apikey.

### 3. Run locally (Windows)

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

Open http://localhost:8501, sign up, and start chatting. Stop the app with `Ctrl+C` and restart it after code changes.

### Environment variables

| Name | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | yes | Gemini key |
| `DATABASE_URL` | yes | Supabase Session pooler connection string |
| `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_KEY` | yes | Auth and Storage |
| `CHAT_MODEL`, `EMBEDDING_MODEL` | no | defaults `gemini-3.8-flash`, `gemini-embedding-001` (confirm availability in Google AI Studio) |
| `CHAT_MODEL_OPTIONS` | no | extra model IDs for the Settings selector |
| `TAVILY_API_KEY` | no | better web search |
| `LANGSMITH_API_KEY` | no | tracing (LangSmith then receives prompts and answers) |
| `MIN_SIMILARITY` | no | retrieval cut-off, default 0.40 |
| `DB_POOL_SIZE` | no | database connections, default 5 |

Never commit `.env` or any key to GitHub (`.gitignore` already excludes it).

## Deploy to Streamlit Community Cloud

1. Push the project to GitHub.
2. On https://share.streamlit.io choose **New app**, pick the repository, branch `main`, main file `app.py`.
3. In **Advanced settings → Secrets**, paste the same values as TOML with quotes:
   ```toml
   GEMINI_API_KEY = "..."
   DATABASE_URL = "postgresql://postgres.xxxx:PASSWORD@aws-0-REGION.pooler.supabase.com:5432/postgres"
   SUPABASE_URL = "https://xxxx.supabase.co"
   SUPABASE_ANON_KEY = "..."
   SUPABASE_SERVICE_KEY = "..."
   ```
4. Deploy, then set the Site URL under Supabase **Authentication → URL Configuration**.

## Troubleshooting

- **A setup screen lists names**: those settings are missing.
- **`PoolTimeout` / cannot reach the database**: use the Session pooler URL, check the password (URL-encode special characters), make sure the project is not paused, and restart the app cleanly instead of relying on hot reload (it can leave connections open).
- **"Email not confirmed"**: confirm the email, or turn off email confirmation in Supabase.
- **429 or quota errors**: wait a minute; `limit: 0` means the model has no free quota for your project.
- **"No readable text found"**: the PDF is a scan (images only). Use a text PDF or run OCR first.
- **Document Chat finds nothing**: lower `MIN_SIMILARITY` or select the document in the "Search in" box.
- **Very slow answers**: check the terminal logs (`graph node finished`, `turn finished`) to see which step is slow.

## Limitations

- Login is kept only for the life of the browser session; a page reload asks you to log in again. There is no password-reset screen.
- Scanned PDFs are not supported (no OCR). PDF export of chats only renders Latin text; use TXT for other scripts.
- Image generation is not implemented.
- The embedding size (768) is fixed in `sql/schema.sql` and `config.py`.
- Light/dark theme uses Streamlit's own menu (⋮ → Settings → Theme).

## Manual test checklist

1. Sign up, log in, log out.
2. Send a message, restart the app, reopen the chat.
3. Rename, delete, search and clear a conversation.
4. Press Stop during a long answer; confirm the input is locked while generating; press Regenerate.
5. Upload a PDF, a TXT and a DOCX together; re-upload one (should say duplicate).
6. Document Chat: ask something in the file, then something unrelated (should say it was not found).
7. Ask a document question that also needs the date or the web.
8. General Chat: arithmetic, weather, currency, a web search, and a coding question.
9. "Remember that I like tea" shows an approval card; test Approve and Deny.
10. A 30-message chat keeps working (summarising); follow-ups still use earlier context.
11. Attach an image and ask about it.
12. Two accounts cannot see each other's chats or documents.