# IntelliChat AI

**An AI assistant that chats, reads your documents, uses tools and remembers you.**

IntelliChat AI is built with Streamlit, LangChain and LangGraph. The chat model runs on Groq, document search uses Google Gemini embeddings, and Supabase (PostgreSQL, pgvector, Storage and Auth) is the backend.

It has two modes. **General Chat** handles everyday questions, coding help and step-by-step explanations, and calls tools when a task needs them: calculator, date and time, web search, Wikipedia, weather and currency conversion. **Document Chat** lets you upload PDF, TXT and DOCX files and ask questions about them, with answers that cite the file and page.

Each conversation keeps its own memory, and long chats are summarised automatically. The assistant also builds a **long-term memory** of lasting facts about you (name, goals, stable preferences) from your chats and uses it, found by meaning, in every new conversation. Answers stream in as they are written, and you can stop or regenerate them. Accounts, chats, documents and memory are stored per user, so everything is still there after a restart. The same setup runs on a local Windows machine and on Streamlit Community Cloud.

## Features

- **Two chat modes**: General Chat and Document Chat. In Document Chat the same question can also use tools, for example a fact from your document plus today's date.
- **Streaming answers** with Markdown and code blocks, a **Stop** button, **Regenerate**, and copy/download for each answer. The input is locked while an answer is being generated.
- **Chat history**: grouped by date, searchable, rename, delete, clear, export to TXT or PDF.
- **Short-term memory**: LangGraph PostgreSQL checkpoints keep each conversation's context; older messages are summarised automatically.
- **Long-term memory**: after each answer a background model call merges new lasting facts into your saved facts (adding, correcting or removing, so duplicates are avoided). Facts are stored with embeddings and searched by meaning before every answer. You can view, add and delete them in Settings.
- **Documents (RAG)**: PDF, TXT and DOCX from the chat box or the Knowledge Base page; duplicate detection by checksum; parallel processing; vectors in pgvector; originals in Supabase Storage; citations with file and page.
- **Tools**: calculator, date and time, web search, Wikipedia, weather, currency. None of them runs model-written code or changes your data.
- **Accounts**: email and password login through Supabase Auth; every query is filtered by the logged-in user.
- **Settings**: response length, temperature, max tokens, reply language, time zone, custom instructions, retrieval top-k, saved memory, JSON backup, usage and response-time statistics.
- **Optional** LangSmith tracing.

## How the whole project works

### 1. System architecture

```mermaid
flowchart LR
    User(["User (browser)"]) --> UI

    subgraph APP["Streamlit application (Python)"]
        UI["app.py chat page<br/>pages/1_Knowledge_Base.py<br/>pages/2_Settings.py"]
        AUTH["auth.py<br/>login / sign-up"]
        DBM["database.py<br/>queries filtered by user_id"]
        CB["chatbot.py<br/>model clients, Job runner"]
        LOOP["Background thread<br/>one asyncio event loop"]
        GR["graph.py<br/>LangGraph workflow"]
        TL["tools.py"]
        RG["rag.py<br/>ingest and retrieve"]
        MEM["memory.py<br/>long-term memory"]
    end

    UI --> AUTH
    UI --> DBM
    UI --> CB
    UI --> RG
    CB --> LOOP
    LOOP --> GR
    GR --> TL
    GR --> RG
    GR --> MEM
    CB --> MEM

    subgraph MODELS["Model providers"]
        GROQ["Groq<br/>chat model"]
        GEM["Google Gemini<br/>embeddings"]
    end

    subgraph SUPA["Supabase"]
        SAUTH["Auth"]
        PG[("PostgreSQL<br/>chats, messages, documents,<br/>preferences, memories")]
        VEC[("pgvector<br/>document chunks, memory facts")]
        CKP[("LangGraph checkpoints")]
        STO[("Storage<br/>original files")]
    end

    AUTH --> SAUTH
    DBM --> PG
    DBM --> VEC
    GR --> GROQ
    MEM --> GROQ
    RG --> GEM
    MEM --> GEM
    RG --> VEC
    RG --> STO
    GR --> CKP
    TL --> EXT["Public web APIs<br/>search, Wikipedia, weather, currency"]
```

### 2. Life of one message

```mermaid
sequenceDiagram
    actor U as User
    participant UI as app.py (Streamlit)
    participant J as Job (chatbot.py)
    participant G as LangGraph (background loop)
    participant M as Groq model
    participant D as Supabase PostgreSQL

    U->>UI: types a message
    UI->>UI: lock the input, show Stop button
    UI->>D: save the user message
    UI->>J: start_turn(message, config)
    J->>G: run graph on the background loop
    G->>D: load checkpoint (thread id) and relevant saved facts
    G->>M: prompt + history + memory (+ document excerpts, tools)
    M-->>G: streamed tokens or tool requests
    G-->>J: events (text, status, tools used)
    J-->>UI: queue of events
    UI-->>U: answer streams in
    G->>D: save checkpoint
    UI->>D: save assistant message, tokens, response time
    UI->>UI: unlock the input
    UI-)J: start memory update in the background
    J-)M: merge new facts into saved memory
```

### 3. LangGraph workflow

```mermaid
flowchart TD
    S(["START"]) --> P["prepare<br/>reset per-turn fields"]
    P --> RC["recall<br/>saved facts, semantic search"]
    RC --> R{"mode?"}
    R -->|document| RT["retrieve<br/>embed question, search pgvector"]
    R -->|general| A["agent<br/>chat model with tools"]
    RT -->|ok| A
    RT -->|error| H["handle_error"]

    A --> D{"model output"}
    D -->|error or empty| H
    D -->|tool call| T["tools<br/>ToolNode, runs calls in parallel"]
    D -->|final answer| L{"long thread?"}
    T --> A

    L -->|yes| SM["summarize<br/>older messages become a summary"]
    L -->|no| E(["END"])
    SM --> E
    H --> E
```

- **recall** looks up the user's saved long-term facts that are relevant to the new message.
- **retrieve** (Document Chat only) embeds the question and fetches the best document chunks for this user. If nothing is relevant enough, the model is told so and says it could not find the answer.
- **agent** is the only node that talks to the chat model. It sees memory, document excerpts, custom instructions and the running summary, and decides whether to answer or call tools.
- **tools** runs independent tool calls at the same time and returns the results to the agent.
- **summarize** keeps the context small by replacing old messages with a summary.
- **handle_error** is the exit for expected failures (empty or blocked output, failed retrieval).
- State is saved to PostgreSQL after every step, keyed by the conversation's thread id.

### 4. Long-term memory

```mermaid
flowchart LR
    A["Answer shown to the user"] --> B["Background job<br/>(does not delay the UI)"]
    B --> C["Find saved facts related to the message<br/>(embedding search)"]
    C --> D["One model call: related facts + new exchange<br/>returns add / update / delete"]
    D --> E["Skip near-duplicates, embed new facts"]
    E --> F[("user_memories<br/>text + embedding")]
    G["Next message, any chat"] --> H["recall: search facts by meaning"]
    F --> H
    H --> I["Relevant facts added to the prompt"]
```

Facts are short sentences about you. The update step is told not to store temporary requests, secrets, ID or card numbers, or sensitive details, but a model can make mistakes, so check **Settings → Saved memory** now and then and delete anything you do not want kept. Set `AUTO_MEMORY=false` to turn the automatic update off (saved facts are still used).

### 5. Document pipeline

```mermaid
flowchart TD
    UP["Upload PDF / TXT / DOCX<br/>(Knowledge Base page or chat box)"] --> CK["SHA-256 checksum"]
    CK --> DUP{"Already uploaded?"}
    DUP -->|yes| SK["Skip: duplicate"]
    DUP -->|no| REC["Create document record<br/>status: processing"]
    REC --> PAR{"run in parallel"}
    PAR --> ST["Upload original file<br/>to Supabase Storage"]
    PAR --> EX["Extract text<br/>(page numbers for PDFs)"]
    EX --> CH["Split into overlapping chunks"]
    CH --> EM["Embed in batches<br/>Gemini, limited concurrency, retries"]
    ST --> JOIN["Both finished"]
    EM --> JOIN
    JOIN --> INS["Batch-insert chunks and vectors<br/>into pgvector"]
    INS --> OK["status: ready"]
    EX -->|no text, error| FAIL["Roll back record and file<br/>show the error"]
    EM -->|error| FAIL
```

Several files are processed at the same time (limited by `DOC_CONCURRENCY`); one failed file never discards the others. Files attached inside a chat belong to that chat and are deleted with it. Files added on the Knowledge Base page are available in every chat.

### 6. Async design

Streamlit scripts are synchronous and re-run on every click. To avoid repeated `asyncio.run()` calls, the app starts **one background thread with one event loop** (created once with `st.cache_resource`). The LangGraph graph and the async PostgreSQL checkpointer live there. The Streamlit script starts a `Job` and reads streamed events from a thread-safe queue, so tokens appear live while nothing blocks the UI. On Windows the loop is a selector loop because psycopg's async mode does not work with the default Proactor loop.

## Project structure

```
app.py                      main chat page (entry point)
pages/1_Knowledge_Base.py   upload and manage documents
pages/2_Settings.py         preferences, saved memory, backup, usage
config.py                   settings, modes
utils.py                    pure helpers (errors, retries, calculator, exports)
database.py                 PostgreSQL access (pool, user-scoped queries)
chatbot.py                  model clients, background loop, streaming jobs
graph.py                    LangGraph workflow
memory.py                   automatic long-term memory
rag.py                      document ingestion and retrieval
tools.py                    chat tools
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

## Database

| Table | Purpose |
|---|---|
| `conversations` | chats, mode (`general` or `document`), LangGraph thread id |
| `messages` | what you see in the chat (with sources, tools, tokens, response time) |
| `documents` | uploaded files (checksum, storage path, scope) |
| `document_chunks` | text chunks and 768-dimension vectors (HNSW index) |
| `user_memories` | long-term facts with embeddings (HNSW index) |
| `user_preferences`, `user_profiles` | settings and profile |
| `checkpoints*` | LangGraph state, created automatically |

Chat history (what you see) and LangGraph checkpoints (what the model remembers inside a chat) are separate on purpose. The app connects as the database owner and adds `user_id` to every query; Row Level Security policies protect the same tables if someone reaches them through Supabase's public API.

## Setup

### 1. Supabase

1. Create a project at https://supabase.com.
2. Open **SQL Editor**, paste all of `sql/schema.sql` and run it (the app also runs it on first start if the tables are missing).
3. **Project Settings → API**: copy the project URL, the `anon` key and the `service_role` key.
4. **Connect → Session pooler**: copy that connection string (port 5432) as `DATABASE_URL`. The direct connection is IPv6-only on the free plan and often fails.
5. **Authentication → Providers → Email**: for testing you may turn off "Confirm email"; otherwise new users must click the link in the confirmation email first.

### 2. API keys

- **Groq** (chat): create a key in the Groq console and set `GROQ_API_KEY`. `CHAT_MODEL` defaults to `openai/gpt-oss-20b`; use a model ID that is listed for your key.
- **Gemini** (document and memory search): create a key at https://aistudio.google.com/apikey and set `GEMINI_API_KEY`.

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
| `GROQ_API_KEY` | yes | chat model key |
| `GEMINI_API_KEY` | yes | embeddings key |
| `DATABASE_URL` | yes | Supabase Session pooler connection string |
| `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_KEY` | yes | Auth and Storage |
| `CHAT_MODEL` | no | chat model ID, default `openai/gpt-oss-20b` |
| `EMBEDDING_MODEL` | no | default `gemini-embedding-001` (768 dimensions) |
| `CHAT_MODEL_OPTIONS` | no | extra chat model IDs for the Settings selector |
| `AUTO_MEMORY` | no | `true` (default) or `false` |
| `MEMORY_TOP_K`, `MEMORY_MIN_SIMILARITY` | no | how many saved facts are recalled (default 5) and how close they must be (default 0.45) |
| `MAX_MEMORIES` | no | cap on saved facts per user (default 200) |
| `TAVILY_API_KEY` | no | better web search |
| `LANGSMITH_API_KEY` | no | tracing (LangSmith then receives prompts and answers) |
| `MIN_SIMILARITY` | no | document retrieval cut-off, default 0.40 |
| `DB_POOL_SIZE` | no | database connections, default 5 |

Never commit `.env` or `.streamlit/secrets.toml`, and never share them: `.gitignore` already excludes both.

## Deploy to Streamlit Community Cloud

1. Push the project to GitHub (check that `.env` and `.streamlit/secrets.toml` are not in the repository).
2. On https://share.streamlit.io choose **New app**, pick the repository, branch `main`, main file `app.py`.
3. In **Advanced settings → Secrets**, paste the same values as TOML with quotes:
   ```toml
   GROQ_API_KEY = "..."
   CHAT_MODEL = "openai/gpt-oss-20b"
   GEMINI_API_KEY = "..."
   DATABASE_URL = "postgresql://postgres.xxxx:PASSWORD@aws-0-REGION.pooler.supabase.com:5432/postgres"
   SUPABASE_URL = "https://xxxx.supabase.co"
   SUPABASE_ANON_KEY = "..."
   SUPABASE_SERVICE_KEY = "..."
   ```
4. Deploy, then set the Site URL under Supabase **Authentication → URL Configuration**.

Things to know after deploying: every visitor shares your Groq and Gemini quota, and anyone can sign up unless you disable sign-ups in Supabase. A free Supabase project pauses after a week without activity; restore it from the dashboard.

## Troubleshooting

- **A setup screen lists names**: those settings are missing.
- **`PoolTimeout` / cannot reach the database**: use the Session pooler URL, check the password (URL-encode special characters), make sure the project is not paused, and restart the app cleanly instead of relying on hot reload.
- **"Email not confirmed"**: confirm the email, or turn off email confirmation in Supabase.
- **429 or quota errors**: wait a minute. `limit: 0` means the model has no free quota for your project.
- **`model_not_found`**: `CHAT_MODEL` must be a Groq model ID that your key can use, with no spaces or quotes.
- **"No readable text found"**: the PDF is a scan (images only). Use a text PDF or run OCR first.
- **Document Chat finds nothing**: lower `MIN_SIMILARITY` or select the document in the "Search in" box.
- **Memory does not recall a fact**: lower `MEMORY_MIN_SIMILARITY`, or check Settings → Saved memory.
- **Very slow answers**: check the terminal logs (`graph node finished`, `turn finished`) to see which step is slow.

## Limitations

- The chat model is text-only, so images cannot be attached.
- Login is kept only for the life of the browser session; a page reload asks you to log in again. There is no password-reset screen.
- Scanned PDFs are not supported (no OCR). PDF export of chats only renders Latin text; use TXT for other scripts.
- Long-term memory is written by a model and can occasionally store something wrong; review it in Settings.
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
9. Tell the assistant your name, open a new chat and ask "what is my name?". Check it under Settings → Saved memory.
10. A 30-message chat keeps working (summarising); follow-ups still use earlier context.
11. Two accounts cannot see each other's chats, documents or memory.
