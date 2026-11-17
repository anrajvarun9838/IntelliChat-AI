# IntelliChat AI

## Live Demo -> https://intellichat-ai.streamlit.app/

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


### 2. LangGraph workflow

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

### 3. Long-term memory

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


### 4. Async design

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



