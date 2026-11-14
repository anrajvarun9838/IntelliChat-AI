# IntelliChat AI

An AI assistant that chats, reads your documents and uses tools, all in one place.

IntelliChat AI is a conversational assistant built with Streamlit, LangChain, LangGraph and Google Gemini, with Supabase (PostgreSQL, pgvector, Storage and Auth) as its backend. It has two modes. General Chat handles everyday questions, coding help and step-by-step explanations, and calls tools when a task needs them, such as a calculator, web search, weather or currency conversion. Document Chat lets you upload PDF, TXT and DOCX files and ask questions about them, with answers that cite the file and page they came from.

Each conversation keeps its own memory, long chats are summarised automatically, and you can save facts to carry across chats. Anything that changes your saved data asks for your approval first. Answers stream in as they are written, and you can stop or regenerate them at any time. Accounts, chats, documents and memory are stored per user, so everything is still there after a restart. The same setup runs on a local Windows machine and on Streamlit Community Cloud.

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


## Complete project workflow

```mermaid
flowchart TD
    subgraph S1["1. Start and login"]
        A(["User opens the app"]) --> B{"Settings present?"}
        B -->|no| B1["Setup screen lists missing keys"]
        B -->|yes| C{"Logged in?"}
        C -->|no| D["Login or sign-up page<br/>Supabase Auth"]
        D -->|success| C
        C -->|yes| E["Start runtime once<br/>create tables if missing, open checkpoint store, compile LangGraph"]
        E --> F["Load preferences"]
    end

    F --> G["Chat page"]

    subgraph S2["2. Pages and sidebar"]
        G --> H["Sidebar<br/>new chat, search, history, rename, delete, clear, export"]
        G --> I["Mode selector<br/>General Chat or Document Chat"]
        G --> J["Knowledge Base page<br/>upload and delete documents"]
        G --> K["Settings page<br/>preferences, saved memory, backup, usage"]
    end

    G --> L["Message box<br/>text, documents, images"]

    subgraph S3["3. Sending a message"]
        L --> M["Lock the input<br/>show Stop button"]
        M --> N{"Attachments?"}
        N -->|documents| DOC["Document ingestion<br/>see section 5"]
        N -->|images| IMG["Resize image"]
        N -->|text only| Q
        DOC --> Q
        IMG --> Q
        Q["Create chat if new<br/>save user message"] --> R["Start background job<br/>load saved memory and settings"]
    end

    R --> PR

    subgraph S4["4. LangGraph workflow"]
        PR["prepare<br/>reset per-turn fields"] --> MD{"Mode?"}
        MD -->|document| RT["retrieve<br/>embed question, search pgvector"]
        MD -->|general| AG
        RT -->|ok| AG["agent<br/>Groq with tools"]
        RT -->|error| ER["handle_error"]
        AG --> OUT{"Model output?"}
        OUT -->|empty or blocked| ER
        OUT -->|tool call| TL["tools<br/>run in parallel"]
        OUT -->|save memory request| AP["approval<br/>graph pauses"]
        OUT -->|final answer| LT{"Long thread?"}
        TL --> AG
        AP -->|approved| TL
        AP -->|denied| AG
        LT -->|yes| SM["summarize<br/>old messages become a summary"]
        LT -->|no| FIN(["Graph finished"])
        SM --> FIN
        ER --> FIN
    end

    AG -.->|streamed tokens| ST["Answer streams to the screen"]
    AP -.-> CARD["Approve or Deny card"]
    CARD -->|decision, graph resumes| AP
    FIN --> SAVE["Save answer, sources, tools used,<br/>tokens and response time"]
    SAVE --> UNL(["Unlock the input"])
    M -.-> STOP["Stop clicked<br/>cancel job, keep partial answer"]
    STOP -.-> UNL

    subgraph S5["5. Document ingestion"]
        D1["Checksum"] --> D2{"Duplicate?"}
        D2 -->|yes| D3["Skip"]
        D2 -->|no| D4["Create record: processing"]
        D4 --> D5["Upload original to Storage"]
        D4 --> D6["Extract text with page numbers"]
        D6 --> D7["Split into chunks"]
        D7 --> D8["Embed in batches"]
        D5 --> D9["Insert chunks and vectors"]
        D8 --> D9
        D9 --> D10(["Status: ready"])
        D6 -->|no text or error| D11["Roll back and show error"]
        D8 -->|error| D11
    end

    DOC -.-> D1
    J --> D1

    subgraph S6["6. Storage and services"]
        PG[("PostgreSQL<br/>chats, messages, documents,<br/>memory, preferences")]
        VEC[("pgvector<br/>document chunks")]
        CKP[("LangGraph checkpoints")]
        STO[("Supabase Storage<br/>original files")]
        GEM["Groq and Google Gemini<br/>chat model and embeddings"]
        WEB["Public web APIs<br/>search, Wikipedia, weather, currency"]
    end

    Q -.-> PG
    SAVE -.-> PG
    D4 -.-> PG
    H -.-> PG
    K -.-> PG
    RT -.-> VEC
    D9 -.-> VEC
    PR -.->|state saved after every step| CKP
    D5 -.-> STO
    RT -.-> GEM
    AG -.-> GEM
    SM -.-> GEM
    D8 -.-> GEM
    TL -.-> WEB
```

**How to read it**
- Solid arrows show the order of steps. Dotted arrows show which service or storage a step uses.
- Documents added on the Knowledge Base page or in the chat box go through the same ingestion pipeline (section 5).
- In Document Chat the question is first matched against your documents (`retrieve`), and then the same `agent` node can still call tools, for example for the date or the web.

## Setup

### 1. Supabase

1. Create a project at https://supabase.com.
2. Open **SQL Editor**, paste all of `sql/schema.sql` and run it (the app also runs it on first start if the tables are missing).
3. **Project Settings → API**: copy the project URL, the `anon` key and the `service_role` key.
4. **Connect → Session pooler**: copy that connection string (port 5432) as `DATABASE_URL`. The direct connection is IPv6-only on the free plan and often fails.
5. **Authentication → Providers → Email**: for local testing you may turn off "Confirm email"; otherwise new users must click the link in the confirmation email first.

### 2. Run locally (Windows)

```bat
cd intellchat-ai
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env
python -m pytest -q
streamlit run app.py
```

Open http://localhost:8501, sign up, and start chatting. Stop the app with `Ctrl+C` and restart it after code changes.

### Environment variables

| Name | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | yes | Gemini key |
| `GROQ_API_KEY` | yes | Groq key |
| `DATABASE_URL` | yes | Supabase Session pooler connection string |
| `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_KEY` | yes | Auth and Storage |
| `CHAT_MODEL`, `EMBEDDING_MODEL` | no | defaults `gemini-3.8-flash`, `gemini-embedding-001` (confirm availability in Google AI Studio) |
| `CHAT_MODEL_OPTIONS` | no | extra model IDs for the Settings selector |
| `TAVILY_API_KEY` | no | better web search |
| `LANGSMITH_API_KEY` | no | tracing (LangSmith then receives prompts and answers) |
| `MIN_SIMILARITY` | no | retrieval cut-off, default 0.40 |
| `DB_POOL_SIZE` | no | database connections, default 5 |

Never commit `.env` or any key to GitHub (`.gitignore` already excludes it).
