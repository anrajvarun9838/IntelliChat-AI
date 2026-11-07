-- IntelliChat AI database schema for Supabase (PostgreSQL + pgvector).
-- Idempotent: safe to run more than once. Paste into Supabase > SQL Editor > Run,
-- or let the app run it automatically on first start.
--
-- NOTE: LangGraph's checkpoint tables (checkpoints, checkpoint_blobs,
-- checkpoint_writes, checkpoint_migrations) are created by the app at startup
-- through langgraph-checkpoint-postgres, not by this file.

create extension if not exists vector;

-- ---------------------------------------------------------------------------
-- Users: Supabase Auth owns auth.users. This table holds app-level profile data.
-- ---------------------------------------------------------------------------
create table if not exists public.user_profiles (
    user_id     uuid primary key references auth.users(id) on delete cascade,
    email       text,
    created_at  timestamptz not null default now()
);

create table if not exists public.user_preferences (
    user_id     uuid primary key references auth.users(id) on delete cascade,
    settings    jsonb not null default '{}'::jsonb,
    updated_at  timestamptz not null default now()
);

-- Long-term memory: only facts the user explicitly saved or approved.
create table if not exists public.user_memories (
    id          bigint generated always as identity primary key,
    user_id     uuid not null references auth.users(id) on delete cascade,
    content     text not null check (char_length(content) between 1 and 500),
    created_at  timestamptz not null default now()
);
create index if not exists user_memories_user_idx
    on public.user_memories (user_id, created_at desc);

-- ---------------------------------------------------------------------------
-- Conversations and messages (what the user sees in the chat history)
-- ---------------------------------------------------------------------------
create table if not exists public.conversations (
    id          uuid primary key default gen_random_uuid(),
    user_id     uuid not null references auth.users(id) on delete cascade,
    -- LangGraph thread id. Rotated when a conversation is cleared.
    thread_id   uuid not null default gen_random_uuid(),
    title       text not null default 'New chat',
    mode        text not null default 'general'
                check (mode in ('general', 'document', 'agent', 'coding', 'study')),
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);
create index if not exists conversations_user_updated_idx
    on public.conversations (user_id, updated_at desc);

create table if not exists public.messages (
    id               bigint generated always as identity primary key,
    conversation_id  uuid not null references public.conversations(id) on delete cascade,
    user_id          uuid not null references auth.users(id) on delete cascade,
    role             text not null check (role in ('user', 'assistant')),
    content          text not null,
    mode             text,
    sources          jsonb not null default '[]'::jsonb,
    tools_used       jsonb not null default '[]'::jsonb,
    tokens_in        integer,
    tokens_out       integer,
    latency_ms       integer,
    created_at       timestamptz not null default now()
);
create index if not exists messages_conversation_idx
    on public.messages (conversation_id, id);
create index if not exists messages_user_idx
    on public.messages (user_id);

-- ---------------------------------------------------------------------------
-- Documents and vector chunks (RAG)
-- conversation_id NULL  -> knowledge-base document, available in every chat
-- conversation_id set   -> attached inside one chat; removed with that chat
-- ---------------------------------------------------------------------------
create table if not exists public.documents (
    id               uuid primary key default gen_random_uuid(),
    user_id          uuid not null references auth.users(id) on delete cascade,
    conversation_id  uuid references public.conversations(id) on delete cascade,
    filename         text not null,
    mime_type        text,
    size_bytes       bigint not null default 0,
    checksum         text not null,
    storage_path     text,
    page_count       integer,
    chunk_count      integer not null default 0,
    status           text not null default 'processing'
                     check (status in ('processing', 'ready')),
    created_at       timestamptz not null default now()
);
-- One copy of a file per scope: blocks duplicate uploads, even concurrent ones.
create unique index if not exists documents_unique_checksum_idx
    on public.documents (user_id, checksum,
                         coalesce(conversation_id, '00000000-0000-0000-0000-000000000000'::uuid));
create index if not exists documents_user_idx
    on public.documents (user_id, created_at desc);
create index if not exists documents_conversation_idx
    on public.documents (conversation_id);

create table if not exists public.document_chunks (
    id           bigint generated always as identity primary key,
    document_id  uuid not null references public.documents(id) on delete cascade,
    user_id      uuid not null references auth.users(id) on delete cascade,
    chunk_index  integer not null,
    page         integer,
    content      text not null,
    -- Must match EMBEDDING_DIM in config.py (gemini-embedding-001 at 768 dims).
    embedding    vector(768) not null
);
create index if not exists document_chunks_document_idx
    on public.document_chunks (document_id);
create index if not exists document_chunks_user_idx
    on public.document_chunks (user_id);
create index if not exists document_chunks_embedding_idx
    on public.document_chunks using hnsw (embedding vector_cosine_ops);

-- ---------------------------------------------------------------------------
-- Row Level Security. The app connects with the database owner role (which
-- bypasses RLS) and filters every query by user_id itself; these policies
-- protect the same tables if anyone reaches them through Supabase's public
-- API with the anon key.
-- ---------------------------------------------------------------------------
do $$
declare t text;
begin
    foreach t in array array[
        'user_profiles', 'user_preferences', 'user_memories', 'conversations',
        'messages', 'documents', 'document_chunks'
    ] loop
        execute format('alter table public.%I enable row level security', t);
        execute format('drop policy if exists "own rows" on public.%I', t);
        execute format(
            'create policy "own rows" on public.%I for all to authenticated '
            'using (user_id = auth.uid()) with check (user_id = auth.uid())', t);
    end loop;
end $$;

-- ---------------------------------------------------------------------------
-- Storage: private bucket for uploaded documents, one folder per user.
-- ---------------------------------------------------------------------------
insert into storage.buckets (id, name, public, file_size_limit)
values ('documents', 'documents', false, 20971520)
on conflict (id) do nothing;

drop policy if exists "documents owner access" on storage.objects;
create policy "documents owner access" on storage.objects
    for all to authenticated
    using (bucket_id = 'documents' and (storage.foldername(name))[1] = auth.uid()::text)
    with check (bucket_id = 'documents' and (storage.foldername(name))[1] = auth.uid()::text);
