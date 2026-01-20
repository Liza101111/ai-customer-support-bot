PRAGMA foreign_keys = ON;

-- ===============================
-- 1) Users (admins & agents)
-- ===============================
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    email           TEXT NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('admin', 'agent')),
    created_at      TEXT NOT NULL
);

-- ===============================
-- 2) Conversations
-- ===============================
-- conversation_id in JSON will map to this `id` column.
-- We can use ULID/UUID generated in Python and store it as TEXT.
CREATE TABLE IF NOT EXISTS conversations (
    id              TEXT PRIMARY KEY,  -- e.g. "abc123", "01HRX2..."
    user_id         INTEGER REFERENCES users(id),
    session_id      TEXT,              -- anonymous browser session
    channel         TEXT NOT NULL,     -- "web", "facebook", "whatsapp", ...
    status          TEXT NOT NULL DEFAULT 'open'
                        CHECK (status IN ('open', 'closed', 'needs_human')),
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

-- Index to quickly list/filter conversations
CREATE INDEX IF NOT EXISTS idx_conversations_status
    ON conversations(status);

CREATE INDEX IF NOT EXISTS idx_conversations_channel
    ON conversations(channel);

CREATE INDEX IF NOT EXISTS idx_conversations_updated_at
    ON conversations(updated_at);

-- ===============================
-- 3) Messages
-- ===============================
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL
                        REFERENCES conversations(id)
                        ON DELETE CASCADE,
    sender_type     TEXT NOT NULL
                        CHECK (sender_type IN ('user', 'bot', 'agent')),
    content         TEXT NOT NULL,
    metadata        TEXT,          -- JSON as string (language, confidence, etc.)
    created_at      TEXT NOT NULL
);

-- Index to quickly load messages in a conversation
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id_created_at
    ON messages(conversation_id, created_at);

-- ===============================
-- 4) FAQ / Knowledge Base
-- ===============================
CREATE TABLE IF NOT EXISTS faq_entries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    tags        TEXT,              -- e.g. "payment,refund" or JSON string
    language    TEXT NOT NULL DEFAULT 'en',
    is_active   INTEGER NOT NULL DEFAULT 1   -- 1 = true, 0 = false
);

CREATE INDEX IF NOT EXISTS idx_faq_language
    ON faq_entries(language);

CREATE INDEX IF NOT EXISTS idx_faq_active
    ON faq_entries(is_active);

-- Prevent duplicates for the same FAQ in the same language
CREATE UNIQUE INDEX IF NOT EXISTS uq_faq_question_language
    ON faq_entries(question, language);
