from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

# __file__ = the current file path (app/db.py)
# BASE_DIR = folder that contains this file (.../app)
# PROJECT_DIR = project root folder (.../ai-customer-support-bot)
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DB_PATH = PROJECT_DIR / "support_bot.db"


def utc_now_iso() -> str:
    """
    Return current UTC time as ISO 8601 string without microseconds.
    Example: 2026-01-20T10:22:30Z

    Why UTC?
    - Consistent across servers/time zones
    - Easier to sort and compare timestamps
    """
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def get_conn() -> sqlite3.Connection:
    """
    Create and return a SQLite connection.

    - timeout=10 helps avoid 'database is locked' errors in small projects.
    - row_factory=sqlite3.Row makes rows act like dicts:
      row["column_name"] instead of row[0]
    """
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row

    return conn


def conversation_exists(conversation_id: str) -> bool:
    """
    Check if a conversation with the given ID exists.
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM conversations WHERE id=?",
            (conversation_id,),
        ).fetchone()

        return row is not None


def create_conversation(
    conversation_id: str, session_id: str | None, channel: str
) -> None:
    """
    Create a new conversation row.

    Fields:
    - id: conversation_id
    - user_id: NULL for now (anonymous users)
    - session_id: browser session or device identifier
    - channel: "web", "facebook", etc.
    - status: starts as "open"
    - created_at/updated_at: timestamps
    """
    now = utc_now_iso()

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO conversations (id, user_id, session_id, channel, status, created_at, updated_at)
            VALUES(?, NULL, ?, ?, 'open', ?, ?)
            """,
            (conversation_id, session_id, channel, now, now),
        )


def set_conversation_updated(conversation_id: str) -> None:
    """
    Update a conversation's updated_at timestamp.
    This is useful when a new message arrives.
    """
    now = utc_now_iso()

    with get_conn() as conn:
        conn.execute(
            "UPDATE conversations SET updated_at=? WHERE id=?",
            (now, conversation_id),
        )


def insert_message(
    conversation_id: str,
    sender_type: str,
    content: str,
    # This parameter may be a dict or None, and if the caller doesn’t pass it, it defaults to None.
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Insert one message into the messages table and return a Python dict
    matching API structure.

    metadata is stored as JSON text in SQLite (because SQLite has no JSON type).
    """
    created_at = utc_now_iso()
    metadata_str = json.dumps(metadata) if metadata else None

    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO messages (conversation_id, sender_type, content, metadata, created_at)
            VALUES(?, ?, ?, ?, ?)
            """,
            (conversation_id, sender_type, content, metadata_str, created_at),
        )

        message_id = cur.lastrowid

        return {
            "id": str(message_id),
            "sender_type": sender_type,
            "content": content,
            "metadata": metadata or {},
            "created-at": created_at,
        }


def get_conversation(conversation_id: str) -> dict[str, Any] | None:
    """
    Load a conversation and all its messages.

    Returns:
    - None if conversation does not exist
    - Otherwise returns a dict:
      { conversation_id, channel, status, created_at, messages: [...] }
    """

    with get_conn() as conn:
        conversation = conn.execute(
            "SELECT id, channel, status, created_at FROM conversations WHERE id=?",
            (conversation_id,),
        ).fetchone()

        if not conversation:
            return None

        msgs = conn.execute(
            """
            SELECT id, sender_type, content, metadata, created_at
            FROM messages
            WHERE conversation_id=?
            ORDER BY created_at ASC, id ASC
            """,
            (conversation_id,),
        ).fetchall()

    # Convert DB rows into JSON-friendly dicts
    messages: list[dict[str, Any]] = []
    for m in msgs:

        # metadata is stored as a JSON string; convert it back into dict
        meta: dict[str, Any] = {}
        if m["metadata"]:
            try:
                meta = json.loads(m["metadata"])
            except Exception:
                meta = {}

        messages.append(
            {
                "id": str(m["id"]),
                "sender_type": m["sender_type"],
                "content": m["content"],
                "created_at": m["created_at"],
                "metadata": meta,
            }
        )

    return {
        "conversation_id": conversation["id"],
        "channel": conversation["channel"],
        "status": conversation["status"],
        "created_at": conversation["created_at"],
        "messages": messages,
    }
