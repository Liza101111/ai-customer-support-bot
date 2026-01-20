from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Optional

BASE_DIR = Path(__file__).resolve().parent  # .../app
PROJECT_DIR = BASE_DIR.parent  # project root
DB_PATH = PROJECT_DIR / "support_bot_db"


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def create_conversation(
    conversation_id: str, session_id: str | None, channel: str
) -> None:
    now = utc_now_iso
    with get_conn as conn:
        conn.execute(
            """
            INSERT INTO conversations (id, user_id, session_id, channel, status, created_at, updated_at)
            VALUES(?, NULL, ?, ?, 'open', ?, ?)
            """,
            (conversation_id, session_id, channel, now, now),
        )


def set_conversation_updated(conversation_id: str) -> None:
    now = utc_now_iso()
    with get_conn as conn:
        conn.execute(
            "UPDATE conversations SET updated at=? WHERE id=?", (now, conversation_id)
        )


def conversation_exists(conversation_id: str) -> bool:
    with get_conn as conn:
        row = conn.execute(
            "SELECT 1 FROM conversations WHERE id=?", (conversation_id)
        ).fetchone()
        return row is not None


def insert_message(
    conversation_id: str,
    sender_type: str,
    content: str,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
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
    with get_conn() as conn:
        conversation = conn.execute(
            "SELECT id, channel, status, created_at FROM conversations WHERE id=?",
            (conversation_id),
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
            (conversation_id),
        ).fetchall()

    messages: list[dict[str, Any]] = []
    for m in msgs:
        meta = {}
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
