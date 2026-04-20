from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Custom exception for database-related errors."""

    pass


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
    try:
        return (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
    except Exception as e:
        logger.error("Failed to generate UTC timestamp: %s", str(e))
        raise DatabaseError(f"Timestamp generation failed: {str(e)}") from e


def get_conn() -> sqlite3.Connection:
    """
    Create and return a SQLite connection.

    - timeout=10 helps avoid 'database is locked' errors in small projects.
    - row_factory=sqlite3.Row makes rows act like dicts:
      row["column_name"] instead of row[0]

    Raises:
        DatabaseError: If connection cannot be established.
    """
    try:
        if not DB_PATH.parent.exists():
            logger.error("Database directory does not exist: %s", DB_PATH.parent)
            raise DatabaseError(f"Database directory not found: {DB_PATH.parent}")

        conn = sqlite3.connect(str(DB_PATH), timeout=10)
        conn.row_factory = sqlite3.Row
        logger.debug("Database connection established")
        return conn
    except sqlite3.Error as e:
        logger.error("Database connection failed: %s", str(e))
        raise DatabaseError(f"Connection failed: {str(e)}") from e
    except Exception as e:
        logger.error("Unexpected error connecting to database: %s", str(e))
        raise DatabaseError(f"Unexpected error: {str(e)}") from e


def conversation_exists(conversation_id: str) -> bool:
    """
    Check if a conversation with the given ID exists.

    Raises:
        DatabaseError: If database query fails.
    """
    if not conversation_id or not isinstance(conversation_id, str):
        logger.warning("Invalid conversation_id for exists check")
        return False

    try:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM conversations WHERE id=?",
                (conversation_id,),
            ).fetchone()
            return row is not None
    except sqlite3.Error as e:
        logger.error("Database error checking conversation existence: %s", str(e))
        raise DatabaseError(f"Query failed: {str(e)}") from e


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

    Raises:
        DatabaseError: If conversation creation fails.
    """
    # Validate inputs
    if not conversation_id or not isinstance(conversation_id, str):
        raise DatabaseError("conversation_id must be a non-empty string")

    if not channel or not isinstance(channel, str):
        raise DatabaseError("channel must be a non-empty string")

    try:
        now = utc_now_iso()

        with get_conn() as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, user_id, session_id, channel, status, created_at, updated_at)
                VALUES(?, NULL, ?, ?, 'open', ?, ?)
                """,
                (conversation_id, session_id, channel, now, now),
            )
            conn.commit()
            logger.debug("Conversation created: %s", conversation_id)
    except sqlite3.IntegrityError as e:
        logger.warning("Conversation already exists or integrity error: %s", str(e))
        raise DatabaseError(f"Integrity error: {str(e)}") from e
    except sqlite3.Error as e:
        logger.error("Database error creating conversation: %s", str(e))
        raise DatabaseError(f"Creation failed: {str(e)}") from e


def set_conversation_updated(conversation_id: str) -> None:
    """
    Update a conversation's updated_at timestamp.
    This is useful when a new message arrives.

    Raises:
        DatabaseError: If update fails.
    """
    if not conversation_id or not isinstance(conversation_id, str):
        raise DatabaseError("conversation_id must be a non-empty string")

    try:
        now = utc_now_iso()

        with get_conn() as conn:
            conn.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?",
                (now, conversation_id),
            )
            conn.commit()
            logger.debug("Conversation updated: %s", conversation_id)
    except sqlite3.Error as e:
        logger.error("Database error updating conversation: %s", str(e))
        raise DatabaseError(f"Update failed: {str(e)}") from e


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

    Raises:
        DatabaseError: If insertion fails.
    """
    # Validate inputs
    if not conversation_id or not isinstance(conversation_id, str):
        raise DatabaseError("conversation_id must be a non-empty string")

    if not sender_type or sender_type not in ("user", "bot", "agent"):
        raise DatabaseError("sender_type must be 'user', 'bot', or 'agent'")

    if not isinstance(content, str) or not content.strip():
        raise DatabaseError("content must be a non-empty string")

    try:
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
            conn.commit()
            message_id = cur.lastrowid
            logger.debug(
                "Message inserted: id=%s conversation=%s", message_id, conversation_id
            )

        return {
            "id": str(message_id),
            "sender_type": sender_type,
            "content": content,
            "metadata": metadata or {},
            "created_at": created_at,
        }
    except sqlite3.Error as e:
        logger.error("Database error inserting message: %s", str(e))
        raise DatabaseError(f"Insertion failed: {str(e)}") from e


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


def fetch_faq_entries(lang: str = "en") -> list[dict[str, Any]]:
    """
    Load active FAQ entries from the DB (for FAQ matching).

    Args:
        lang: Language code (default: "en").

    Returns:
        List of FAQ entry dicts.

    Raises:
        DatabaseError: If query fails.
    """
    if not isinstance(lang, str) or not lang.strip():
        raise DatabaseError("lang must be a non-empty string")

    try:
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT id, question, answer, tags, language, embedding
                FROM faq_entries
                WHERE is_active=1 AND language=?
                """,
                (lang,),
            ).fetchall()

        logger.debug("Fetched %d FAQ entries | lang=%s", len(rows), lang)

        result = []
        for r in rows:
            try:
                result.append(
                    {
                        "id": int(r["id"]),
                        "question": r["question"],
                        "answer": r["answer"],
                        "tags": r["tags"],
                        "language": r["language"],
                        "embedding": r["embedding"],
                    }
                )
            except (ValueError, TypeError, KeyError) as e:
                logger.warning("Skipping malformed FAQ row: %s", str(e))
                continue

        return result
    except sqlite3.Error as e:
        logger.error("Database error fetching FAQ entries: %s", str(e))
        raise DatabaseError(f"Query failed: {str(e)}") from e


def row_to_conversation_preview(r) -> dict[str, Any]:
    """
    Convert a SQLite row into a conversation preview dict.
    """
    return {
        "conversation_id": r["id"],
        "channel": r["channel"],
        "status": r["status"],
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
        "last_message": r["last_message"],
        "last_sender_type": r["last_sender_type"],
        "last_created_at": r["last_created_at"],
    }


def list_conversation(
    limit: int = 20,
    offset: int = 0,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """
    Return conversations newest first with last-message preview.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                c.id,
                c.channel,
                c.status,
                c.created_at,
                c.updated_at,
                lm.content     AS last_message,
                lm.sender_type AS last_sender_type,
                lm.created_at  AS last_created_at
            FROM conversations c
            LEFT JOIN messages lm
                ON lm.id = (
                    SELECT m2.id
                    FROM messages m2
                    WHERE m2.conversation_id = c.id
                    ORDER BY m2.created_at DESC, m2.id DESC
                    LIMIT 1
                )
            WHERE (? IS NULL OR c.status = ?)
            ORDER BY c.updated_at DESC, c.created_at DESC
            LIMIT ? OFFSET ?
            """,
            (status, status, limit, offset),
        ).fetchall()
    return [row_to_conversation_preview(r) for r in rows]


def set_conversation_status(conversation_id: str, status: str) -> None:
    """
    Set conversation status ("open" or "closed") and update updated_at.
    """
    now = utc_now_iso()

    with get_conn() as conn:
        conn.execute(
            """
            UPDATE conversations
            SET status = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, now, conversation_id),
        )


def update_faq_embedding(faq_id: int, embedding_json: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE faq_entries SET embedding=? WHERE id=?",
            (embedding_json, faq_id),
        )


def fetch_recent_messages(conversation_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """
    Return the most recent messages for a conversation, oldest-first,
    suitable for passing as LLM conversation history.

    Args:
        conversation_id: The conversation to fetch messages for.
        limit: Maximum number of recent messages to return (default 10).

    Returns:
        List of dicts with "role" ("user" or "assistant") and "content".
    """
    try:
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT sender_type, content
                FROM (
                    SELECT sender_type, content, created_at, id
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                )
                ORDER BY created_at ASC, id ASC
                """,
                (conversation_id, limit),
            ).fetchall()

        return [
            {
                "role": "assistant" if r["sender_type"] in ("bot", "agent") else "user",
                "content": r["content"],
            }
            for r in rows
        ]
    except sqlite3.Error as e:
        logger.error("Database error fetching messages: %s", str(e))
        raise DatabaseError(f"Query failed: {str(e)}") from e
