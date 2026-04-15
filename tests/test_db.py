"""
Tests for app/db.py

Uses an in-memory SQLite database so tests never touch the real support_bot.db.
Each test gets a fresh database via the `test_db` fixture.
"""

import json
import pytest
import sqlite3
from pathlib import Path
from unittest.mock import patch

# Load schema SQL to recreate tables in test DB
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """
    Create a temporary SQLite database with the real schema.
    Patches app.db.DB_PATH so all db functions use this test DB.
    """
    db_path = tmp_path / "test.db"

    # Create schema in temp DB
    conn = sqlite3.connect(str(db_path))
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()

    # Patch DB_PATH in app.db to point to test DB
    monkeypatch.setattr("app.db.DB_PATH", db_path)

    return db_path


# -----------------------------------------------------------------------------
# conversation_exists
# -----------------------------------------------------------------------------

def test_conversation_exists_returns_false_when_missing(test_db):
    from app import db
    assert db.conversation_exists("nonexistent") is False


def test_conversation_exists_returns_true_after_create(test_db):
    from app import db
    db.create_conversation("conv1", None, "web")
    assert db.conversation_exists("conv1") is True


# -----------------------------------------------------------------------------
# create_conversation
# -----------------------------------------------------------------------------

def test_create_conversation_succeeds(test_db):
    from app import db
    db.create_conversation("conv2", "session-abc", "web")
    assert db.conversation_exists("conv2") is True


def test_create_conversation_invalid_id_raises(test_db):
    from app import db
    from app.db import DatabaseError
    with pytest.raises(DatabaseError):
        db.create_conversation("", None, "web")


def test_create_conversation_invalid_channel_raises(test_db):
    from app import db
    from app.db import DatabaseError
    with pytest.raises(DatabaseError):
        db.create_conversation("conv3", None, "")


# -----------------------------------------------------------------------------
# insert_message
# -----------------------------------------------------------------------------

def test_insert_message_returns_correct_structure(test_db):
    from app import db
    db.create_conversation("conv4", None, "web")
    msg = db.insert_message("conv4", "user", "Hello!")

    assert msg["sender_type"] == "user"
    assert msg["content"] == "Hello!"
    assert "id" in msg
    assert "created_at" in msg


def test_insert_message_with_metadata(test_db):
    from app import db
    db.create_conversation("conv5", None, "web")
    meta = {"confidence": 0.9, "source": "faq"}
    msg = db.insert_message("conv5", "bot", "Here is your answer.", metadata=meta)

    assert msg["metadata"] == meta


def test_insert_message_invalid_sender_raises(test_db):
    from app import db
    from app.db import DatabaseError
    db.create_conversation("conv6", None, "web")
    with pytest.raises(DatabaseError):
        db.insert_message("conv6", "unknown", "Hello!")


def test_insert_message_empty_content_raises(test_db):
    from app import db
    from app.db import DatabaseError
    db.create_conversation("conv7", None, "web")
    with pytest.raises(DatabaseError):
        db.insert_message("conv7", "user", "")


# -----------------------------------------------------------------------------
# get_conversation
# -----------------------------------------------------------------------------

def test_get_conversation_returns_none_when_missing(test_db):
    from app import db
    result = db.get_conversation("does-not-exist")
    assert result is None


def test_get_conversation_returns_correct_structure(test_db):
    from app import db
    db.create_conversation("conv8", None, "web")
    db.insert_message("conv8", "user", "Hi")
    db.insert_message("conv8", "bot", "Hello!")

    result = db.get_conversation("conv8")

    assert result["conversation_id"] == "conv8"
    assert result["channel"] == "web"
    assert result["status"] == "open"
    assert len(result["messages"]) == 2
    assert result["messages"][0]["sender_type"] == "user"
    assert result["messages"][1]["sender_type"] == "bot"


# -----------------------------------------------------------------------------
# set_conversation_status
# -----------------------------------------------------------------------------

def test_set_conversation_status_to_closed(test_db):
    from app import db
    db.create_conversation("conv9", None, "web")
    db.set_conversation_status("conv9", "closed")

    result = db.get_conversation("conv9")
    assert result["status"] == "closed"


def test_set_conversation_status_reopen(test_db):
    from app import db
    db.create_conversation("conv10", None, "web")
    db.set_conversation_status("conv10", "closed")
    db.set_conversation_status("conv10", "open")

    result = db.get_conversation("conv10")
    assert result["status"] == "open"