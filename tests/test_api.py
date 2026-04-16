"""
Tests for app/main.py API endpoints

Uses FastAPI's TestClient to make real HTTP requests against the app.
Uses a temporary SQLite database so tests never touch the real support_bot.db.
Mocks find_best_faq to avoid loading the embedding model.
"""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


@pytest.fixture
def client(tmp_path, monkeypatch):
    """
    Set up a test client with a temporary database.
    Patches DB_PATH so all db calls use the test DB.
    Also sets a dummy ADMIN_TOKEN for protected endpoint tests.
    """
    import sqlite3

    # Create temp DB with real schema
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()

    # Patch DB_PATH before importing app
    monkeypatch.setattr("app.db.DB_PATH", db_path)
    monkeypatch.setenv("ADMIN_TOKEN", "test-token")

    from app.main import app

    return TestClient(app)


# -----------------------------------------------------------------------------
# GET /health
# -----------------------------------------------------------------------------


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


# -----------------------------------------------------------------------------
# POST /api/messages
# -----------------------------------------------------------------------------


def test_send_message_returns_correct_structure(client):
    """Basic message send should return conversation_id, user and bot messages."""
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("app.main.find_best_faq", lambda *a, **kw: None)

        response = client.post("/api/messages", json={"text": "Hello"})

    assert response.status_code == 200
    data = response.json()
    assert "conversation_id" in data
    assert data["user_message"]["content"] == "Hello"
    assert data["user_message"]["sender_type"] == "user"
    assert data["bot_message"]["sender_type"] == "bot"
    assert data["status"] == "open"


def test_send_message_creates_new_conversation(client):
    """Sending a message without conversation_id should create a new one."""
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("app.main.find_best_faq", lambda *a, **kw: None)

        response = client.post("/api/messages", json={"text": "Hi there"})

    assert response.status_code == 200
    assert response.json()["conversation_id"] is not None


def test_send_message_continues_existing_conversation(client):
    """Sending with an existing conversation_id should reuse it."""
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("app.main.find_best_faq", lambda *a, **kw: None)

        r1 = client.post("/api/messages", json={"text": "First message"})
        conv_id = r1.json()["conversation_id"]

        r2 = client.post(
            "/api/messages",
            json={
                "text": "Second message",
                "conversation_id": conv_id,
            },
        )

    assert r2.status_code == 200
    assert r2.json()["conversation_id"] == conv_id


def test_send_message_empty_text_returns_400(client):
    """Empty text should return 400 Bad Request."""
    response = client.post("/api/messages", json={"text": ""})
    assert response.status_code == 400


def test_send_message_missing_text_returns_422(client):
    """Missing text field should return 422 Unprocessable Entity."""
    response = client.post("/api/messages", json={})
    assert response.status_code == 422


# -----------------------------------------------------------------------------
# GET /api/conversations/{conversation_id}
# -----------------------------------------------------------------------------


def test_get_conversation_returns_404_when_missing(client):
    response = client.get("/api/conversations/does-not-exist")
    assert response.status_code == 404


def test_get_conversation_returns_messages(client):
    """After sending messages, get conversation should return them."""
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("app.main.find_best_faq", lambda *a, **kw: None)

        r = client.post("/api/messages", json={"text": "Hello"})
        conv_id = r.json()["conversation_id"]

    response = client.get(f"/api/conversations/{conv_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["conversation_id"] == conv_id
    assert len(data["messages"]) == 2  # user + bot


# -----------------------------------------------------------------------------
# GET /api/conversations
# -----------------------------------------------------------------------------


def test_list_conversations_returns_empty(client):
    response = client.get("/api/conversations")
    assert response.status_code == 200
    data = response.json()
    assert data["items"] == []


def test_list_conversations_returns_items(client):
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("app.main.find_best_faq", lambda *a, **kw: None)
        client.post("/api/messages", json={"text": "Hello"})

    response = client.get("/api/conversations")
    assert response.status_code == 200
    assert len(response.json()["items"]) == 1


# -----------------------------------------------------------------------------
# POST /api/conversations/{id}/close and /reopen
# -----------------------------------------------------------------------------


def test_close_conversation(client):
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("app.main.find_best_faq", lambda *a, **kw: None)
        r = client.post("/api/messages", json={"text": "Hello"})
        conv_id = r.json()["conversation_id"]

    response = client.post(f"/api/conversations/{conv_id}/close")
    assert response.status_code == 200
    assert response.json()["status"] == "closed"


def test_reopen_conversation(client):
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("app.main.find_best_faq", lambda *a, **kw: None)
        r = client.post("/api/messages", json={"text": "Hello"})
        conv_id = r.json()["conversation_id"]

    client.post(f"/api/conversations/{conv_id}/close")
    response = client.post(f"/api/conversations/{conv_id}/reopen")
    assert response.status_code == 200
    assert response.json()["status"] == "open"


def test_close_nonexistent_conversation_returns_404(client):
    response = client.post("/api/conversations/fake-id/close")
    assert response.status_code == 404


# -----------------------------------------------------------------------------
# POST /api/conversations/{id}/admin-reply
# -----------------------------------------------------------------------------


def test_admin_reply_without_token_returns_401(client):
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("app.main.find_best_faq", lambda *a, **kw: None)
        r = client.post("/api/messages", json={"text": "Hello"})
        conv_id = r.json()["conversation_id"]

    response = client.post(
        f"/api/conversations/{conv_id}/admin-reply",
        json={"text": "We are looking into this."},
    )
    assert response.status_code == 401


def test_admin_reply_with_valid_token(client):
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("app.main.find_best_faq", lambda *a, **kw: None)
        r = client.post("/api/messages", json={"text": "Hello"})
        conv_id = r.json()["conversation_id"]

    response = client.post(
        f"/api/conversations/{conv_id}/admin-reply",
        json={"text": "We are looking into this."},
        headers={"X-Admin-Token": "test-token"},
    )
    assert response.status_code == 200
    assert response.json()["admin_message"]["content"] == "We are looking into this."
