from __future__ import annotations
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app import db
from app.faq import find_best_faq


app = FastAPI(title="AI Customer Support Bot")


# Simple endpoint to verify the server is running.
@app.get("/health")
def health():
    return {"ok": True}


# -----------------------------------------------------------------------------
# REQUEST MODEL (Pydantic)
# -----------------------------------------------------------------------------
# Pydantic validates incoming JSON for this endpoint.
# FastAPI uses this model to:
#   - parse the request body
#   - validate types (str, optional, default values)
#   - generate OpenAPI schema automatically (/docs)


class SendMessageRequest(BaseModel):

    conversation_id: str | None = None
    session_id: str | None = None
    text: str
    channel: str = "web"


# -----------------------------------------------------------------------------
# POST /api/messages
# -----------------------------------------------------------------------------
# This is the main endpoint:
# - Ensures a conversation exists
# - Saves the user's message
# - Generates and saves a bot reply (stub for now)
# - Returns a response matching your API examples


@app.post("/api/messages")
def send_message(payload: SendMessageRequest):

    # 1) Determine conversation ID
    # If the client did not provide one, generate a new ID.
    # Using uuid4().hex avoids extra dependencies and works everywhere.
    conversation_id = payload.conversation_id or uuid4().hex

    # 2) Create conversation if missing
    if not db.conversation_exists(conversation_id):
        db.create_conversation(
            conversation_id,
            payload.session_id,
            payload.channel,
        )

    # 3) Store the user message in DB
    user_msg = db.insert_message(
        conversation_id=conversation_id,
        sender_type="user",
        content=payload.text,
        metadata=None,
    )

    # 4) bot reply (FAQ first, then fallback stub)

    match = find_best_faq(payload.text, lang="en")
    print("FAQ DEBUG match =", match)

    if match:
        bot_text = match.answer
        bot_meta = {
            "confidence": round(match.score, 2),
            "language": "en",
            "source": "faq",
            "faq_id": match.id,
        }
    else:
        bot_text = "Thanks! I got your message. How can I help you next?"
        bot_meta = {"confidence": 0.2, "language": "en", "source": "stub"}

    bot_msg = db.insert_message(
        conversation_id=conversation_id,
        sender_type="bot",
        content=bot_text,
        metadata=bot_meta,
    )

    # 5) Update conversation updated_at (so list view can sort by latest activity)
    db.set_conversation_updated(conversation_id)

    # 6) Return response payload
    # We keep the response small: the new user message + the bot reply.
    return {
        "conversation_id": conversation_id,
        "user_message": {
            "id": user_msg["id"],
            "sender_type": user_msg["sender_type"],
            "content": user_msg["content"],
            "created_at": user_msg["created_at"],
        },
        "bot_message": {
            "id": bot_msg["id"],
            "sender_type": bot_msg["sender_type"],
            "content": bot_msg["content"],
            "created_at": bot_msg["created_at"],
        },
        "status": "open",
    }


# -----------------------------------------------------------------------------
# GET /api/conversations/{conversation_id}
# -----------------------------------------------------------------------------
# Fetch the conversation details + messages from DB.


@app.get("/api/conversations/{conversation_id}")
def read_conversation(conversation_id: str):
    conversation = db.get_conversation(conversation_id)

    # If conversation doesn't exist, return HTTP 404.
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return conversation
