from __future__ import annotations
from uuid import uuid4
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from dotenv import load_dotenv

from app import db
from app.faq import find_best_faq, FAQError
from app.admin_auth import require_admin
from app.db import DatabaseError

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Suppress noisy external library logs
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


app = FastAPI(title="AI Customer Support Bot")

project_dir = Path(__file__).resolve().parent.parent
load_dotenv(project_dir / ".env")


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


class AdminReplyRequest(BaseModel):
    text: str


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
    """
    Send a message and get an AI-powered bot reply.

    The bot uses vector similarity search on the FAQ database to find
    the most relevant answer. If no confident match is found, returns
    a generic fallback response.
    """
    try:
        # Validate payload
        if not payload.text or not isinstance(payload.text, str):
            raise HTTPException(
                status_code=400, detail="text field is required and must be a string"
            )

        if not payload.channel or not isinstance(payload.channel, str):
            raise HTTPException(
                status_code=400, detail="channel field is required and must be a string"
            )

        if payload.text.strip() == "":
            raise HTTPException(
                status_code=400, detail="text cannot be empty or whitespace-only"
            )

        # 1) Determine conversation ID
        conversation_id = payload.conversation_id or uuid4().hex

        # 2) Create conversation if missing
        try:
            if not db.conversation_exists(conversation_id):
                db.create_conversation(
                    conversation_id,
                    payload.session_id,
                    payload.channel,
                )
        except DatabaseError as e:
            logger.error("Failed to create conversation: %s", str(e))
            raise HTTPException(status_code=500, detail="Failed to create conversation")

        # 3) Store the user message in DB
        try:
            user_msg = db.insert_message(
                conversation_id=conversation_id,
                sender_type="user",
                content=payload.text,
                metadata=None,
            )
        except DatabaseError as e:
            logger.error("Failed to save user message: %s", str(e))
            raise HTTPException(status_code=500, detail="Failed to save user message")

        # 4) Find best FAQ match
        try:
            faq_match = find_best_faq(payload.text, lang="en")
        except FAQError as e:
            logger.warning("FAQ search failed, using fallback: %s", str(e))
            faq_match = None
        except Exception as e:
            logger.error("Unexpected error during FAQ search: %s", str(e))
            faq_match = None

        # Determine bot response
        if faq_match:
            bot_text = faq_match.answer
            bot_meta = {
                "confidence": round(faq_match.score, 2),
                "language": "en",
                "source": "faq",
                "faq_id": faq_match.id,
            }
            logger.debug(
                "FAQ match found: id=%s score=%.2f", faq_match.id, faq_match.score
            )
        else:
            bot_text = "Thanks! I got your message. How can I help you next?"
            bot_meta = {"confidence": 0.2, "language": "en", "source": "stub"}
            logger.debug("No FAQ match found, using fallback response")

        # 5) Save bot message
        try:
            bot_msg = db.insert_message(
                conversation_id=conversation_id,
                sender_type="bot",
                content=bot_text,
                metadata=bot_meta,
            )
        except DatabaseError as e:
            logger.error("Failed to save bot message: %s", str(e))
            raise HTTPException(status_code=500, detail="Failed to save bot message")

        # 6) Update conversation timestamp
        try:
            db.set_conversation_updated(conversation_id)
        except DatabaseError as e:
            logger.warning("Failed to update conversation timestamp: %s", str(e))
            # Don't fail the request for this non-critical operation

        # 7) Return response
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

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error("Unexpected error in send_message: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred")


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


# -----------------------------------------------------------------------------
# GET /api/conversations
# -----------------------------------------------------------------------------
# List conversations with pagination and optional status filtering.
#
# Behavior:
# - Returns conversations ordered by most recent activity (updated_at DESC)
# - Supports pagination using limit + offset
# - Can filter by conversation status (e.g. "open", "closed")
# - Includes a preview of the last message for each conversation


@app.get("/api/conversations")
def list_conversations(
    limit: int = 20,
    offset: int = 0,
    status: str | None = None,
):

    # Safety clamps to prevent abuse and bad input
    limit = max(1, min(limit, 100))
    offset = max(0, offset)

    items = db.list_conversation(
        limit=limit,
        offset=offset,
        status=status,
    )

    return {
        "items": items,
        "limit": limit,
        "offset": offset,
    }


# -----------------------------------------------------------------------------
# POST /api/conversations/{conversation_id}/admin-reply
# -----------------------------------------------------------------------------
# Add an admin/support reply message to an existing conversation.
#
# Behavior:
# - Protected endpoint: requires valid X-Admin-Token header
# - Returns 404 if conversation does not exist
# - Stores an "admin" message in the messages table
# - Updates conversations.updated_at so list view sorts correctly


@app.post("/api/conversations/{conversation_id}/admin-reply")
def admin_reply(
    conversation_id: str, payload: AdminReplyRequest, _=Depends(require_admin)
):

    # 1) Validate conversation exists
    if not db.conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")

    # 2) Insert admin message
    admin_msg = db.insert_message(
        conversation_id=conversation_id,
        sender_type="agent",
        content=payload.text,
        metadata={"source": "admin"},
    )

    # 3) Update conversation timestamp
    db.set_conversation_updated(conversation_id)

    return {
        "conversation_id": conversation_id,
        "admin_message": {
            "id": admin_msg["id"],
            "sender_type": admin_msg["sender_type"],
            "content": admin_msg["content"],
            "created_at": admin_msg["created_at"],
        },
        "status": "open",
    }


# -----------------------------------------------------------------------------
# POST /api/conversations/{conversation_id}/close
# -----------------------------------------------------------------------------
# Close a conversation (mark as resolved).


@app.post("/api/conversations/{conversation_id}/close")
def close_conversation(conversation_id: str):

    if not db.conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")

    db.set_conversation_status(conversation_id, "closed")

    return {
        "conversation_id": conversation_id,
        "status": "closed",
    }


# -----------------------------------------------------------------------------
# POST /api/conversations/{conversation_id}/reopen
# -----------------------------------------------------------------------------
# Reopen a closed conversation.


@app.post("/api/conversations/{conversation_id}/reopen")
def reopen_conversation(conversation_id: str):
    if not db.conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")

    db.set_conversation_status(conversation_id, "open")

    return {
        "conversation_id": conversation_id,
        "status": "open",
    }
