from __future__ import annotations

import logging
import os
from typing import Generator

import ollama

logger = logging.getLogger(__name__)

# Default model — override via LLM_MODEL env var
DEFAULT_MODEL = "llama3.2"

SYSTEM_PROMPT = (
    "You are a helpful customer support assistant. "
    "Answer the customer's question clearly and concisely. "
    "If you do not know the answer, say so politely and offer to escalate to a human agent."
)


class LLMError(Exception):
    """Raised when LLM generation fails."""

    pass


def _model() -> str:
    return os.getenv("LLM_MODEL", DEFAULT_MODEL)


def generate_reply(user_message: str, conversation_history: list[dict] | None = None) -> str:
    """
    Generate a reply using the configured Ollama model.

    Args:
        user_message: The latest message from the user.
        conversation_history: Optional list of prior messages in
            {"role": "user"|"assistant", "content": "..."} format.
            Enables multi-turn context.

    Returns:
        The model's reply as a plain string.

    Raises:
        LLMError: If generation fails for any reason.
    """
    if not user_message or not isinstance(user_message, str):
        raise LLMError("user_message must be a non-empty string")

    user_message = user_message.strip()
    if not user_message:
        raise LLMError("user_message cannot be whitespace-only")

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": user_message})

    model = _model()
    logger.debug("Calling LLM | model=%s history_len=%d", model, len(messages) - 2)

    try:
        response = ollama.chat(model=model, messages=messages)
        reply = response["message"]["content"]
        logger.info("LLM reply generated | model=%s length=%d", model, len(reply))
        return reply
    except ollama.ResponseError as e:
        logger.error("Ollama response error: %s", str(e))
        raise LLMError(f"Ollama error: {str(e)}") from e
    except Exception as e:
        logger.error("Unexpected LLM error: %s", str(e))
        raise LLMError(f"Unexpected error: {str(e)}") from e


def stream_reply(
    user_message: str,
    conversation_history: list[dict] | None = None,
) -> Generator[str, None, None]:
    """
    Stream a reply token-by-token from Ollama.

    Yields each text chunk as it arrives. Useful for SSE / streaming endpoints.

    Raises:
        LLMError: If streaming fails.
    """
    if not user_message or not isinstance(user_message, str):
        raise LLMError("user_message must be a non-empty string")

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": user_message.strip()})

    model = _model()
    logger.debug("Streaming LLM | model=%s", model)

    try:
        for chunk in ollama.chat(model=model, messages=messages, stream=True):
            token = chunk["message"]["content"]
            if token:
                yield token
    except ollama.ResponseError as e:
        logger.error("Ollama stream error: %s", str(e))
        raise LLMError(f"Ollama error: {str(e)}") from e
    except Exception as e:
        logger.error("Unexpected LLM stream error: %s", str(e))
        raise LLMError(f"Unexpected error: {str(e)}") from e
