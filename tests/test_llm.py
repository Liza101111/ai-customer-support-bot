"""
Tests for app/llm.py

All Ollama calls are mocked — no model needs to be running locally.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.llm import generate_reply, stream_reply, LLMError


# ---------------------------------------------------------------------------
# generate_reply
# ---------------------------------------------------------------------------


def _mock_chat_response(content: str):
    return {"message": {"content": content}}


def test_generate_reply_returns_string():
    with patch("app.llm.ollama.chat", return_value=_mock_chat_response("Hello, how can I help?")):
        result = generate_reply("I need help with my order")
    assert result == "Hello, how can I help?"


def test_generate_reply_includes_history():
    history = [
        {"role": "user", "content": "What are your hours?"},
        {"role": "assistant", "content": "We are open 9-5 Mon-Fri."},
    ]
    captured = {}

    def fake_chat(model, messages, **kwargs):
        captured["messages"] = messages
        return _mock_chat_response("Sure!")

    with patch("app.llm.ollama.chat", side_effect=fake_chat):
        generate_reply("Can I call you?", conversation_history=history)

    # system + 2 history + current user = 4
    assert len(captured["messages"]) == 4
    assert captured["messages"][1]["content"] == "What are your hours?"
    assert captured["messages"][-1]["role"] == "user"
    assert captured["messages"][-1]["content"] == "Can I call you?"


def test_generate_reply_empty_message_raises():
    with pytest.raises(LLMError):
        generate_reply("")


def test_generate_reply_whitespace_only_raises():
    with pytest.raises(LLMError):
        generate_reply("   ")


def test_generate_reply_non_string_raises():
    with pytest.raises(LLMError):
        generate_reply(None)  # type: ignore


def test_generate_reply_ollama_error_raises_llm_error():
    import ollama as _ollama

    with patch("app.llm.ollama.chat", side_effect=_ollama.ResponseError("model not found")):
        with pytest.raises(LLMError, match="Ollama error"):
            generate_reply("hello")


def test_generate_reply_unexpected_error_raises_llm_error():
    with patch("app.llm.ollama.chat", side_effect=RuntimeError("network failure")):
        with pytest.raises(LLMError, match="Unexpected error"):
            generate_reply("hello")


def test_generate_reply_uses_env_model(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "mistral")
    captured = {}

    def fake_chat(model, messages, **kwargs):
        captured["model"] = model
        return _mock_chat_response("ok")

    with patch("app.llm.ollama.chat", side_effect=fake_chat):
        generate_reply("test")

    assert captured["model"] == "mistral"


# ---------------------------------------------------------------------------
# stream_reply
# ---------------------------------------------------------------------------


def _mock_stream(tokens: list[str]):
    return iter([{"message": {"content": t}} for t in tokens])


def test_stream_reply_yields_tokens():
    with patch("app.llm.ollama.chat", return_value=_mock_stream(["Hi", " there", "!"])):
        result = list(stream_reply("Hello"))
    assert result == ["Hi", " there", "!"]


def test_stream_reply_skips_empty_tokens():
    with patch("app.llm.ollama.chat", return_value=_mock_stream(["Hello", "", " world"])):
        result = list(stream_reply("Hi"))
    assert result == ["Hello", " world"]


def test_stream_reply_empty_message_raises():
    with pytest.raises(LLMError):
        list(stream_reply(""))


def test_stream_reply_ollama_error_raises_llm_error():
    import ollama as _ollama

    def bad_stream(*a, **kw):
        raise _ollama.ResponseError("stream failed")

    with patch("app.llm.ollama.chat", side_effect=bad_stream):
        with pytest.raises(LLMError, match="Ollama error"):
            list(stream_reply("hello"))
