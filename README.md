# AI Customer Support Bot

A backend API for a customer support chatbot built with FastAPI and SQLite. Combines embedding-based semantic FAQ matching with an LLM fallback for questions the FAQ doesn't cover.

## Features

- Create and manage support conversations
- Store user, bot, and agent messages in SQLite
- Continue conversations across requests via `conversation_id`
- Semantic FAQ matching using vector embeddings (cosine similarity)
- LLM-powered fallback replies via Ollama (multi-turn context)
- Admin reply endpoint with token-based auth
- Conversation open/close/reopen lifecycle
- Health check and Swagger/OpenAPI docs out of the box

## How It Works

### 1. FAQ Matching

User messages are embedded with `sentence-transformers/all-MiniLM-L6-v2` and compared against stored FAQ embeddings using dot product (cosine similarity). If the best match exceeds a confidence threshold (0.35), the FAQ answer is returned directly.

```
"I want my money back"  →  matches  →  "How do I request a refund?"
```

### 2. LLM Fallback

When no FAQ entry is confident enough, the message is sent to a local Ollama model (`llama3.2` by default). The last 10 messages from the conversation are passed as context, enabling coherent multi-turn replies.

```
FAQ miss  →  fetch conversation history  →  Ollama  →  reply
```

Override the model with the `LLM_MODEL` environment variable.

## Tech Stack

| Layer | Library |
|---|---|
| API | FastAPI + Uvicorn |
| Validation | Pydantic |
| Database | SQLite (via `sqlite3`) |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| LLM | Ollama (`llama3.2`) |

## Project Structure

```
app/
  main.py          # FastAPI routes
  llm.py           # Ollama LLM integration
  faq.py           # FAQ matching logic
  embeddings.py    # Sentence transformer wrapper
  db.py            # SQLite queries
  admin_auth.py    # Admin token auth
db/
  schema.sql       # Database schema
  init_db.py       # DB initialisation script
  seed_faq.py      # FAQ seed data
tests/
  test_api.py      # API endpoint tests
  test_faq.py      # FAQ matching tests
  test_db.py       # Database layer tests
  test_llm.py      # LLM module tests
```

## Setup

**Requirements:** Python 3.11+, [Ollama](https://ollama.com) running locally.

```bash
# Install dependencies
pip install -r requirements.txt

# Pull the default model
ollama pull llama3.2

# Initialise the database and seed FAQ entries
python db/init_db.py
python db/seed_faq.py

# Start the server
uvicorn app.main:app --reload
```

API docs available at `http://localhost:8000/docs`.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `llama3.2` | Ollama model to use for LLM fallback |
| `ADMIN_TOKEN` | — | Required for the admin-reply endpoint |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/api/messages` | Send a message, get a bot reply |
| `GET` | `/api/conversations` | List conversations |
| `GET` | `/api/conversations/{id}` | Get conversation with messages |
| `POST` | `/api/conversations/{id}/admin-reply` | Add an agent reply (auth required) |
| `POST` | `/api/conversations/{id}/close` | Close a conversation |
| `POST` | `/api/conversations/{id}/reopen` | Reopen a conversation |

## Running Tests

```bash
pytest
```
