AI Customer Support Bot (FastAPI + SQLite + Semantic Search)

A backend API for a customer support chatbot built with FastAPI and SQLite, featuring embedding-based semantic FAQ matching.

This project demonstrates:

RESTful API design

Database persistence

Clean architecture separation

Embedding-based semantic search

AI-ready backend structure

✨ Features

Create and manage customer support conversations

Store user and bot messages in SQLite

Continue conversations using conversation_id

Embedding-based FAQ retrieval (semantic search)

Cosine similarity scoring with configurable threshold

Health check endpoint

Swagger / OpenAPI documentation out of the box

🧠 Semantic FAQ Matching

The system uses vector embeddings to match user queries against stored FAQ entries.

How it works:

Generate embedding for user query

Compare against stored FAQ embeddings

Compute similarity via dot product (cosine similarity)

Return best match if confidence threshold is met

This allows matching:

“I want money back”
with
“How do I request a refund?”

Even without exact keyword overlap.

🛠 Tech Stack

Python 3.11+

FastAPI

SQLite

Pydantic

Uvicorn

Sentence Transformers

Vector similarity search (cosine)
