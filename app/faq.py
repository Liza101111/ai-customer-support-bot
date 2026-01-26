from __future__ import annotations

from dataclasses import dataclass
from app import db

# Common words that do not carry useful meaning for FAQ matching.
# Removing these improves match quality by focusing on keywords like "refund".
STOPWORDS = {
    "a",
    "an",
    "the",
    "to",
    "of",
    "and",
    "or",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "i",
    "you",
    "my",
    "we",
    "it",
    "this",
    "that",
    "please",
    "can",
    "could",
    "help",
}


def tokenize(text: str) -> set[str]:

    # Replace non-alphanumeric characters with spaces
    cleaned = []
    for ch in text.lower():
        cleaned.append(ch if ch.isalnum() else " ")

    # Split text into words
    words = " ".join(cleaned).split

    # Filter out stopwords and short tokens
    return {w for w in words if w not in STOPWORDS and len(w) >= 2}


class FaqMatch:
    """
    Represents a matched FAQ entry with a similarity score.
    This object is returned only when a match is confident enough.
    """

    id: int
    question: str
    answer: str
    tags: str
    language: str
    score: float  # Range: 0.0 .. 1.0


# -----------------------------------------------------------------------------
# FAQ retrieval logic
# -----------------------------------------------------------------------------
# Find the best FAQ match for a user query.
#
# Matching strategy:
#  - Tokenize the user query
#  - Tokenize FAQ question + tags
#  - Score = token overlap / number of query tokens
#  - Return the highest scoring FAQ if it exceeds a confidence threshold
#
# Args:
#     query: User input text
#     lang: Language code (default: "en")
#
# Returns:
#     FaqMatch if a confident match is found, otherwise None.


def find_best_faq(query: str, lang: str = "en") -> FaqMatch | None:

    # Tokenize user input
    q_tokens = tokenize(query)

    # If no meaningful tokens, skip FAQ matching
    if not q_tokens:
        return None

    # Load active FAQ entries from the database
    rows = db.fetch_faq_entries(lang=lang)

    best: FaqMatch | None = None

    for r in rows:

        # Combine question and tags into a single searchable text
        haystack = f"{r['question']} {r.get('tags', '')}"

        # Tokenize FAQ content
        h_tokens = tokenize(haystack)

        # Compute token overlap score
        overlap = len(q_tokens & h_tokens)
        score = overlap / max(len(q_tokens), 1)

        # Keep the highest scoring FAQ
        if best is None or score > best.score:
            best = FaqMatch(
                id=r["id"],
                question=r["question"],
                answer=r["answer"],
                tags=r.get("tags") or "",
                language=r.get("language") or lang,
                score=score,
            )

        # Apply confidence threshold to avoid weak matches
        if best and best.score >= 0.34:
            return best

        # No suitable FAQ found
        return None


# To do in the future:
#  replace tokenize() with embeddings
#  replace overlap score with cosine similarity
#  keep the same API (find_best_faq). no API changes, only smarter internals.
