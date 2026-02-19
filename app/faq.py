from __future__ import annotations

from dataclasses import dataclass
import json
from math import sqrt

from app import db
from app.embeddings import embed


@dataclass
class FaqMatch:
    """
    Represents a matched FAQ entry with a similarity score.
    This object is returned only when the similarity score
    exceeds the defined confidence threshold.
    """

    id: int
    question: str
    answer: str
    tags: str
    language: str
    score: float  # Range: 0.0 .. 1.0


def dot(a: list[float], b: list[float]) -> float:
    """
    Compute the dot product of two vectors.

    Assumes both vectors are already normalized.
    When embeddings are normalized, dot product == cosine similarity.
    """
    return sum(x * y for x, y in zip(a, b))


# -----------------------------------------------------------------------------
# FAQ retrieval logic
# -----------------------------------------------------------------------------
# Find the best FAQ match for a user query using vector similarity.
#
# Matching strategy:
#  - Generate embedding for the user query
#  - Load stored FAQ embeddings from the database
#  - Compute similarity using dot product
#  - Select the FAQ with the highest similarity score
#  - Return the match only if it exceeds a confidence threshold
#
# Args:
#     query: User input text
#     lang: Language code (default: "en")
#
# Returns:
#     FaqMatch if a confident semantic match is found, otherwise None.


def find_best_faq(query: str, lang: str = "en") -> FaqMatch | None:

    # Clean and validate user input
    q = query.strip()

    # Skip empty queries
    if not q:
        return None

    # Generate embedding for the query
    q_vec = embed(q)

    # Load FAQ entries (with stored embeddings) from database
    rows = db.fetch_faq_entries(lang=lang)
    print(f"FAQ DEBUG rows={len(rows)}")

    emb_count = sum(1 for r in rows if r.get("embedding"))
    print(f"FAQ DEBUG rows_with_embedding={emb_count}")

    best: FaqMatch | None = None

    for r in rows:

        # Skip entries without embeddings
        if not r.get("embedding"):
            continue

        try:
            faq_vec = json.loads(r["embedding"])
        except Exception:

            # Skip corrupted or invalid embeddings
            continue

        # Compute similarity score
        # Embeddings are normalized → dot product equals cosine similarity
        score = dot(q_vec, faq_vec)

        print(f"FAQ id={r['id']} | score={score:.3f} | question={r['question'][:40]}")

        # Keep the highest scoring FAQ
        if best is None or score > best.score:
            best = FaqMatch(
                id=r["id"],
                question=r["question"],
                answer=r["answer"],
                tags=r.get("tags") or "",
                language=r.get("language") or lang,
                score=float(score),
            )

    # Apply confidence threshold to avoid weak or irrelevant matches
    if best and best.score >= 0.35:
        return best

    # No sufficiently confident match found
    return None
