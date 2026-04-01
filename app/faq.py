from __future__ import annotations

from dataclasses import dataclass
import json
import logging

from app import db
from app.embeddings import embed, EmbeddingError

logger = logging.getLogger(__name__)


class FAQError(Exception):
    """Custom exception for FAQ-related errors."""

    pass


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

    Raises:
        FAQError: If vectors are empty or have mismatched lengths.
    """
    if not a or not b:
        raise FAQError("Vectors cannot be empty")

    if len(a) != len(b):
        raise FAQError(f"Vector length mismatch: {len(a)} vs {len(b)}")

    return sum(x * y for x, y in zip(a, b))


# FAQ retrieval logic
# Find the best FAQ match for a user query using vector similarity.
#
# Matching strategy:
#  - Generate embedding for the user query
#  - Load stored FAQ embeddings from the database
#  - Compute similarity using dot product
#  - Select the FAQ with the highest similarity score
#  - Return the match only if it exceeds a confidence threshold


def find_best_faq(query: str, lang: str = "en") -> FaqMatch | None:
    """
    Find the best matching FAQ entry for a user query.

    Args:
        query: User input text.
        lang: Language code (default: "en").

    Returns:
        FaqMatch if a confident semantic match is found, otherwise None.

    Raises:
        FAQError: If query processing or FAQ lookup fails critically.
    """
    try:
        # Clean and validate user input
        if not query or not isinstance(query, str):
            logger.warning("Invalid query: not a string")
            return None

        q = query.strip()

        # Skip empty queries
        if not q:
            logger.debug("Empty query after strip")
            return None

        # Generate embedding for the query
        try:
            q_vec = embed(q)
        except EmbeddingError as e:
            logger.error("Failed to embed query: %s", str(e))
            raise FAQError(f"Embedding failed: {str(e)}") from e

        # Load FAQ entries (with stored embeddings) from database
        try:
            rows = db.fetch_faq_entries(lang=lang)
        except Exception as e:
            logger.error("Failed to fetch FAQ entries: %s", str(e))
            raise FAQError(f"Database error: {str(e)}") from e

        emb_count = sum(1 for r in rows if r.get("embedding"))
        logger.debug(
            "FAQ search | total_rows=%d rows_with_embedding=%d lang=%s",
            len(rows),
            emb_count,
            lang,
        )

        best: FaqMatch | None = None

        for r in rows:
            try:
                # Skip entries without embeddings
                if not r.get("embedding"):
                    continue

                # Parse stored embedding
                try:
                    faq_vec = json.loads(r["embedding"])
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(
                        "Skipping FAQ %s: corrupted embedding - %s", r.get("id"), str(e)
                    )
                    continue

                # Validate parsed embedding
                if not isinstance(faq_vec, list) or len(faq_vec) == 0:
                    logger.warning(
                        "Skipping FAQ %s: invalid embedding format", r.get("id")
                    )
                    continue

                # Compute similarity score
                try:
                    score = dot(q_vec, faq_vec)
                except FAQError as e:
                    logger.warning(
                        "Skipping FAQ %s: similarity computation failed - %s",
                        r.get("id"),
                        str(e),
                    )
                    continue

                logger.debug(
                    "FAQ match | id=%s score=%.3f question=%s",
                    r["id"],
                    score,
                    r["question"][:40],
                )

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
            except Exception as e:
                # Log unexpected errors per-row but continue processing
                logger.warning(
                    "Unexpected error processing FAQ row %s: %s", r.get("id"), str(e)
                )
                continue

        # Apply confidence threshold
        if best and best.score >= 0.35:
            logger.info(
                "FAQ match found | id=%s score=%.3f question=%s",
                best.id,
                best.score,
                best.question[:40],
            )
            return best

        logger.debug(
            "No confident FAQ match found (threshold=0.35, best_score=%s)",
            best.score if best else "None",
        )
        return None

    except FAQError:
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        logger.error("Unexpected error in find_best_faq: %s", str(e), exc_info=True)
        raise FAQError(f"Unexpected error: {str(e)}") from e
