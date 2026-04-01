from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Custom exception for embedding-related errors."""

    pass


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    """
    Load the embedding model once (cached).
    First run downloads the model, later runs use local cache.

    Raises:
        EmbeddingError: If model loading fails.
    """
    try:
        logger.debug("Loading embedding model: sentence-transformers/all-MiniLM-L6-v2")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Embedding model loaded successfully")
        return model
    except Exception as e:
        error_msg = f"Failed to load embedding model: {str(e)}"
        logger.error(error_msg)
        raise EmbeddingError(error_msg) from e


def embed(text: str) -> List[float]:
    """
    Convert text to an embedding vector (list[float]).

    Args:
        text: Input text to embed.

    Returns:
        List of floats representing the embedding vector.

    Raises:
        EmbeddingError: If text is empty, model fails to load, or encoding fails.
    """
    # Validate input
    if not text or not isinstance(text, str):
        raise EmbeddingError("Text must be a non-empty string")

    text = text.strip()
    if not text:
        raise EmbeddingError("Text cannot be empty or whitespace-only")

    try:
        logger.debug("Embedding text: %s", text[:50])
        model = _model()
        vec = model.encode(text, normalize_embeddings=True)
        embedding = vec.tolist()
        logger.debug("Embedding generated successfully, dimension=%d", len(embedding))
        return embedding
    except EmbeddingError:
        # Re-raise our custom exceptions as-is
        raise
    except Exception as e:
        error_msg = f"Failed to generate embedding: {str(e)}"
        logger.error(error_msg)
        raise EmbeddingError(error_msg) from e
