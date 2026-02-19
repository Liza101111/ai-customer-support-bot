from __future__ import annotations

from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    """
    Load the embedding model once (cached).
    First run downloads the model, later runs use local cache.
    """
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed(text: str) -> List[float]:
    """
    Convert text to an embedding vector (list[float]).
    """
    vec = _model().encode(text, normalize_embeddings=True)
    return vec.tolist()
