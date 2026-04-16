"""
Tests for app/faq.py

Uses mocking to avoid loading the real embedding model,
so tests run fast without downloading or running sentence-transformers.
"""

import pytest
from unittest.mock import patch
from app.faq import dot, find_best_faq, FaqMatch, FAQError


# -----------------------------------------------------------------------------
# dot()
# -----------------------------------------------------------------------------


def test_dot_product_identical_vectors():
    """Identical normalized vectors should give score close to 1.0."""
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    assert dot(a, b) == pytest.approx(1.0)


def test_dot_product_opposite_vectors():
    """Opposite vectors should give score close to -1.0."""
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert dot(a, b) == pytest.approx(-1.0)


def test_dot_product_orthogonal_vectors():
    """Perpendicular vectors should give score of 0.0."""
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert dot(a, b) == pytest.approx(0.0)


def test_dot_empty_vectors_raises():
    from app.faq import FAQError

    with pytest.raises(FAQError):
        dot([], [])


def test_dot_mismatched_lengths_raises():
    from app.faq import FAQError

    with pytest.raises(FAQError):
        dot([1.0, 0.0], [1.0, 0.0, 0.0])


# -----------------------------------------------------------------------------
# find_best_faq()
# -----------------------------------------------------------------------------

# Fake FAQ rows returned by db.fetch_faq_entries
FAKE_FAQ_ROWS = [
    {
        "id": 1,
        "question": "How do I request a refund?",
        "answer": "Go to Orders and click Request Refund.",
        "tags": "refund,payment",
        "language": "en",
        # Embedding pointing in direction [1, 0, 0]
        "embedding": "[1.0, 0.0, 0.0]",
    },
    {
        "id": 2,
        "question": "Where is my order?",
        "answer": "Track your order in Orders → Track shipment.",
        "tags": "shipping,order",
        "language": "en",
        # Embedding pointing in direction [0, 1, 0]
        "embedding": "[0.0, 1.0, 0.0]",
    },
]


def test_find_best_faq_returns_match_above_threshold():
    """Query embedding close to FAQ 1 should return FAQ 1."""
    # Query vector close to FAQ 1 embedding [1, 0, 0]
    query_vec = [0.99, 0.1, 0.0]

    with patch("app.faq.embed", return_value=query_vec), patch(
        "app.faq.db.fetch_faq_entries", return_value=FAKE_FAQ_ROWS
    ):

        result = find_best_faq("I want a refund")

        assert result is not None
        assert result.id == 1
        assert result.score > 0.35


def test_find_best_faq_returns_none_below_threshold():
    """Query embedding far from all FAQs should return None."""
    # Query vector orthogonal to all FAQ embeddings
    query_vec = [0.0, 0.0, 1.0]

    with patch("app.faq.embed", return_value=query_vec), patch(
        "app.faq.db.fetch_faq_entries", return_value=FAKE_FAQ_ROWS
    ):

        result = find_best_faq("something completely unrelated")

        assert result is None


def test_find_best_faq_empty_query_returns_none():
    """Empty query should return None without calling embed."""
    with patch("app.faq.embed") as mock_embed:
        result = find_best_faq("")
        mock_embed.assert_not_called()
        assert result is None


def test_find_best_faq_whitespace_query_returns_none():
    """Whitespace-only query should return None."""
    with patch("app.faq.embed") as mock_embed:
        result = find_best_faq("   ")
        mock_embed.assert_not_called()
        assert result is None


def test_find_best_faq_no_faq_entries_returns_none():
    """If DB has no FAQ entries, should return None."""
    with patch("app.faq.embed", return_value=[1.0, 0.0, 0.0]), patch(
        "app.faq.db.fetch_faq_entries", return_value=[]
    ):

        result = find_best_faq("I want a refund")
        assert result is None


def test_find_best_faq_skips_entries_without_embeddings():
    """FAQ entries missing embeddings should be skipped gracefully."""
    rows_without_embeddings = [
        {
            "id": 1,
            "question": "How do I request a refund?",
            "answer": "Go to Orders.",
            "tags": "refund",
            "language": "en",
            "embedding": None,  # No embedding
        }
    ]

    with patch("app.faq.embed", return_value=[1.0, 0.0, 0.0]), patch(
        "app.faq.db.fetch_faq_entries", return_value=rows_without_embeddings
    ):

        result = find_best_faq("I want a refund")
        assert result is None


def test_find_best_faq_skips_corrupted_embeddings():
    """FAQ entries with invalid JSON embeddings should be skipped."""
    rows_with_bad_embeddings = [
        {
            "id": 1,
            "question": "How do I request a refund?",
            "answer": "Go to Orders.",
            "tags": "refund",
            "language": "en",
            "embedding": "not valid json {{",
        }
    ]

    with patch("app.faq.embed", return_value=[1.0, 0.0, 0.0]), patch(
        "app.faq.db.fetch_faq_entries", return_value=rows_with_bad_embeddings
    ):

        result = find_best_faq("I want a refund")
        assert result is None


def test_find_best_faq_returns_highest_scoring_match():
    """Should return the FAQ with the highest similarity score."""
    # Query vector close to FAQ 2 [0, 1, 0]
    query_vec = [0.1, 0.99, 0.0]

    with patch("app.faq.embed", return_value=query_vec), patch(
        "app.faq.db.fetch_faq_entries", return_value=FAKE_FAQ_ROWS
    ):

        result = find_best_faq("where is my package")

        assert result is not None
        assert result.id == 2
