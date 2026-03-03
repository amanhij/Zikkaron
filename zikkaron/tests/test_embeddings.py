"""Tests for the embedding engine."""

import numpy as np
import pytest

from zikkaron.embeddings import EmbeddingEngine

# Detect whether the model can actually be loaded
_engine = EmbeddingEngine()
try:
    _engine._ensure_model()
    _model_available = not _engine._unavailable
except Exception:
    _model_available = False

requires_model = pytest.mark.skipif(
    not _model_available,
    reason="sentence-transformers model not available",
)


@requires_model
def test_encode_returns_bytes():
    engine = EmbeddingEngine()
    result = engine.encode("hello world")
    assert isinstance(result, bytes)
    assert len(result) > 0
    # Should be float32 elements — length divisible by 4
    assert len(result) % 4 == 0


@requires_model
def test_encode_deterministic():
    engine = EmbeddingEngine()
    a = engine.encode("deterministic test input")
    b = engine.encode("deterministic test input")
    assert a == b


@requires_model
def test_similarity_identical():
    engine = EmbeddingEngine()
    emb = engine.encode("the quick brown fox")
    score = engine.similarity(emb, emb)
    assert score == pytest.approx(1.0, abs=1e-5)


@requires_model
def test_similarity_different():
    engine = EmbeddingEngine()
    emb_a = engine.encode("python programming language syntax")
    emb_b = engine.encode("underwater basket weaving techniques")
    score = engine.similarity(emb_a, emb_b)
    assert score < 0.7


@requires_model
def test_search_ranking():
    engine = EmbeddingEngine()
    texts = [
        "SQLite database schema migration",
        "chocolate cake recipe with frosting",
        "database connection pooling in Python",
        "hiking trails in the mountains",
    ]
    embeddings = [(i, engine.encode(t)) for i, t in enumerate(texts)]
    query = engine.encode("SQL database operations")
    results = engine.search(query, embeddings, top_k=4)

    # The two database-related texts (ids 0 and 2) should rank in the top 2
    top_ids = {results[0][0], results[1][0]}
    assert 0 in top_ids
    assert 2 in top_ids


@requires_model
def test_batch_encode():
    engine = EmbeddingEngine()
    texts = ["hello world", "goodbye world", "hello world"]
    batch = engine.encode_batch(texts)
    individual = [engine.encode(t) for t in texts]

    assert len(batch) == 3
    for b, i in zip(batch, individual):
        assert isinstance(b, bytes)
        b_arr = np.frombuffer(b, dtype=np.float32)
        i_arr = np.frombuffer(i, dtype=np.float32)
        np.testing.assert_allclose(b_arr, i_arr, atol=1e-5)


def test_encode_without_model():
    """Verify graceful degradation when sentence-transformers is unavailable."""
    engine = EmbeddingEngine()
    # Simulate unavailable model
    engine._unavailable = True
    assert engine.encode("test") is None


def test_batch_encode_without_model():
    engine = EmbeddingEngine()
    engine._unavailable = True
    result = engine.encode_batch(["a", "b", "c"])
    assert result == [None, None, None]


def test_similarity_zero_norm():
    """Similarity with a zero vector should return 0.0."""
    engine = EmbeddingEngine()
    zero = np.zeros(384, dtype=np.float32).tobytes()
    nonzero = np.ones(384, dtype=np.float32).tobytes()
    assert engine.similarity(zero, nonzero) == 0.0
