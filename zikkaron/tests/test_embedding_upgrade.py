"""Tests for embedding upgrade: multi-model support, Matryoshka, versioning, re-embedding."""

import numpy as np
import pytest

from zikkaron.embeddings import EmbeddingEngine, MODEL_DIMENSIONS
from zikkaron.storage import StorageEngine

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


@pytest.fixture
def storage(tmp_path):
    db_path = str(tmp_path / "test_emb_upgrade.db")
    engine = StorageEngine(db_path, embedding_dim=384)
    yield engine
    engine.close()


def _make_embedding(dim: int = 384, seed: int = 0) -> bytes:
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    return vec.tobytes()


def _make_memory(content="test memory", directory="/tmp/project", **kwargs):
    base = {
        "content": content,
        "directory_context": directory,
        "tags": ["test"],
    }
    base.update(kwargs)
    return base


class TestModelDimensions:
    """Verify MODEL_DIMENSIONS has correct entries for all supported models."""

    def test_minilm_l6(self):
        assert MODEL_DIMENSIONS["all-MiniLM-L6-v2"] == 384

    def test_bge_small(self):
        assert MODEL_DIMENSIONS["BAAI/bge-small-en-v1.5"] == 384

    def test_bge_base(self):
        assert MODEL_DIMENSIONS["BAAI/bge-base-en-v1.5"] == 768

    def test_nomic(self):
        assert MODEL_DIMENSIONS["nomic-ai/nomic-embed-text-v1.5"] == 768

    def test_get_dimensions_uses_dict(self):
        for model_name, expected_dim in MODEL_DIMENSIONS.items():
            engine = EmbeddingEngine(model_name)
            assert engine.get_dimensions() == expected_dim

    def test_get_dimensions_fallback_default(self):
        """Unknown model with no sentence-transformers returns 384 default."""
        engine = EmbeddingEngine("unknown/model-xyz")
        engine._unavailable = True
        assert engine.get_dimensions() == 384


class TestGetModelName:
    def test_default_model_name(self):
        engine = EmbeddingEngine()
        assert engine.get_model_name() == "all-MiniLM-L6-v2"

    def test_custom_model_name(self):
        engine = EmbeddingEngine("BAAI/bge-small-en-v1.5")
        assert engine.get_model_name() == "BAAI/bge-small-en-v1.5"


class TestAdaptiveDimensions:
    """Test Matryoshka (adaptive dimensionality) truncation."""

    @requires_model
    def test_adaptive_full_dimensions(self):
        """With dimensions=None or dimensions=native, returns full embedding."""
        engine = EmbeddingEngine()
        full = engine.encode("test text")
        adaptive = engine.encode_adaptive("test text")
        assert len(full) == len(adaptive)

    @requires_model
    def test_adaptive_truncated(self):
        """Truncated embedding should have fewer dimensions."""
        engine = EmbeddingEngine()
        target_dim = 128
        result = engine.encode_adaptive("test text", dimensions=target_dim)
        assert result is not None
        arr = np.frombuffer(result, dtype=np.float32)
        assert len(arr) == target_dim

    @requires_model
    def test_adaptive_renormalized(self):
        """Truncated embedding should be unit-normalized."""
        engine = EmbeddingEngine()
        result = engine.encode_adaptive("test text", dimensions=128)
        arr = np.frombuffer(result, dtype=np.float32)
        norm = np.linalg.norm(arr)
        assert norm == pytest.approx(1.0, abs=1e-5)

    @requires_model
    def test_adaptive_various_dimensions(self):
        """Multiple truncation sizes should all work."""
        engine = EmbeddingEngine()
        for dim in [32, 64, 128, 256]:
            result = engine.encode_adaptive("hello world", dimensions=dim)
            assert result is not None
            arr = np.frombuffer(result, dtype=np.float32)
            assert len(arr) == dim

    def test_adaptive_unavailable(self):
        """Should return None when model is unavailable."""
        engine = EmbeddingEngine()
        engine._unavailable = True
        assert engine.encode_adaptive("test") is None


class TestNeedsReembedding:
    def test_same_model(self):
        engine = EmbeddingEngine("all-MiniLM-L6-v2")
        assert engine.needs_reembedding("all-MiniLM-L6-v2") is False

    def test_different_model(self):
        engine = EmbeddingEngine("BAAI/bge-small-en-v1.5")
        assert engine.needs_reembedding("all-MiniLM-L6-v2") is True

    def test_none_model(self):
        engine = EmbeddingEngine()
        assert engine.needs_reembedding(None) is True


class TestBatchReembed:
    @requires_model
    def test_batch_reembed_matches_encode_batch(self):
        engine = EmbeddingEngine()
        texts = ["hello", "world", "test"]
        reembedded = engine.batch_reembed(texts)
        encoded = engine.encode_batch(texts)
        assert len(reembedded) == len(encoded)
        for r, e in zip(reembedded, encoded):
            assert r == e

    def test_batch_reembed_unavailable(self):
        engine = EmbeddingEngine()
        engine._unavailable = True
        result = engine.batch_reembed(["a", "b"])
        assert result == [None, None]


class TestModelVersioningStorage:
    """Verify embedding_model is stored and retrievable in storage."""

    def test_embedding_model_stored(self, storage):
        emb = _make_embedding(seed=1)
        mid = storage.insert_memory(
            _make_memory(
                content="versioned memory",
                embedding=emb,
                embedding_model="all-MiniLM-L6-v2",
            )
        )
        mem = storage.get_memory(mid)
        assert mem["embedding_model"] == "all-MiniLM-L6-v2"

    def test_embedding_model_null_by_default(self, storage):
        emb = _make_embedding(seed=2)
        mid = storage.insert_memory(
            _make_memory(content="no model", embedding=emb)
        )
        mem = storage.get_memory(mid)
        assert mem["embedding_model"] is None

    def test_get_memories_needing_reembedding(self, storage):
        emb1 = _make_embedding(seed=10)
        emb2 = _make_embedding(seed=11)
        emb3 = _make_embedding(seed=12)

        # Memory with current model — should NOT need re-embedding
        storage.insert_memory(
            _make_memory(
                content="current model",
                embedding=emb1,
                embedding_model="BAAI/bge-small-en-v1.5",
            )
        )

        # Memory with old model — should need re-embedding
        mid_old = storage.insert_memory(
            _make_memory(
                content="old model",
                embedding=emb2,
                embedding_model="all-MiniLM-L6-v2",
            )
        )

        # Memory with no model tag — should need re-embedding
        mid_null = storage.insert_memory(
            _make_memory(content="null model", embedding=emb3)
        )

        needs = storage.get_memories_needing_reembedding("BAAI/bge-small-en-v1.5")
        need_ids = {m["id"] for m in needs}
        assert mid_old in need_ids
        assert mid_null in need_ids
        assert len(needs) == 2

    def test_get_memories_needing_reembedding_empty(self, storage):
        """No memories need re-embedding when all match current model."""
        emb = _make_embedding(seed=20)
        storage.insert_memory(
            _make_memory(
                content="up to date",
                embedding=emb,
                embedding_model="all-MiniLM-L6-v2",
            )
        )
        needs = storage.get_memories_needing_reembedding("all-MiniLM-L6-v2")
        assert len(needs) == 0

    def test_update_memory_embedding(self, storage):
        """update_memory_embedding should update both blob and model name."""
        emb_old = _make_embedding(seed=30)
        mid = storage.insert_memory(
            _make_memory(
                content="to reembed",
                embedding=emb_old,
                embedding_model="all-MiniLM-L6-v2",
            )
        )

        emb_new = _make_embedding(seed=31)
        storage.update_memory_embedding(mid, emb_new, "BAAI/bge-small-en-v1.5")

        mem = storage.get_memory(mid)
        assert mem["embedding_model"] == "BAAI/bge-small-en-v1.5"
        assert mem["embedding"] == emb_new

        # Vector search should find with new embedding
        results = storage.search_vectors(emb_new, top_k=1)
        assert len(results) >= 1
        assert results[0][0] == mid


class TestRecreateVectorTable:
    def test_recreate_with_new_dim(self, tmp_path):
        """Recreating vec0 table with new dimensions works."""
        db_path = str(tmp_path / "test_recreate.db")
        storage = StorageEngine(db_path, embedding_dim=384)

        # Insert a memory with 384-dim embedding
        emb_384 = _make_embedding(dim=384, seed=50)
        storage.insert_memory(
            _make_memory(content="384 dim", embedding=emb_384)
        )

        # Recreate with 768 dimensions
        storage.recreate_vector_table(768)

        # Old vectors are gone — verify by searching with a 768-dim query
        emb_768_query = _make_embedding(dim=768, seed=99)
        results = storage.search_vectors(emb_768_query, top_k=10)
        assert len(results) == 0

        # Can now insert 768-dim vectors
        emb_768 = _make_embedding(dim=768, seed=51)
        mid = storage.insert_memory(
            _make_memory(content="768 dim")
        )
        storage.insert_vector(mid, emb_768)
        results = storage.search_vectors(emb_768, top_k=1)
        assert len(results) == 1
        assert results[0][0] == mid

        storage.close()
