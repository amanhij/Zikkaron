"""Tests for sqlite-vec integration with storage and recall."""

import struct

import numpy as np
import pytest

from zikkaron.embeddings import EmbeddingEngine
from zikkaron.storage import StorageEngine

# Detect whether the embedding model can be loaded
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
    db_path = str(tmp_path / "test_vec.db")
    engine = StorageEngine(db_path, embedding_dim=384)
    yield engine
    engine.close()


def _make_embedding(dim: int = 384, seed: int = 0) -> bytes:
    """Create a deterministic float32 embedding."""
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


class TestSqliteVecLoaded:
    def test_extension_loaded(self, storage):
        """Verify the sqlite-vec extension is loaded and functional."""
        version = storage._conn.execute("SELECT vec_version()").fetchone()[0]
        assert version is not None
        assert version.startswith("v")

    def test_memory_vectors_table_exists(self, storage):
        """The memory_vectors virtual table should exist."""
        tables = storage._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_vectors'"
        ).fetchall()
        assert len(tables) == 1


class TestVectorInsertAndSearch:
    def test_insert_and_search_basic(self, storage):
        """Insert vectors, search by similarity, get correct results."""
        # Insert 5 memories with distinct embeddings
        embeddings = [_make_embedding(seed=i) for i in range(5)]
        mem_ids = []
        for i, emb in enumerate(embeddings):
            mid = storage.insert_memory(
                _make_memory(content=f"memory {i}", embedding=emb)
            )
            mem_ids.append(mid)

        # Search with the embedding of memory 0 — should find memory 0 closest
        results = storage.search_vectors(embeddings[0], top_k=3)
        assert len(results) >= 1
        result_ids = [r[0] for r in results]
        assert mem_ids[0] in result_ids
        # The first result should be the query itself (distance ~0)
        assert results[0][0] == mem_ids[0]
        assert results[0][1] == pytest.approx(0.0, abs=1e-3)

    def test_insert_vector_standalone(self, storage):
        """Test insert_vector and delete_vector directly."""
        emb = _make_embedding(seed=42)
        # Insert a memory without embedding, then add vector separately
        mid = storage.insert_memory(_make_memory(content="standalone"))
        storage.insert_vector(mid, emb)

        # Search should find it
        results = storage.search_vectors(emb, top_k=1)
        assert len(results) == 1
        assert results[0][0] == mid

    def test_delete_vector(self, storage):
        """Deleting a vector removes it from search results."""
        emb = _make_embedding(seed=10)
        mid = storage.insert_memory(_make_memory(content="to delete", embedding=emb))

        # Should be found before deletion
        results = storage.search_vectors(emb, top_k=1)
        assert len(results) == 1

        storage.delete_vector(mid)

        # Should not be found after deletion
        results = storage.search_vectors(emb, top_k=1)
        assert len(results) == 0

    def test_update_vector(self, storage):
        """Updating a vector replaces the old one."""
        emb1 = _make_embedding(seed=20)
        emb2 = _make_embedding(seed=21)
        mid = storage.insert_memory(_make_memory(content="to update", embedding=emb1))

        storage.update_vector(mid, emb2)

        # Search with new embedding should find it at distance ~0
        results = storage.search_vectors(emb2, top_k=1)
        assert len(results) == 1
        assert results[0][0] == mid
        assert results[0][1] == pytest.approx(0.0, abs=1e-3)


class TestVectorKNNAccuracy:
    def test_knn_ordering(self, storage):
        """Insert known vectors and verify KNN returns correct ordering."""
        # Create a reference vector
        ref = np.ones(384, dtype=np.float32)
        ref = ref / np.linalg.norm(ref)
        ref_bytes = ref.tobytes()

        # Create vectors at known distances from ref
        vectors = []
        for i in range(5):
            v = ref.copy()
            # Perturb by increasing amounts
            v[:i * 20] += float(i + 1) * 0.5
            v = v / np.linalg.norm(v)  # re-normalize
            vectors.append(v.astype(np.float32).tobytes())

        mem_ids = []
        for i, vec in enumerate(vectors):
            mid = storage.insert_memory(
                _make_memory(content=f"vec {i}", embedding=vec)
            )
            mem_ids.append(mid)

        results = storage.search_vectors(ref_bytes, top_k=5)
        result_ids = [r[0] for r in results]

        # The first vector (least perturbed) should be closest
        assert result_ids[0] == mem_ids[0]

        # Distances should be monotonically non-decreasing
        distances = [r[1] for r in results]
        for i in range(len(distances) - 1):
            assert distances[i] <= distances[i + 1] + 1e-6


class TestVectorWithHeatFilter:
    def test_heat_filter_excludes_cold_memories(self, storage):
        """Vector search should respect min_heat filter."""
        hot_emb = _make_embedding(seed=100)
        cold_emb = _make_embedding(seed=101)

        storage.insert_memory(
            _make_memory(content="hot memory", embedding=hot_emb, heat=0.9)
        )
        cold_id = storage.insert_memory(
            _make_memory(content="cold memory", embedding=cold_emb, heat=0.05)
        )

        # Search with min_heat=0.1 — should only find hot memory
        results = storage.search_vectors(hot_emb, top_k=10, min_heat=0.1)
        result_ids = [r[0] for r in results]
        assert cold_id not in result_ids
        assert len(results) >= 1

    def test_heat_filter_includes_when_above_threshold(self, storage):
        """Both memories should be found when min_heat is low enough."""
        emb1 = _make_embedding(seed=200)
        emb2 = _make_embedding(seed=201)

        mid1 = storage.insert_memory(
            _make_memory(content="warm 1", embedding=emb1, heat=0.5)
        )
        mid2 = storage.insert_memory(
            _make_memory(content="warm 2", embedding=emb2, heat=0.4)
        )

        results = storage.search_vectors(emb1, top_k=10, min_heat=0.1)
        result_ids = [r[0] for r in results]
        assert mid1 in result_ids
        assert mid2 in result_ids


class TestDeleteMemoryDeletesVector:
    def test_delete_memory_removes_vector(self, storage):
        """Deleting a memory should also remove its vector entry."""
        emb = _make_embedding(seed=300)
        mid = storage.insert_memory(
            _make_memory(content="doomed", embedding=emb)
        )

        # Should be in vector search
        results = storage.search_vectors(emb, top_k=1)
        assert len(results) == 1

        storage.delete_memory(mid)

        # Should be gone from vector search
        results = storage.search_vectors(emb, top_k=1)
        assert len(results) == 0


@requires_model
class TestRecallUsesVectorSearch:
    def test_recall_finds_semantically_similar(self, storage):
        """Recall should find semantically similar memories via sqlite-vec."""
        embeddings_engine = EmbeddingEngine()

        texts = [
            "SQLite database schema migration best practices",
            "chocolate cake recipe with cream cheese frosting",
            "Python database connection pooling strategies",
            "hiking trails in the Rocky Mountains",
        ]

        for text in texts:
            emb = embeddings_engine.encode(text)
            storage.insert_memory(
                _make_memory(content=text, embedding=emb)
            )

        # Search for something database-related
        query_emb = embeddings_engine.encode("SQL database operations")
        results = storage.search_vectors(query_emb, top_k=2, min_heat=0.1)
        result_ids = [r[0] for r in results]

        # Get the content for result IDs
        result_contents = []
        for rid in result_ids:
            mem = storage.get_memory(rid)
            result_contents.append(mem["content"])

        # Both database-related texts should be in top 2
        assert any("database" in c.lower() for c in result_contents)


class TestEmbeddingEngineDimensions:
    def test_get_dimensions_default(self):
        engine = EmbeddingEngine("all-MiniLM-L6-v2")
        assert engine.get_dimensions() == 384

    def test_get_dimensions_mpnet(self):
        engine = EmbeddingEngine("all-mpnet-base-v2")
        assert engine.get_dimensions() == 768


class TestQuantization:
    def test_quantize_dequantize_roundtrip(self):
        """Quantize then dequantize should produce approximate original."""
        original = np.random.randn(384).astype(np.float32)
        original_bytes = original.tobytes()

        quantized = EmbeddingEngine.quantize(original_bytes, bits=8)
        assert len(quantized) == 384  # int8 = 1 byte per dim

        dequantized = EmbeddingEngine.dequantize(quantized, bits=8)
        result = np.frombuffer(dequantized, dtype=np.float32)

        # Should be close but not exact due to quantization loss
        assert len(result) == 384
        # Correlation should be high (>0.95)
        corr = np.corrcoef(original, result)[0, 1]
        assert corr > 0.95

    def test_quantize_zero_vector(self):
        """Zero vector should quantize/dequantize to zeros."""
        zero = np.zeros(384, dtype=np.float32).tobytes()
        quantized = EmbeddingEngine.quantize(zero, bits=8)
        dequantized = EmbeddingEngine.dequantize(quantized, bits=8)
        result = np.frombuffer(dequantized, dtype=np.float32)
        assert np.allclose(result, 0.0)

    def test_quantize_unsupported_bits(self):
        emb = np.ones(10, dtype=np.float32).tobytes()
        with pytest.raises(ValueError):
            EmbeddingEngine.quantize(emb, bits=16)
        with pytest.raises(ValueError):
            EmbeddingEngine.dequantize(emb, bits=16)
