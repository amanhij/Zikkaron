"""Tests for Modern Hopfield Network energy-based retrieval."""

import numpy as np
import pytest

from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.hopfield import HopfieldMemory, _softmax, _sparsemax, _logsumexp
from zikkaron.knowledge_graph import KnowledgeGraph
from zikkaron.retrieval import HippoRetriever
from zikkaron.storage import StorageEngine


@pytest.fixture
def storage(tmp_path):
    engine = StorageEngine(str(tmp_path / "test_hopfield.db"))
    yield engine
    engine.close()


@pytest.fixture
def settings():
    return Settings(DB_PATH=":memory:", HOPFIELD_BETA=8.0, HOPFIELD_MAX_PATTERNS=5000)


@pytest.fixture
def embeddings():
    return EmbeddingEngine("all-MiniLM-L6-v2")


@pytest.fixture
def hopfield(storage, embeddings, settings):
    return HopfieldMemory(storage, embeddings, settings)


@pytest.fixture
def graph(storage, settings):
    return KnowledgeGraph(storage, settings)


@pytest.fixture
def retriever(storage, embeddings, graph, settings):
    return HippoRetriever(storage, embeddings, graph, settings)


def _make_memory(storage, embeddings, content, directory="/proj", tags=None, heat=1.0):
    """Insert a memory with embedding and return its ID."""
    embedding = embeddings.encode(content)
    mid = storage.insert_memory({
        "content": content,
        "embedding": embedding,
        "tags": tags or [],
        "directory_context": directory,
        "heat": heat,
        "is_stale": False,
        "file_hash": None,
        "embedding_model": embeddings.get_model_name(),
    })
    return mid


class TestBuildPatternMatrix:
    def test_build_from_storage(self, storage, embeddings, hopfield):
        """Pattern matrix builds correctly from stored memories."""
        _make_memory(storage, embeddings, "Python web development with Flask")
        _make_memory(storage, embeddings, "Database optimization techniques")
        _make_memory(storage, embeddings, "React frontend components")

        hopfield._build_pattern_matrix()

        assert hopfield._pattern_matrix is not None
        assert hopfield._pattern_matrix.shape[0] == 3
        assert hopfield._pattern_matrix.shape[1] == 384  # all-MiniLM-L6-v2 dim
        assert len(hopfield._pattern_ids) == 3
        assert not hopfield._dirty

    def test_build_empty_store(self, hopfield):
        """Building from empty store produces empty matrix."""
        hopfield._build_pattern_matrix()

        assert hopfield._pattern_matrix.size == 0
        assert len(hopfield._pattern_ids) == 0
        assert not hopfield._dirty

    def test_build_filters_cold_memories(self, storage, embeddings, settings, hopfield):
        """Cold memories (below COLD_THRESHOLD) are excluded."""
        _make_memory(storage, embeddings, "Hot memory", heat=1.0)
        _make_memory(storage, embeddings, "Cold memory", heat=0.01)

        hopfield._build_pattern_matrix()

        assert hopfield._pattern_matrix.shape[0] == 1

    def test_build_sorts_by_heat(self, storage, embeddings, hopfield):
        """Memories are sorted by heat descending."""
        m1 = _make_memory(storage, embeddings, "Low heat memory", heat=0.3)
        m2 = _make_memory(storage, embeddings, "High heat memory", heat=0.9)

        hopfield._build_pattern_matrix()

        # Higher heat memory should be first
        assert hopfield._pattern_ids[0] == m2
        assert hopfield._pattern_ids[1] == m1

    def test_rows_are_l2_normalized(self, storage, embeddings, hopfield):
        """Each row in the pattern matrix should be L2-normalized."""
        _make_memory(storage, embeddings, "Test normalization")

        hopfield._build_pattern_matrix()

        norms = np.linalg.norm(hopfield._pattern_matrix, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


class TestRetrieveBasic:
    def test_retrieve_returns_top_k(self, storage, embeddings, hopfield):
        """Retrieve returns correct number of results."""
        _make_memory(storage, embeddings, "Python web development with Flask")
        _make_memory(storage, embeddings, "Database optimization with indexes")
        _make_memory(storage, embeddings, "React component lifecycle")

        query_emb = embeddings.encode("Python Flask web server")
        results = hopfield.retrieve(query_emb, top_k=2)

        assert len(results) <= 2
        for mid, score in results:
            assert isinstance(mid, int)
            assert isinstance(score, float)
            assert 0 <= score <= 1.0

    def test_retrieve_scores_sum_to_one(self, storage, embeddings, hopfield):
        """Softmax attention weights should sum to ~1.0 across all patterns."""
        _make_memory(storage, embeddings, "Alpha topic content")
        _make_memory(storage, embeddings, "Beta topic content")
        _make_memory(storage, embeddings, "Gamma topic content")

        query_emb = embeddings.encode("Alpha topic")
        # Get all results
        results = hopfield.retrieve(query_emb, top_k=10)

        total = sum(score for _, score in results)
        assert abs(total - 1.0) < 0.01  # softmax sums to 1

    def test_retrieve_ranks_relevant_higher(self, storage, embeddings, hopfield):
        """More relevant memory should rank higher."""
        _make_memory(storage, embeddings, "Python machine learning with scikit-learn and numpy")
        _make_memory(storage, embeddings, "Italian cooking recipes for pasta and pizza")
        _make_memory(storage, embeddings, "Deep learning neural networks with PyTorch")

        query_emb = embeddings.encode("machine learning neural networks")
        results = hopfield.retrieve(query_emb, top_k=3)

        # The cooking recipe should not be the top result
        top_id = results[0][0]
        top_mem = storage.get_memory(top_id)
        assert "cooking" not in top_mem["content"].lower()


class TestRetrieveEmpty:
    def test_retrieve_empty_store(self, hopfield, embeddings):
        """Retrieve on empty store returns empty list."""
        query_emb = embeddings.encode("anything")
        results = hopfield.retrieve(query_emb, top_k=5)
        assert results == []

    def test_retrieve_sparse_empty_store(self, hopfield, embeddings):
        """Sparse retrieve on empty store returns empty list."""
        query_emb = embeddings.encode("anything")
        results = hopfield.retrieve_sparse(query_emb, top_k=5)
        assert results == []


class TestBetaSharpness:
    def test_higher_beta_more_concentrated(self, storage, embeddings, settings):
        """Higher beta produces more concentrated (peakier) attention."""
        _make_memory(storage, embeddings, "Python programming language features")
        _make_memory(storage, embeddings, "Java programming language features")
        _make_memory(storage, embeddings, "Cooking Italian pasta recipes")

        query_emb = embeddings.encode("Python programming")

        # Low beta: blended
        low_settings = Settings(DB_PATH=":memory:", HOPFIELD_BETA=1.0)
        hopfield_low = HopfieldMemory(storage, embeddings, low_settings)
        results_low = hopfield_low.retrieve(query_emb, top_k=3)

        # High beta: sharp
        high_settings = Settings(DB_PATH=":memory:", HOPFIELD_BETA=50.0)
        hopfield_high = HopfieldMemory(storage, embeddings, high_settings)
        results_high = hopfield_high.retrieve(query_emb, top_k=3)

        # High beta should have a more concentrated top score
        top_score_low = results_low[0][1]
        top_score_high = results_high[0][1]
        assert top_score_high > top_score_low

    def test_low_beta_distributes_weight(self, storage, embeddings):
        """Low beta distributes attention more evenly."""
        _make_memory(storage, embeddings, "Topic A first variation")
        _make_memory(storage, embeddings, "Topic A second variation")
        _make_memory(storage, embeddings, "Topic A third variation")

        query_emb = embeddings.encode("Topic A")

        low_settings = Settings(DB_PATH=":memory:", HOPFIELD_BETA=0.5)
        hopfield_low = HopfieldMemory(storage, embeddings, low_settings)
        results = hopfield_low.retrieve(query_emb, top_k=3)

        # With very low beta, scores should be relatively even
        scores = [s for _, s in results]
        spread = max(scores) - min(scores)
        assert spread < 0.5  # not too concentrated


class TestSparseRetrieval:
    def test_sparsemax_produces_zeros(self, storage, embeddings):
        """Sparsemax should produce exact zeros for irrelevant patterns."""
        _make_memory(storage, embeddings, "Python machine learning scikit-learn")
        _make_memory(storage, embeddings, "JavaScript React frontend components")
        _make_memory(storage, embeddings, "Italian cooking recipes for pasta")
        _make_memory(storage, embeddings, "Advanced deep learning PyTorch models")

        query_emb = embeddings.encode("Python machine learning")

        high_settings = Settings(DB_PATH=":memory:", HOPFIELD_BETA=20.0)
        hopfield = HopfieldMemory(storage, embeddings, high_settings)
        results = hopfield.retrieve_sparse(query_emb, top_k=10)

        # Sparse retrieval should return fewer results than total patterns
        # because irrelevant ones get exactly 0
        assert len(results) < 4  # not all 4 should be nonzero with high beta

        # All returned scores should be positive
        for _, score in results:
            assert score > 0

    def test_sparsemax_returns_valid_scores(self, storage, embeddings, hopfield):
        """Sparse retrieval returns valid positive scores."""
        _make_memory(storage, embeddings, "Test memory content A")
        _make_memory(storage, embeddings, "Test memory content B")

        query_emb = embeddings.encode("Test memory content")
        results = hopfield.retrieve_sparse(query_emb, top_k=5)

        assert len(results) > 0
        for mid, score in results:
            assert isinstance(mid, int)
            assert score > 0


class TestPatternCompletion:
    def test_completion_converges_toward_stored_pattern(self, storage, embeddings, hopfield):
        """Pattern completion should converge toward the closest stored pattern."""
        # Store a specific memory
        mid = _make_memory(storage, embeddings, "Python web development with Flask framework")
        stored_mem = storage.get_memory(mid)
        stored_emb = stored_mem["embedding"]

        # Create a noisy/partial query
        partial_emb = embeddings.encode("something about Python web")

        # Run pattern completion
        completed = hopfield.pattern_completion(partial_emb, iterations=10)

        # The completed embedding should be more similar to the stored pattern
        # than the partial query was
        stored_vec = np.frombuffer(stored_emb, dtype=np.float32)
        partial_vec = np.frombuffer(partial_emb, dtype=np.float32)
        completed_vec = np.frombuffer(completed, dtype=np.float32)

        # Normalize for cosine similarity
        stored_norm = stored_vec / np.linalg.norm(stored_vec)
        partial_norm = partial_vec / np.linalg.norm(partial_vec)
        completed_norm = completed_vec / np.linalg.norm(completed_vec)

        sim_before = float(np.dot(partial_norm, stored_norm))
        sim_after = float(np.dot(completed_norm, stored_norm))

        assert sim_after >= sim_before

    def test_completion_returns_correct_shape(self, storage, embeddings, hopfield):
        """Completed embedding has the same dimensionality."""
        _make_memory(storage, embeddings, "Test pattern for completion")

        query_emb = embeddings.encode("test query")
        completed = hopfield.pattern_completion(query_emb)

        query_vec = np.frombuffer(query_emb, dtype=np.float32)
        completed_vec = np.frombuffer(completed, dtype=np.float32)
        assert len(completed_vec) == len(query_vec)

    def test_completion_empty_store(self, hopfield, embeddings):
        """Pattern completion on empty store returns the input."""
        query_emb = embeddings.encode("test query")
        completed = hopfield.pattern_completion(query_emb)

        query_vec = np.frombuffer(query_emb, dtype=np.float32)
        completed_vec = np.frombuffer(completed, dtype=np.float32)

        # Should be the normalized input back
        expected = query_vec / np.linalg.norm(query_vec)
        np.testing.assert_allclose(completed_vec, expected, atol=1e-5)


class TestEnergyComputation:
    def test_lower_energy_for_matching_query(self, storage, embeddings, hopfield):
        """A matching query should have lower energy than an unrelated one."""
        _make_memory(storage, embeddings, "Python machine learning with scikit-learn")
        _make_memory(storage, embeddings, "Python data science with pandas and numpy")

        matching_emb = embeddings.encode("Python machine learning data science")
        unrelated_emb = embeddings.encode("Italian cooking recipes for desserts")

        energy_matching = hopfield.get_energy(matching_emb)
        energy_unrelated = hopfield.get_energy(unrelated_emb)

        assert energy_matching < energy_unrelated

    def test_energy_empty_store(self, hopfield, embeddings):
        """Energy on empty store returns just the norm term."""
        query_emb = embeddings.encode("test")
        energy = hopfield.get_energy(query_emb)
        # Should be 0.5 * |query|^2
        query_vec = np.frombuffer(query_emb, dtype=np.float32)
        expected = 0.5 * float(np.dot(query_vec, query_vec))
        assert abs(energy - expected) < 1e-3

    def test_energy_is_finite(self, storage, embeddings, hopfield):
        """Energy should always be a finite number."""
        _make_memory(storage, embeddings, "Test energy computation")
        query_emb = embeddings.encode("test query")
        energy = hopfield.get_energy(query_emb)
        assert np.isfinite(energy)


class TestCacheInvalidation:
    def test_dirty_flag_triggers_rebuild(self, storage, embeddings, hopfield):
        """Invalidating cache causes rebuild on next retrieve."""
        _make_memory(storage, embeddings, "Initial memory")

        # First retrieval builds cache
        query_emb = embeddings.encode("test")
        hopfield.retrieve(query_emb)
        assert not hopfield._dirty

        # Invalidate
        hopfield.invalidate_cache()
        assert hopfield._dirty

        # Add a new memory
        _make_memory(storage, embeddings, "New memory added")

        # Next retrieval should rebuild and include the new memory
        results = hopfield.retrieve(query_emb, top_k=10)
        assert not hopfield._dirty
        assert hopfield.get_pattern_count() == 2

    def test_invalidate_sets_dirty(self, hopfield):
        """invalidate_cache sets the dirty flag."""
        hopfield._dirty = False
        hopfield.invalidate_cache()
        assert hopfield._dirty


class TestRetrievalIntegration:
    def test_hopfield_scores_in_recall(self, storage, embeddings, graph, retriever):
        """Hopfield scores should appear in HippoRetriever.recall() results."""
        # Insert memories
        _make_memory(storage, embeddings, "Python web development with FastAPI")
        _make_memory(storage, embeddings, "Database optimization with indexes")
        _make_memory(storage, embeddings, "React frontend component design")

        results = retriever.recall("Python FastAPI web", max_results=5)

        # Should get results — the hopfield signal is one of 6 signals
        assert len(results) > 0
        for mem in results:
            assert "_retrieval_score" in mem
            assert mem["_retrieval_score"] > 0

    def test_hopfield_integrated_in_retriever(self, retriever):
        """HippoRetriever should have a _hopfield attribute."""
        assert hasattr(retriever, "_hopfield")
        assert isinstance(retriever._hopfield, HopfieldMemory)


class TestMaxPatternsLimit:
    def test_respects_max_patterns_cap(self, storage, embeddings):
        """Pattern matrix respects HOPFIELD_MAX_PATTERNS setting."""
        # Create settings with very low max
        low_max_settings = Settings(
            DB_PATH=":memory:",
            HOPFIELD_BETA=8.0,
            HOPFIELD_MAX_PATTERNS=3,
        )
        hopfield = HopfieldMemory(storage, embeddings, low_max_settings)

        # Insert more memories than the cap
        for i in range(10):
            _make_memory(storage, embeddings, f"Memory number {i} about topic {i}")

        hopfield._build_pattern_matrix()

        assert hopfield._pattern_matrix.shape[0] == 3
        assert len(hopfield._pattern_ids) == 3

    def test_highest_heat_retained(self, storage, embeddings):
        """When capped, highest-heat memories are retained."""
        low_max_settings = Settings(
            DB_PATH=":memory:",
            HOPFIELD_BETA=8.0,
            HOPFIELD_MAX_PATTERNS=2,
        )
        hopfield = HopfieldMemory(storage, embeddings, low_max_settings)

        m_low = _make_memory(storage, embeddings, "Low heat", heat=0.2)
        m_mid = _make_memory(storage, embeddings, "Mid heat", heat=0.5)
        m_high = _make_memory(storage, embeddings, "High heat", heat=0.9)

        hopfield._build_pattern_matrix()

        # Only top-2 by heat should be retained
        assert m_high in hopfield._pattern_ids
        assert m_mid in hopfield._pattern_ids
        assert m_low not in hopfield._pattern_ids


class TestHelperFunctions:
    def test_softmax_sums_to_one(self):
        logits = np.array([1.0, 2.0, 3.0])
        result = _softmax(logits)
        assert abs(np.sum(result) - 1.0) < 1e-6

    def test_softmax_monotonic(self):
        logits = np.array([1.0, 2.0, 3.0])
        result = _softmax(logits)
        assert result[2] > result[1] > result[0]

    def test_softmax_numerical_stability(self):
        """Large logits should not cause overflow."""
        logits = np.array([1000.0, 1001.0, 1002.0])
        result = _softmax(logits)
        assert np.all(np.isfinite(result))
        assert abs(np.sum(result) - 1.0) < 1e-6

    def test_sparsemax_produces_zeros(self):
        logits = np.array([5.0, 1.0, 0.1, 0.01])
        result = _sparsemax(logits)
        # The dominant entry should have weight, small ones may be zero
        assert result[0] > 0
        assert np.sum(result == 0) >= 1  # at least one zero

    def test_sparsemax_sums_correctly(self):
        logits = np.array([3.0, 2.0, 1.0])
        result = _sparsemax(logits)
        # Sparsemax projects onto simplex: weights sum to 1
        assert abs(np.sum(result) - 1.0) < 1e-6

    def test_logsumexp_matches_numpy(self):
        x = np.array([1.0, 2.0, 3.0])
        result = _logsumexp(x)
        expected = np.log(np.sum(np.exp(x)))
        assert abs(result - expected) < 1e-6

    def test_logsumexp_numerical_stability(self):
        x = np.array([1000.0, 1001.0, 1002.0])
        result = _logsumexp(x)
        assert np.isfinite(result)


class TestGetPatternCount:
    def test_count_reflects_memories(self, storage, embeddings, hopfield):
        _make_memory(storage, embeddings, "Memory A")
        _make_memory(storage, embeddings, "Memory B")
        assert hopfield.get_pattern_count() == 2

    def test_count_zero_empty(self, hopfield):
        assert hopfield.get_pattern_count() == 0
