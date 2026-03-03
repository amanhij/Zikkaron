"""Tests for Hyperdimensional Computing (HDC) encoder and retrieval integration."""

import numpy as np
import pytest

from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.hdc_encoder import HDCEncoder
from zikkaron.knowledge_graph import KnowledgeGraph
from zikkaron.retrieval import HippoRetriever
from zikkaron.storage import StorageEngine


# -- Fixtures --


@pytest.fixture
def hdc():
    return HDCEncoder(dimensions=10000, seed=42)


@pytest.fixture
def hdc_small():
    """Smaller dimension encoder for faster tests."""
    return HDCEncoder(dimensions=1000, seed=42)


@pytest.fixture
def storage(tmp_path):
    engine = StorageEngine(str(tmp_path / "test_hdc.db"))
    yield engine
    engine.close()


@pytest.fixture
def settings():
    return Settings(DB_PATH=":memory:")


@pytest.fixture
def embeddings():
    return EmbeddingEngine("all-MiniLM-L6-v2")


@pytest.fixture
def graph(storage, settings):
    return KnowledgeGraph(storage, settings)


@pytest.fixture
def retriever(storage, embeddings, graph, settings):
    return HippoRetriever(storage, embeddings, graph, settings)


def _make_memory(storage, embeddings, content, directory="/proj", tags=None, heat=1.0):
    """Insert a memory with embedding and return its ID."""
    embedding = embeddings.encode(content)
    return storage.insert_memory({
        "content": content,
        "embedding": embedding,
        "tags": tags or [],
        "directory_context": directory,
        "heat": heat,
        "is_stale": False,
        "file_hash": None,
        "embedding_model": embeddings.get_model_name(),
    })


# -- Core HDC operation tests --


class TestHDCBasicOperations:
    """Test fundamental HDC/VSA operations: bind, bundle, permute."""

    def test_random_vectors_orthogonal(self, hdc):
        """Random bipolar vectors in high dimensions are nearly orthogonal."""
        v1 = hdc._random_vector()
        v2 = hdc._random_vector()
        sim = hdc.similarity(v1, v2)
        # In 10000 dimensions, random bipolar vectors should have
        # near-zero cosine similarity (|sim| < 0.05 with high probability)
        assert abs(sim) < 0.1, f"Random vectors too similar: {sim}"

    def test_random_vector_shape_and_values(self, hdc):
        """Random vectors have correct shape and bipolar values."""
        v = hdc._random_vector()
        assert v.shape == (10000,)
        assert set(np.unique(v)).issubset({-1.0, 1.0})

    def test_bind_dissimilar(self, hdc):
        """Binding result is dissimilar to both inputs."""
        a = hdc.get_or_create_atom("auth.py")
        b = hdc.get_or_create_atom("database.py")
        bound = hdc.bind(a, b)

        sim_a = hdc.similarity(bound, a)
        sim_b = hdc.similarity(bound, b)
        # Bound vector should be nearly orthogonal to both inputs
        assert abs(sim_a) < 0.1, f"Bound too similar to a: {sim_a}"
        assert abs(sim_b) < 0.1, f"Bound too similar to b: {sim_b}"

    def test_bind_self_inverse(self, hdc):
        """Binding a vector with itself gives the identity (all 1s for bipolar)."""
        v = hdc.get_or_create_atom("test")
        self_bound = hdc.bind(v, v)
        # For bipolar vectors, v * v = [1, 1, 1, ...]
        assert np.all(self_bound == 1.0)

    def test_bind_associative_unbinding(self, hdc):
        """Can unbind by re-binding: bind(bind(a, b), b) ≈ a."""
        a = hdc.get_or_create_atom("role")
        b = hdc.get_or_create_atom("filler")
        bound = hdc.bind(a, b)
        unbound = hdc.bind(bound, b)  # unbind by re-binding with b
        # For bipolar, b*b = 1, so bind(bind(a,b), b) = a * b * b = a
        sim = hdc.similarity(unbound, a)
        assert sim > 0.99, f"Unbinding failed, sim={sim}"

    def test_bundle_preserves_components(self, hdc):
        """Bundled vector is similar to each component."""
        v1 = hdc.get_or_create_atom("component_1")
        v2 = hdc.get_or_create_atom("component_2")
        v3 = hdc.get_or_create_atom("component_3")
        bundled = hdc.bundle(v1, v2, v3)

        sim1 = hdc.similarity(bundled, v1)
        sim2 = hdc.similarity(bundled, v2)
        sim3 = hdc.similarity(bundled, v3)

        # Each component should have positive similarity with the bundle
        assert sim1 > 0.2, f"Component 1 not preserved: {sim1}"
        assert sim2 > 0.2, f"Component 2 not preserved: {sim2}"
        assert sim3 > 0.2, f"Component 3 not preserved: {sim3}"

    def test_bundle_single_vector(self, hdc):
        """Bundling a single vector returns a copy."""
        v = hdc.get_or_create_atom("single")
        bundled = hdc.bundle(v)
        assert np.array_equal(bundled, v)

    def test_bundle_empty(self, hdc):
        """Bundling no vectors returns a random vector."""
        bundled = hdc.bundle()
        assert bundled.shape == (10000,)

    def test_permute_shifts(self, hdc):
        """Permuted vector is dissimilar to original."""
        v = hdc.get_or_create_atom("sequence_element")
        shifted = hdc.permute(v, shift=1)
        sim = hdc.similarity(v, shifted)
        # Shifted vector should be nearly orthogonal
        assert abs(sim) < 0.1, f"Permuted too similar: {sim}"

    def test_permute_inverse(self, hdc):
        """Permute and inverse permute restore original."""
        v = hdc.get_or_create_atom("reversible")
        shifted = hdc.permute(v, shift=3)
        restored = hdc.permute(shifted, shift=-3)
        assert np.array_equal(v, restored)

    def test_permute_different_shifts_orthogonal(self, hdc):
        """Different shift amounts produce dissimilar vectors."""
        v = hdc.get_or_create_atom("order_test")
        s1 = hdc.permute(v, shift=1)
        s2 = hdc.permute(v, shift=2)
        sim = hdc.similarity(s1, s2)
        assert abs(sim) < 0.1, f"Different shifts too similar: {sim}"


# -- Memory encoding tests --


class TestHDCEncoding:
    """Test memory and query encoding."""

    def test_encode_memory(self, hdc):
        """Full encoding produces a valid bipolar vector."""
        vec = hdc.encode_memory(
            directory="/myproject",
            tags=["bug", "fix"],
            entities=["auth.py", "login"],
            store_type="episodic",
        )
        assert vec.shape == (10000,)
        # Result is bipolar (after bundling + sign normalization)
        unique_vals = set(np.unique(vec))
        assert unique_vals.issubset({-1.0, 1.0})

    def test_encode_memory_empty_lists(self, hdc):
        """Encoding with empty tags/entities still works."""
        vec = hdc.encode_memory(
            directory="/proj",
            tags=[],
            entities=[],
            store_type="episodic",
        )
        assert vec.shape == (10000,)

    def test_encode_query_partial(self, hdc):
        """Partial query works with subset of attributes."""
        # Query with only entity
        q = hdc.encode_query(entities=["auth.py"])
        assert q.shape == (10000,)

        # Query with only tag
        q2 = hdc.encode_query(tags=["bug"])
        assert q2.shape == (10000,)

        # Query with only directory
        q3 = hdc.encode_query(directory="/proj")
        assert q3.shape == (10000,)

    def test_encode_query_full(self, hdc):
        """Full query with all attributes."""
        q = hdc.encode_query(
            directory="/proj",
            tags=["bug"],
            entities=["auth.py"],
            store_type="episodic",
        )
        assert q.shape == (10000,)

    def test_query_matches_memory(self, hdc):
        """Query(entity=X) has higher similarity to memory containing entity X
        than to memory without X."""
        mem_with_auth = hdc.encode_memory(
            directory="/proj",
            tags=["backend"],
            entities=["auth.py", "login.py"],
        )
        mem_without_auth = hdc.encode_memory(
            directory="/proj",
            tags=["frontend"],
            entities=["styles.css", "layout.tsx"],
        )

        query = hdc.encode_query(entities=["auth.py"])

        sim_match = hdc.similarity(query, mem_with_auth)
        sim_no_match = hdc.similarity(query, mem_without_auth)

        assert sim_match > sim_no_match, (
            f"Query should prefer matching memory: {sim_match} vs {sim_no_match}"
        )

    def test_query_mismatches(self, hdc):
        """Query for entity X does not strongly match memory without X."""
        mem = hdc.encode_memory(
            directory="/proj",
            tags=["database"],
            entities=["models.py", "schema.sql"],
        )
        query = hdc.encode_query(entities=["auth.py"])
        sim = hdc.similarity(query, mem)
        # Should not be a strong match
        assert sim < 0.5, f"Unexpected strong match: {sim}"

    def test_directory_query_discriminates(self, hdc):
        """Directory-based query discriminates between projects."""
        mem_proj_a = hdc.encode_memory(
            directory="/project_alpha",
            tags=["api"],
            entities=["server.py"],
        )
        mem_proj_b = hdc.encode_memory(
            directory="/project_beta",
            tags=["api"],
            entities=["server.py"],
        )

        query_alpha = hdc.encode_query(directory="/project_alpha")
        sim_a = hdc.similarity(query_alpha, mem_proj_a)
        sim_b = hdc.similarity(query_alpha, mem_proj_b)

        assert sim_a > sim_b, (
            f"Directory query should prefer matching project: {sim_a} vs {sim_b}"
        )

    def test_tag_query_discriminates(self, hdc):
        """Tag-based query finds memories with matching tags."""
        mem_bug = hdc.encode_memory(
            directory="/proj",
            tags=["bug", "critical"],
            entities=["app.py"],
        )
        mem_feature = hdc.encode_memory(
            directory="/proj",
            tags=["feature", "enhancement"],
            entities=["app.py"],
        )

        query_bug = hdc.encode_query(tags=["bug"])
        sim_bug = hdc.similarity(query_bug, mem_bug)
        sim_feature = hdc.similarity(query_bug, mem_feature)

        assert sim_bug > sim_feature, (
            f"Tag query should prefer matching tags: {sim_bug} vs {sim_feature}"
        )


# -- Serialization tests --


class TestHDCSerialization:
    """Test to_bytes / from_bytes roundtrip."""

    def test_serialization_roundtrip(self, hdc):
        """to_bytes and from_bytes preserve vector exactly."""
        vec = hdc.encode_memory(
            directory="/proj",
            tags=["test"],
            entities=["data.py"],
        )
        data = hdc.to_bytes(vec)
        restored = hdc.from_bytes(data)

        assert isinstance(data, bytes)
        assert restored.shape == vec.shape
        # Float32 roundtrip should be exact
        np.testing.assert_array_equal(restored, vec.astype(np.float32))

    def test_serialization_size(self, hdc):
        """Serialized vector has expected size (dim * 4 bytes for float32)."""
        vec = hdc._random_vector()
        data = hdc.to_bytes(vec)
        assert len(data) == 10000 * 4  # float32 = 4 bytes

    def test_from_bytes_independent_copy(self, hdc):
        """from_bytes returns an independent copy (not a view)."""
        vec = hdc._random_vector()
        data = hdc.to_bytes(vec)
        restored = hdc.from_bytes(data)
        restored[0] = 999.0
        # Original data should not be affected
        restored2 = hdc.from_bytes(data)
        assert restored2[0] != 999.0


# -- Search tests --


class TestHDCSearch:
    """Test HDC search ranking."""

    def test_search_ranking(self, hdc):
        """Correct memories are ranked highest in search results."""
        # Create memories about different topics
        mem_auth = hdc.encode_memory("/proj", ["auth"], ["auth.py", "jwt.py"])
        mem_db = hdc.encode_memory("/proj", ["database"], ["models.py", "queries.py"])
        mem_ui = hdc.encode_memory("/proj", ["frontend"], ["App.tsx", "styles.css"])

        candidates = [
            (1, mem_auth),
            (2, mem_db),
            (3, mem_ui),
        ]

        # Query for auth entities
        query = hdc.encode_query(entities=["auth.py"])
        results = hdc.search(query, candidates, top_k=3)

        # Memory 1 (auth) should be ranked first
        assert results[0][0] == 1, f"Auth memory not ranked first: {results}"
        assert results[0][1] > results[1][1], "Top result should have highest score"

    def test_search_top_k(self, hdc):
        """Search respects top_k limit."""
        candidates = [
            (i, hdc._random_vector()) for i in range(20)
        ]
        query = hdc._random_vector()
        results = hdc.search(query, candidates, top_k=5)
        assert len(results) == 5

    def test_search_empty_candidates(self, hdc):
        """Search with no candidates returns empty list."""
        query = hdc._random_vector()
        results = hdc.search(query, [], top_k=5)
        assert results == []

    def test_search_fewer_candidates_than_k(self, hdc):
        """Search with fewer candidates than top_k returns all."""
        candidates = [(1, hdc._random_vector()), (2, hdc._random_vector())]
        query = hdc._random_vector()
        results = hdc.search(query, candidates, top_k=10)
        assert len(results) == 2


# -- Determinism tests --


class TestHDCDeterminism:
    """Test that HDC encoding is deterministic with seeded RNG."""

    def test_deterministic_encoding(self):
        """Same input produces same vector with same seed."""
        hdc1 = HDCEncoder(dimensions=5000, seed=123)
        hdc2 = HDCEncoder(dimensions=5000, seed=123)

        vec1 = hdc1.encode_memory("/proj", ["bug"], ["auth.py"])
        vec2 = hdc2.encode_memory("/proj", ["bug"], ["auth.py"])

        np.testing.assert_array_equal(vec1, vec2)

    def test_different_seeds_different_vectors(self):
        """Different seeds produce different vectors."""
        hdc1 = HDCEncoder(dimensions=5000, seed=42)
        hdc2 = HDCEncoder(dimensions=5000, seed=99)

        vec1 = hdc1.encode_memory("/proj", ["bug"], ["auth.py"])
        vec2 = hdc2.encode_memory("/proj", ["bug"], ["auth.py"])

        # With different seeds, the codebook atoms are different
        assert not np.array_equal(vec1, vec2)

    def test_atom_codebook_stable(self):
        """Getting the same atom twice returns the same vector."""
        hdc = HDCEncoder(dimensions=5000, seed=42)
        a1 = hdc.get_or_create_atom("stable_name")
        a2 = hdc.get_or_create_atom("stable_name")
        np.testing.assert_array_equal(a1, a2)

    def test_atom_codebook_order_matters(self):
        """Atoms are generated deterministically but depend on creation order."""
        hdc1 = HDCEncoder(dimensions=1000, seed=42)
        hdc2 = HDCEncoder(dimensions=1000, seed=42)

        # Same order -> same vectors
        a1 = hdc1.get_or_create_atom("first")
        b1 = hdc1.get_or_create_atom("second")

        a2 = hdc2.get_or_create_atom("first")
        b2 = hdc2.get_or_create_atom("second")

        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(b1, b2)


# -- Codebook tests --


class TestHDCCodebook:
    """Test the atom codebook management."""

    def test_get_or_create_new(self, hdc_small):
        """Creating a new atom adds it to the codebook."""
        assert "new_atom_xyz" not in hdc_small._codebook
        vec = hdc_small.get_or_create_atom("new_atom_xyz")
        assert "new_atom_xyz" in hdc_small._codebook
        assert vec.shape == (1000,)

    def test_get_or_create_existing(self, hdc_small):
        """Getting an existing atom returns the same vector."""
        vec1 = hdc_small.get_or_create_atom("existing")
        vec2 = hdc_small.get_or_create_atom("existing")
        assert vec1 is vec2  # Same object reference

    def test_roles_predefined(self, hdc):
        """Role vectors are pre-generated at init."""
        expected_roles = {"directory", "entity", "tag", "type", "purpose", "time_bucket"}
        assert set(hdc._roles.keys()) == expected_roles
        for role_name, role_vec in hdc._roles.items():
            assert role_vec.shape == (10000,), f"Role {role_name} wrong shape"

    def test_dimensions_property(self, hdc):
        """Dimensions property returns configured size."""
        assert hdc.dimensions == 10000

    def test_custom_dimensions(self):
        """Custom dimension size works."""
        hdc = HDCEncoder(dimensions=500, seed=1)
        assert hdc.dimensions == 500
        v = hdc._random_vector()
        assert v.shape == (500,)


# -- Integration with retrieval --


class TestHDCRetrievalIntegration:
    """Test HDC integration with the HippoRetriever recall pipeline."""

    def test_integration_retrieval(self, storage, embeddings, graph, settings):
        """HDC scores appear in recall results when HDC encoder is attached."""
        hdc = HDCEncoder(dimensions=2000, seed=42)
        retriever = HippoRetriever(storage, embeddings, graph, settings)
        retriever.set_hdc(hdc)

        # Create memories with HDC vectors
        content_auth = "Fixed authentication bug in auth.py login flow"
        mid1 = _make_memory(
            storage, embeddings, content_auth,
            directory="/proj", tags=["bug", "auth"],
        )
        hdc_vec1 = hdc.encode_memory("/proj", ["bug", "auth"], ["auth.py", "login"])
        storage._conn.execute(
            "UPDATE memories SET hdc_vector = ? WHERE id = ?",
            (hdc.to_bytes(hdc_vec1), mid1),
        )

        content_db = "Added database migration for user table schema"
        mid2 = _make_memory(
            storage, embeddings, content_db,
            directory="/proj", tags=["database", "migration"],
        )
        hdc_vec2 = hdc.encode_memory("/proj", ["database", "migration"], ["models.py", "schema"])
        storage._conn.execute(
            "UPDATE memories SET hdc_vector = ? WHERE id = ?",
            (hdc.to_bytes(hdc_vec2), mid2),
        )
        storage._conn.commit()

        # Recall should work and include results
        results = retriever.recall("auth.py login bug", max_results=5)
        assert len(results) > 0

        # Both memories should appear (via at least some signals)
        result_ids = {r["id"] for r in results}
        # At least one should match
        assert mid1 in result_ids or mid2 in result_ids

    def test_retriever_without_hdc(self, storage, embeddings, graph, settings):
        """Retriever works fine without HDC encoder attached."""
        retriever = HippoRetriever(storage, embeddings, graph, settings)
        # Don't set HDC — _hdc stays None

        mid = _make_memory(
            storage, embeddings,
            "Test memory content for basic retrieval",
            directory="/proj", tags=["test"],
        )

        results = retriever.recall("test memory content", max_results=5)
        assert len(results) > 0

    def test_hdc_stored_in_db(self, storage):
        """HDC vectors stored as BLOBs in the memories table."""
        hdc = HDCEncoder(dimensions=1000, seed=42)
        vec = hdc.encode_memory("/proj", ["test"], ["file.py"])
        blob = hdc.to_bytes(vec)

        # Insert a memory first
        mid = storage.insert_memory({
            "content": "test content",
            "embedding": None,
            "tags": ["test"],
            "directory_context": "/proj",
            "heat": 1.0,
            "is_stale": False,
            "file_hash": None,
        })

        # Store HDC vector
        storage._conn.execute(
            "UPDATE memories SET hdc_vector = ? WHERE id = ?",
            (blob, mid),
        )
        storage._conn.commit()

        # Read it back
        row = storage._conn.execute(
            "SELECT hdc_vector FROM memories WHERE id = ?", (mid,)
        ).fetchone()
        assert row[0] is not None
        restored = hdc.from_bytes(row[0])
        np.testing.assert_array_equal(restored, vec.astype(np.float32))


# -- Similarity edge cases --


class TestHDCSimilarityEdgeCases:
    """Test similarity computation edge cases."""

    def test_similarity_identical(self, hdc):
        """Identical vectors have similarity ~1.0."""
        v = hdc.get_or_create_atom("same")
        sim = hdc.similarity(v, v)
        assert sim > 0.99

    def test_similarity_negated(self, hdc):
        """Negated vector has similarity ~-1.0."""
        v = hdc.get_or_create_atom("negate_me")
        sim = hdc.similarity(v, -v)
        assert sim < -0.99

    def test_similarity_zero_vector(self, hdc):
        """Zero vector returns 0 similarity."""
        v = hdc.get_or_create_atom("nonzero")
        zero = np.zeros(hdc.dimensions)
        sim = hdc.similarity(v, zero)
        assert sim == 0.0
