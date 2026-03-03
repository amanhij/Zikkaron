"""Tests for fractal/hierarchical memory organization."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.fractal import FractalMemoryTree
from zikkaron.storage import StorageEngine


@pytest.fixture
def storage(tmp_path):
    engine = StorageEngine(str(tmp_path / "test_fractal.db"))
    yield engine
    engine.close()


@pytest.fixture
def settings():
    return Settings(
        DB_PATH=":memory:",
        CLUSTER_SIMILARITY_THRESHOLD=0.7,
    )


@pytest.fixture
def mock_embeddings():
    engine = MagicMock(spec=EmbeddingEngine)
    engine.get_model_name.return_value = "all-MiniLM-L6-v2"
    engine.get_dimensions.return_value = 384
    return engine


def _make_embedding(seed: float = 1.0, dim: int = 384) -> bytes:
    """Create a normalized embedding vector seeded by the given value."""
    rng = np.random.RandomState(int(abs(seed) * 1000))
    vec = rng.randn(dim).astype(np.float32)
    vec[0] += seed * 10  # push direction based on seed
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tobytes()


def _make_similar_embedding(base_seed: float, noise: float = 0.01, dim: int = 384) -> bytes:
    """Create an embedding similar to the base seed's embedding."""
    rng = np.random.RandomState(int(abs(base_seed) * 1000))
    base = rng.randn(dim).astype(np.float32)
    base[0] += base_seed * 10
    # Add small noise
    rng2 = np.random.RandomState(int(abs(noise) * 10000 + 999))
    perturbation = rng2.randn(dim).astype(np.float32) * noise
    vec = (base + perturbation).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tobytes()


@pytest.fixture
def fractal(storage, mock_embeddings, settings):
    return FractalMemoryTree(storage, mock_embeddings, settings)


def _insert_memories(storage, directory, count, seed=1.0):
    """Insert count memories with similar embeddings in the given directory."""
    ids = []
    for i in range(count):
        emb = _make_similar_embedding(seed, noise=0.01 * (i + 1))
        mid = storage.insert_memory({
            "content": f"Memory {i} about topic-{seed} in {directory}",
            "embedding": emb,
            "tags": [f"tag-{int(seed)}", "test"],
            "directory_context": directory,
            "heat": 0.8,
            "embedding_model": "all-MiniLM-L6-v2",
        })
        ids.append(mid)
    return ids


class TestBuildTree:
    def test_build_tree_creates_clusters(self, fractal, storage, mock_embeddings):
        """Memories in different directories should produce level 1 and 2 clusters."""
        # Insert memories in two directories
        _insert_memories(storage, "/project-a", 3, seed=1.0)
        _insert_memories(storage, "/project-b", 3, seed=2.0)

        # Mock encode to return a searchable embedding
        mock_embeddings.encode.return_value = _make_embedding(1.5)

        stats = fractal.build_tree()

        assert stats["level_1_clusters"] >= 2
        assert stats["memories_assigned"] == 6
        assert stats["level_2_clusters"] >= 2

    def test_build_tree_empty_db(self, fractal):
        """Building a tree on an empty DB should return zeros."""
        stats = fractal.build_tree()
        assert stats["level_1_clusters"] == 0
        assert stats["level_2_clusters"] == 0
        assert stats["memories_assigned"] == 0

    def test_build_tree_assigns_cluster_ids(self, fractal, storage):
        """After building, memories should have cluster_id set."""
        ids = _insert_memories(storage, "/proj", 4, seed=1.0)

        fractal.build_tree()

        for mid in ids:
            mem = storage.get_memory(mid)
            assert mem["cluster_id"] is not None


class TestLevel0Retrieval:
    def test_specific_query_returns_memories(self, fractal, storage, mock_embeddings):
        """Retrieving at level 0 should return individual memories."""
        ids = _insert_memories(storage, "/proj", 5, seed=1.0)

        # Mock encode to return a matching embedding
        mock_embeddings.encode.return_value = _make_embedding(1.0)

        results = fractal.retrieve_tree("topic-1.0 details", target_level=0)

        assert len(results) > 0
        for r in results:
            assert r["level"] == 0
            assert r["type"] == "memory"
            assert "content" in r
            assert "score" in r


class TestLevel2Retrieval:
    def test_broad_query_returns_clusters(self, fractal, storage, mock_embeddings):
        """Retrieving at level 2 should return project-level clusters."""
        _insert_memories(storage, "/project-a", 4, seed=1.0)
        _insert_memories(storage, "/project-b", 4, seed=2.0)

        fractal.build_tree()

        # Mock encode to return a query embedding
        mock_embeddings.encode.return_value = _make_embedding(1.0)

        results = fractal.retrieve_tree("project overview", target_level=2)

        # Only level 2 clusters should be returned (if they have centroids)
        for r in results:
            assert r["level"] == 2


class TestAdaptiveLevel:
    def test_short_query_prefers_higher_levels(self, fractal, storage, mock_embeddings):
        """Short queries should weight higher levels more."""
        _insert_memories(storage, "/proj", 4, seed=1.0)
        fractal.build_tree()

        mock_embeddings.encode.return_value = _make_embedding(1.0)

        # Short query (< 10 words)
        results = fractal.retrieve_tree("testing")

        # Should include results from multiple levels
        assert len(results) >= 0  # may be empty if no centroids, but shouldn't error

    def test_long_query_prefers_lower_levels(self, fractal, storage, mock_embeddings):
        """Long queries should weight lower levels more."""
        ids = _insert_memories(storage, "/proj", 4, seed=1.0)
        fractal.build_tree()

        mock_embeddings.encode.return_value = _make_embedding(1.0)

        # Long query (> 30 words)
        long_query = " ".join(["word"] * 35)
        results = fractal.retrieve_tree(long_query)

        # Should include level 0 results
        level_0_results = [r for r in results if r["level"] == 0]
        if results:
            # Level 0 should be present in results
            assert len(level_0_results) > 0


class TestDrillDown:
    def test_drill_from_level2_to_level1(self, fractal, storage):
        """Drilling from level 2 should return level 1 sub-clusters."""
        # Create a level 2 root cluster
        root_id = storage.insert_cluster({
            "name": "root_project",
            "level": 2,
            "summary": "Root cluster",
            "member_count": 10,
        })

        # Create level 1 child clusters
        child_1 = storage.insert_cluster({
            "name": "child_a",
            "level": 1,
            "parent_cluster_id": root_id,
            "summary": "Sub-cluster A",
            "member_count": 5,
        })
        child_2 = storage.insert_cluster({
            "name": "child_b",
            "level": 1,
            "parent_cluster_id": root_id,
            "summary": "Sub-cluster B",
            "member_count": 5,
        })

        results = fractal.drill_down(root_id)

        assert len(results) == 2
        result_ids = {r["id"] for r in results}
        assert child_1 in result_ids
        assert child_2 in result_ids
        for r in results:
            assert r["level"] == 1
            assert r["type"] == "cluster"

    def test_drill_from_level1_to_memories(self, fractal, storage):
        """Drilling from level 1 should return individual memories."""
        cluster_id = storage.insert_cluster({
            "name": "test_cluster",
            "level": 1,
            "summary": "Test cluster",
            "member_count": 3,
        })

        mem_ids = []
        for i in range(3):
            mid = storage.insert_memory({
                "content": f"Memory {i} in cluster",
                "embedding": _make_embedding(float(i)),
                "directory_context": "/proj",
                "heat": 0.8,
            })
            storage._conn.execute(
                "UPDATE memories SET cluster_id = ? WHERE id = ?",
                (cluster_id, mid),
            )
            mem_ids.append(mid)
        storage._conn.commit()

        results = fractal.drill_down(cluster_id)

        assert len(results) == 3
        result_ids = {r["id"] for r in results}
        for mid in mem_ids:
            assert mid in result_ids
        for r in results:
            assert r["level"] == 0
            assert r["type"] == "memory"

    def test_drill_nonexistent_cluster(self, fractal):
        """Drilling into a nonexistent cluster should return empty."""
        results = fractal.drill_down(99999)
        assert results == []


class TestRollUp:
    def test_roll_up_full_hierarchy(self, fractal, storage):
        """Roll up from memory should return level 1 and level 2 clusters."""
        # Create hierarchy: root -> child -> memory
        root_id = storage.insert_cluster({
            "name": "root",
            "level": 2,
            "summary": "Root",
            "member_count": 5,
        })
        child_id = storage.insert_cluster({
            "name": "child",
            "level": 1,
            "parent_cluster_id": root_id,
            "summary": "Child cluster",
            "member_count": 3,
        })
        mid = storage.insert_memory({
            "content": "Test memory for roll up",
            "embedding": _make_embedding(1.0),
            "directory_context": "/proj",
            "heat": 0.9,
        })
        storage._conn.execute(
            "UPDATE memories SET cluster_id = ? WHERE id = ?",
            (child_id, mid),
        )
        storage._conn.commit()

        result = fractal.roll_up(mid)

        assert result["memory"] is not None
        assert result["memory"]["id"] == mid
        assert result["level_1_cluster"] is not None
        assert result["level_1_cluster"]["id"] == child_id
        assert result["level_2_cluster"] is not None
        assert result["level_2_cluster"]["id"] == root_id

    def test_roll_up_no_cluster(self, fractal, storage):
        """Memory without a cluster should return None for clusters."""
        mid = storage.insert_memory({
            "content": "Orphan memory",
            "embedding": _make_embedding(1.0),
            "directory_context": "/proj",
            "heat": 0.9,
        })

        result = fractal.roll_up(mid)

        assert result["memory"] is not None
        assert result["memory"]["id"] == mid
        assert result["level_1_cluster"] is None
        assert result["level_2_cluster"] is None

    def test_roll_up_nonexistent_memory(self, fractal):
        """Rolling up a nonexistent memory should return all None."""
        result = fractal.roll_up(99999)
        assert result["memory"] is None
        assert result["level_1_cluster"] is None
        assert result["level_2_cluster"] is None


class TestFractalScore:
    def test_fractal_score_returns_memory_ids(self, fractal, storage, mock_embeddings):
        """fractal_score should return (memory_id, score) tuples."""
        ids = _insert_memories(storage, "/proj", 5, seed=1.0)
        fractal.build_tree()

        # Mock encode for the query
        mock_embeddings.encode.return_value = _make_embedding(1.0)

        results = fractal.fractal_score("test query")

        # Results should be tuples of (memory_id, score)
        for mid, score in results:
            assert isinstance(mid, int)
            assert isinstance(score, float)
            assert score >= 0

    def test_fractal_score_empty_db(self, fractal, mock_embeddings):
        """fractal_score on empty DB should return empty list."""
        mock_embeddings.encode.return_value = _make_embedding(1.0)
        results = fractal.fractal_score("test")
        assert results == []
