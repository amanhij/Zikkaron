"""Tests for the sleep-time compute system."""

import random
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from zikkaron.config import Settings
from zikkaron.curation import MemoryCurator
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.knowledge_graph import KnowledgeGraph
from zikkaron.sleep_compute import SleepComputeEngine
from zikkaron.storage import StorageEngine
from zikkaron.thermodynamics import MemoryThermodynamics


@pytest.fixture
def storage(tmp_path):
    engine = StorageEngine(str(tmp_path / "test_sleep.db"))
    yield engine
    engine.close()


@pytest.fixture
def settings():
    return Settings(
        DB_PATH=":memory:",
        DREAM_REPLAY_PAIRS=10,
    )


@pytest.fixture
def mock_embeddings():
    engine = MagicMock(spec=EmbeddingEngine)
    engine.get_model_name.return_value = "all-MiniLM-L6-v2"
    engine.encode.return_value = np.ones(384, dtype=np.float32).tobytes()
    engine.encode_batch.return_value = [np.ones(384, dtype=np.float32).tobytes()]
    return engine


@pytest.fixture
def sleep_engine(storage, mock_embeddings, settings):
    graph = KnowledgeGraph(storage, settings)
    thermo = MemoryThermodynamics(storage, mock_embeddings, settings)
    curator = MemoryCurator(storage, mock_embeddings, thermo, settings)
    return SleepComputeEngine(
        storage, mock_embeddings, graph, curator, thermo, settings,
    )


def _make_embedding(value: float = 1.0) -> bytes:
    """Create a normalized embedding vector filled with the given value."""
    vec = np.full(384, value, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tobytes()


def _old_timestamp(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


class TestDreamReplay:
    def test_finds_connections(self, sleep_engine, storage, mock_embeddings):
        """Related but unconnected memories should be discovered."""
        vec = _make_embedding(1.0)
        storage.insert_memory({
            "content": "Python FastAPI web server implementation",
            "embedding": vec,
            "directory_context": "/project-a",
            "heat": 0.8,
        })
        storage.insert_memory({
            "content": "Flask HTTP API endpoint design",
            "embedding": vec,
            "directory_context": "/project-b",
            "heat": 0.7,
        })

        # Mock similarity above moderate threshold
        mock_embeddings.similarity.return_value = 0.5

        random.seed(42)
        stats = sleep_engine.dream_replay()

        assert stats["pairs_examined"] >= 1
        assert stats["connections_found"] >= 1

    def test_no_false_positives(self, sleep_engine, storage, mock_embeddings):
        """Unrelated memories should not get connected."""
        storage.insert_memory({
            "content": "Database optimization techniques",
            "embedding": _make_embedding(1.0),
            "directory_context": "/proj",
            "heat": 0.8,
        })
        storage.insert_memory({
            "content": "Cooking recipe for pasta",
            "embedding": _make_embedding(-1.0),
            "directory_context": "/other",
            "heat": 0.7,
        })

        # Mock similarity below threshold
        mock_embeddings.similarity.return_value = 0.1

        random.seed(42)
        stats = sleep_engine.dream_replay()

        assert stats["connections_found"] == 0
        assert stats["insights_generated"] == 0

    def test_strong_connection_generates_insight(self, sleep_engine, storage, mock_embeddings):
        """Strongly similar memories (>0.7) should generate dream insights."""
        vec = _make_embedding(1.0)
        storage.insert_memory({
            "content": "React component lifecycle hooks",
            "embedding": vec,
            "directory_context": "/frontend",
            "heat": 0.9,
        })
        storage.insert_memory({
            "content": "Vue.js component lifecycle methods",
            "embedding": vec,
            "directory_context": "/other-frontend",
            "heat": 0.8,
        })

        mock_embeddings.similarity.return_value = 0.8

        random.seed(42)
        stats = sleep_engine.dream_replay()

        assert stats["insights_generated"] >= 1

        # Verify dream insight memory was created
        dream_mems = storage._conn.execute(
            "SELECT * FROM memories WHERE content LIKE 'Dream connection:%'"
        ).fetchall()
        assert len(dream_mems) >= 1

    def test_skips_already_connected(self, sleep_engine, storage, mock_embeddings):
        """Memories already connected should be skipped."""
        vec = _make_embedding(1.0)
        mid_a = storage.insert_memory({
            "content": "Memory A",
            "embedding": vec,
            "directory_context": "/proj",
            "heat": 0.8,
        })
        mid_b = storage.insert_memory({
            "content": "Memory B",
            "embedding": vec,
            "directory_context": "/proj",
            "heat": 0.7,
        })

        # Pre-connect them
        eid_a = storage.insert_entity({"name": f"memory:{mid_a}", "type": "file"})
        eid_b = storage.insert_entity({"name": f"memory:{mid_b}", "type": "file"})
        storage.insert_relationship({
            "source_entity_id": eid_a,
            "target_entity_id": eid_b,
            "relationship_type": "co_occurrence",
        })

        mock_embeddings.similarity.return_value = 0.9

        random.seed(42)
        stats = sleep_engine.dream_replay()

        # Should have examined 0 pairs since the only pair is already connected
        assert stats["pairs_examined"] == 0
        assert stats["connections_found"] == 0


class TestCommunityDetection:
    def test_detects_communities(self, sleep_engine, storage):
        """Connected entity clusters should be detected as communities."""
        # Cluster 1: Python backend
        e1 = storage.insert_entity({"name": "fastapi", "type": "dependency"})
        e2 = storage.insert_entity({"name": "uvicorn", "type": "dependency"})
        e3 = storage.insert_entity({"name": "starlette", "type": "dependency"})

        # Cluster 2: Frontend
        e4 = storage.insert_entity({"name": "react", "type": "dependency"})
        e5 = storage.insert_entity({"name": "webpack", "type": "dependency"})

        # Intra-cluster edges for cluster 1
        storage.insert_relationship({
            "source_entity_id": e1, "target_entity_id": e2,
            "relationship_type": "co_occurrence",
        })
        storage.insert_relationship({
            "source_entity_id": e1, "target_entity_id": e3,
            "relationship_type": "co_occurrence",
        })
        storage.insert_relationship({
            "source_entity_id": e2, "target_entity_id": e3,
            "relationship_type": "co_occurrence",
        })

        # Intra-cluster edges for cluster 2
        storage.insert_relationship({
            "source_entity_id": e4, "target_entity_id": e5,
            "relationship_type": "co_occurrence",
        })

        communities = sleep_engine.detect_communities()

        assert len(communities) >= 1
        total_entities = sum(c["entity_count"] for c in communities)
        assert total_entities >= 4

    def test_no_communities_without_relationships(self, sleep_engine, storage):
        """No communities should be detected when entities have no relationships."""
        storage.insert_entity({"name": "isolated_a", "type": "function"})
        storage.insert_entity({"name": "isolated_b", "type": "function"})

        communities = sleep_engine.detect_communities()
        assert communities == []


class TestClusterSummarization:
    def test_clusters_get_summaries(self, sleep_engine, storage, mock_embeddings):
        """Clusters with > 3 members should get summaries and centroids."""
        cluster_id = storage.insert_cluster({
            "name": "test_cluster",
            "level": 1,
            "member_count": 5,
        })

        vec = _make_embedding(1.0)
        for i in range(5):
            mid = storage.insert_memory({
                "content": f"Memory about import fastapi and zikkaron/server.py part {i}",
                "embedding": vec,
                "directory_context": "/project",
                "heat": 0.7,
            })
            storage._conn.execute(
                "UPDATE memories SET cluster_id = ? WHERE id = ?",
                (cluster_id, mid),
            )
        storage._conn.commit()

        sleep_engine.generate_cluster_summaries()

        cluster = storage.get_cluster(cluster_id)
        assert cluster["summary"] != ""
        assert cluster["centroid_embedding"] is not None

    def test_small_clusters_skipped(self, sleep_engine, storage):
        """Clusters with <= 3 members should not be summarized."""
        cluster_id = storage.insert_cluster({
            "name": "small_cluster",
            "level": 1,
            "summary": "original",
            "member_count": 2,
        })

        sleep_engine.generate_cluster_summaries()

        cluster = storage.get_cluster(cluster_id)
        assert cluster["summary"] == "original"


class TestReembedStale:
    def test_stale_memories_reembedded(self, sleep_engine, storage, mock_embeddings):
        """Memories with wrong model version should get new embeddings."""
        old_vec = _make_embedding(0.5)
        new_vec = _make_embedding(1.0)

        mid1 = storage.insert_memory({
            "content": "Old embedding test",
            "embedding": old_vec,
            "directory_context": "/proj",
            "heat": 0.8,
            "embedding_model": "old-model-v1",
        })
        mid2 = storage.insert_memory({
            "content": "Another old embedding",
            "embedding": old_vec,
            "directory_context": "/proj",
            "heat": 0.7,
            "embedding_model": "old-model-v1",
        })

        mock_embeddings.encode_batch.return_value = [new_vec, new_vec]

        count = sleep_engine.reembed_stale()
        assert count == 2

        mem1 = storage.get_memory(mid1)
        assert mem1["embedding_model"] == "all-MiniLM-L6-v2"
        mem2 = storage.get_memory(mid2)
        assert mem2["embedding_model"] == "all-MiniLM-L6-v2"

    def test_current_model_not_reembedded(self, sleep_engine, storage, mock_embeddings):
        """Memories already using the current model should not be re-embedded."""
        vec = _make_embedding(1.0)
        storage.insert_memory({
            "content": "Current model memory",
            "embedding": vec,
            "directory_context": "/proj",
            "heat": 0.8,
            "embedding_model": "all-MiniLM-L6-v2",
        })

        count = sleep_engine.reembed_stale()
        assert count == 0


class TestMemoryCompression:
    def test_old_long_memories_compressed(self, sleep_engine, storage, mock_embeddings):
        """Old memories with long content should be compressed."""
        long_content = (
            "This is a long memory about Python development. "
            "We worked on zikkaron/server.py and fixed the API endpoint. "
            + "This is filler content that does not contain entities. " * 30
            + "The final fix was in zikkaron/storage.py which resolved the issue."
        )
        old_time = _old_timestamp(60)
        mid = storage.insert_memory({
            "content": long_content,
            "embedding": _make_embedding(1.0),
            "directory_context": "/proj",
            "heat": 0.5,
            "created_at": old_time,
            "last_accessed": old_time,
        })

        count = sleep_engine.compress_old_memories(days_threshold=30)
        assert count >= 1

        mem = storage.get_memory(mid)
        assert len(mem["content"]) < len(long_content)
        assert mem["compressed"] is True

    def test_recent_memories_not_compressed(self, sleep_engine, storage):
        """Recent memories should not be compressed regardless of length."""
        mid = storage.insert_memory({
            "content": "x " * 600,  # long but recent
            "embedding": _make_embedding(1.0),
            "directory_context": "/proj",
            "heat": 0.8,
        })

        count = sleep_engine.compress_old_memories(days_threshold=30)
        assert count == 0

    def test_short_memories_not_compressed(self, sleep_engine, storage):
        """Short old memories should not be compressed."""
        old_time = _old_timestamp(60)
        storage.insert_memory({
            "content": "Short memory",
            "embedding": _make_embedding(1.0),
            "directory_context": "/proj",
            "heat": 0.5,
            "created_at": old_time,
            "last_accessed": old_time,
        })

        count = sleep_engine.compress_old_memories(days_threshold=30)
        assert count == 0


class TestFullSleepCycle:
    def test_all_phases_run(self, sleep_engine, storage, mock_embeddings):
        """Full sleep cycle should execute all phases without errors."""
        vec = _make_embedding(1.0)
        for i in range(3):
            storage.insert_memory({
                "content": f"Memory {i} about testing sleep cycle",
                "embedding": vec,
                "directory_context": "/proj",
                "heat": 0.7,
                "embedding_model": "all-MiniLM-L6-v2",
            })

        mock_embeddings.similarity.return_value = 0.3  # below threshold

        stats = sleep_engine.run_sleep_cycle()

        assert "dream_replay" in stats
        assert "communities" in stats
        assert "cluster_summaries_generated" in stats
        assert "reembedded" in stats
        assert "compressed" in stats

    def test_sleep_cycle_with_empty_db(self, sleep_engine):
        """Sleep cycle should handle an empty database gracefully."""
        stats = sleep_engine.run_sleep_cycle()

        assert stats["dream_replay"]["pairs_examined"] == 0
        assert stats["communities"] == []
        assert stats["reembedded"] == 0
        assert stats["compressed"] == 0
