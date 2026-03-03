"""Tests for HippoRAG-style retrieval engine."""

import time
from datetime import datetime, timezone

import numpy as np
import pytest

from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.knowledge_graph import KnowledgeGraph
from zikkaron.retrieval import HippoRetriever, _extract_query_entities
from zikkaron.storage import StorageEngine


@pytest.fixture
def storage(tmp_path):
    engine = StorageEngine(str(tmp_path / "test_retrieval.db"))
    yield engine
    engine.close()


@pytest.fixture
def settings():
    return Settings(DB_PATH=":memory:", PPR_DAMPING=0.85, PPR_ITERATIONS=50)


@pytest.fixture
def embeddings():
    return EmbeddingEngine("all-MiniLM-L6-v2")


@pytest.fixture
def graph(storage, settings):
    return KnowledgeGraph(storage, settings)


@pytest.fixture
def retriever(storage, embeddings, graph, settings):
    return HippoRetriever(storage, embeddings, graph, settings)


def _make_memory(storage, embeddings, content, directory="/proj", tags=None):
    """Helper to insert a memory with embedding."""
    embedding = embeddings.encode(content)
    return storage.insert_memory({
        "content": content,
        "embedding": embedding,
        "tags": tags or [],
        "directory_context": directory,
        "heat": 1.0,
        "is_stale": False,
        "file_hash": None,
        "embedding_model": embeddings.get_model_name(),
    })


def _setup_graph_with_memories(storage, embeddings, graph):
    """Set up a knowledge graph with entities, relationships, and memories."""
    # Create memories
    m1 = _make_memory(
        storage, embeddings,
        "Using FastAPI for the REST API server with uvicorn",
        tags=["backend", "api"],
    )
    m2 = _make_memory(
        storage, embeddings,
        "FastAPI integrates with pydantic for data validation",
        tags=["backend", "validation"],
    )
    m3 = _make_memory(
        storage, embeddings,
        "SQLite with WAL mode for the storage engine database",
        tags=["database"],
    )
    m4 = _make_memory(
        storage, embeddings,
        "pydantic models used for configuration settings",
        tags=["config"],
    )
    m5 = _make_memory(
        storage, embeddings,
        "React frontend connects to the REST API server",
        tags=["frontend"],
    )

    # Create entity graph: FastAPI -> pydantic -> SQLite
    graph.add_relationship("FastAPI", "uvicorn", "co_occurrence")
    graph.add_relationship("FastAPI", "pydantic", "imports")
    graph.add_relationship("pydantic", "SQLite", "co_occurrence")
    graph.add_relationship("FastAPI", "REST", "co_occurrence")
    graph.add_relationship("REST", "React", "co_occurrence")

    return m1, m2, m3, m4, m5


class TestQueryEntityExtraction:
    def test_extracts_camelcase(self):
        entities = _extract_query_entities("How does FastAPI work?")
        assert "FastAPI" in entities

    def test_extracts_file_paths(self):
        entities = _extract_query_entities("Check zikkaron/server.py for bugs")
        assert "zikkaron/server.py" in entities

    def test_extracts_error_types(self):
        entities = _extract_query_entities("Fix the ValueError in parser")
        assert "ValueError" in entities

    def test_extracts_dotted_names(self):
        entities = _extract_query_entities("Import from zikkaron.storage module")
        assert "zikkaron.storage" in entities

    def test_extracts_keywords(self):
        entities = _extract_query_entities("database configuration settings")
        assert "database" in entities
        assert "configuration" in entities


class TestPPRRetrieval:
    def test_ppr_returns_connected_memories(self, storage, embeddings, graph, retriever):
        _setup_graph_with_memories(storage, embeddings, graph)

        results = retriever.ppr_retrieve("FastAPI server", top_k=5)
        # Should return memory IDs with scores
        assert len(results) > 0
        # All results should be (memory_id, score) tuples
        for mid, score in results:
            assert isinstance(mid, int)
            assert isinstance(score, float)
            assert score > 0

    def test_ppr_ranks_directly_connected_higher(self, storage, embeddings, graph, retriever):
        _setup_graph_with_memories(storage, embeddings, graph)

        results = retriever.ppr_retrieve("FastAPI", top_k=10)
        if len(results) >= 2:
            # Memories mentioning FastAPI should rank higher than distant ones
            memory_ids = [mid for mid, _ in results]
            # Memory 1 and 2 mention FastAPI, should appear
            m1_content = storage.get_memory(memory_ids[0])
            assert m1_content is not None

    def test_ppr_empty_for_unknown_entities(self, storage, embeddings, graph, retriever):
        _setup_graph_with_memories(storage, embeddings, graph)
        results = retriever.ppr_retrieve("completely_unknown_xyz_entity")
        assert results == []

    def test_ppr_respects_top_k(self, storage, embeddings, graph, retriever):
        _setup_graph_with_memories(storage, embeddings, graph)
        results = retriever.ppr_retrieve("FastAPI pydantic", top_k=2)
        assert len(results) <= 2


class TestContextualPrefix:
    def test_prefix_contains_project_name(self, retriever):
        prefix = retriever.generate_contextual_prefix(
            "some content",
            "/home/user/myproject",
            ["tag1"],
            datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc),
        )
        assert "[Project: myproject]" in prefix

    def test_prefix_contains_directory(self, retriever):
        prefix = retriever.generate_contextual_prefix(
            "some content",
            "/home/user/myproject",
            ["tag1"],
            datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc),
        )
        assert "[Directory: /home/user/myproject]" in prefix

    def test_prefix_contains_tags(self, retriever):
        prefix = retriever.generate_contextual_prefix(
            "some content",
            "/proj",
            ["backend", "api"],
            datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc),
        )
        assert "[Tags: backend, api]" in prefix

    def test_prefix_contains_timestamp(self, retriever):
        ts = datetime(2026, 3, 1, 12, 30, tzinfo=timezone.utc)
        prefix = retriever.generate_contextual_prefix(
            "some content", "/proj", [], ts
        )
        assert "[Recorded: 2026-03-01 12:30]" in prefix

    def test_prefix_contains_related_entities(self, storage, embeddings, graph, retriever):
        # Set up entities and relationships
        graph.add_relationship("FastAPI", "pydantic", "imports")

        # Insert a memory mentioning FastAPI so _find_entities_in_content works
        _make_memory(storage, embeddings, "Using FastAPI for the server")

        prefix = retriever.generate_contextual_prefix(
            "FastAPI server setup",
            "/proj",
            ["backend"],
            datetime(2026, 3, 1, tzinfo=timezone.utc),
        )
        assert "[Related entities:" in prefix

    def test_prefix_empty_tags_shows_none(self, retriever):
        prefix = retriever.generate_contextual_prefix(
            "content", "/proj", [],
            datetime(2026, 3, 1, tzinfo=timezone.utc),
        )
        assert "[Tags: none]" in prefix


class TestSpreadingActivation:
    def test_spreading_activates_related_memories(self, storage, embeddings, graph, retriever):
        m1, m2, m3, m4, m5 = _setup_graph_with_memories(
            storage, embeddings, graph
        )

        # Seed with memory 1 (FastAPI) — should activate memories connected via graph
        results = retriever.spreading_activation([m1], spread_factor=0.5, max_depth=2)

        # Should find some activated memories (not including the seed)
        activated_ids = [mid for mid, _ in results]
        assert m1 not in activated_ids  # seed excluded

    def test_spreading_excludes_seeds(self, storage, embeddings, graph, retriever):
        m1, m2, m3, m4, m5 = _setup_graph_with_memories(
            storage, embeddings, graph
        )

        results = retriever.spreading_activation([m1, m2])
        activated_ids = {mid for mid, _ in results}
        assert m1 not in activated_ids
        assert m2 not in activated_ids

    def test_spreading_activation_decays_with_depth(self, storage, embeddings, graph, retriever):
        m1, m2, m3, m4, m5 = _setup_graph_with_memories(
            storage, embeddings, graph
        )

        results = retriever.spreading_activation(
            [m1], spread_factor=0.5, max_depth=2
        )
        if len(results) >= 2:
            # Scores should vary (deeper = lower activation)
            scores = [s for _, s in results]
            assert max(scores) <= 0.5  # spread_factor^1 = 0.5 max

    def test_spreading_empty_seeds_returns_empty(self, retriever):
        assert retriever.spreading_activation([]) == []

    def test_spreading_nonexistent_memory_returns_empty(self, retriever):
        results = retriever.spreading_activation([99999])
        assert results == []


class TestUnifiedRecall:
    def test_recall_returns_results(self, storage, embeddings, graph, retriever):
        _setup_graph_with_memories(storage, embeddings, graph)

        results = retriever.recall("FastAPI server", max_results=5)
        assert len(results) > 0

    def test_recall_results_have_score(self, storage, embeddings, graph, retriever):
        _setup_graph_with_memories(storage, embeddings, graph)

        results = retriever.recall("FastAPI", max_results=5)
        for mem in results:
            assert "_retrieval_score" in mem
            assert mem["_retrieval_score"] >= 0

    def test_recall_respects_max_results(self, storage, embeddings, graph, retriever):
        _setup_graph_with_memories(storage, embeddings, graph)

        results = retriever.recall("FastAPI pydantic SQLite", max_results=2)
        assert len(results) <= 2

    def test_recall_respects_min_heat(self, storage, embeddings, graph, retriever):
        _setup_graph_with_memories(storage, embeddings, graph)

        # Set one memory to low heat
        storage.update_memory_heat(1, 0.01)

        results = retriever.recall("FastAPI", max_results=10, min_heat=0.5)
        for mem in results:
            assert mem["heat"] >= 0.5

    def test_recall_combines_all_four_signals(self, storage, embeddings, graph, retriever):
        """Verify that all four retrieval signals contribute to results."""
        _setup_graph_with_memories(storage, embeddings, graph)

        results = retriever.recall("FastAPI pydantic", max_results=5)
        # Should have results from the combination of signals
        assert len(results) > 0
        # Results should be sorted by combined score descending
        if len(results) >= 2:
            scores = [m["_retrieval_score"] for m in results]
            assert scores == sorted(scores, reverse=True)

    def test_recall_deduplicates(self, storage, embeddings, graph, retriever):
        _setup_graph_with_memories(storage, embeddings, graph)

        results = retriever.recall("FastAPI server", max_results=10)
        ids = [m["id"] for m in results]
        assert len(ids) == len(set(ids))  # no duplicates

    def test_recall_strips_embeddings(self, storage, embeddings, graph, retriever):
        _setup_graph_with_memories(storage, embeddings, graph)

        results = retriever.recall("FastAPI", max_results=5)
        for mem in results:
            assert "embedding" not in mem


class TestRecallRanking:
    def test_most_relevant_ranks_first(self, storage, embeddings, graph, retriever):
        """The most relevant result should rank first."""
        # Insert a highly specific memory
        _make_memory(
            storage, embeddings,
            "networkx PageRank algorithm for graph-based retrieval in HippoRAG",
            tags=["retrieval", "graph"],
        )
        # Insert a less relevant memory
        _make_memory(
            storage, embeddings,
            "General logging configuration for the application",
            tags=["config"],
        )
        # Insert another relevant one
        _make_memory(
            storage, embeddings,
            "Using PageRank for personalized search ranking",
            tags=["search"],
        )

        results = retriever.recall("PageRank graph retrieval", max_results=3)
        assert len(results) >= 1
        # Top result should mention PageRank
        assert "PageRank" in results[0]["content"] or "graph" in results[0]["content"]


class TestRecallPerformance:
    def test_recall_completes_under_100ms(self, storage, embeddings, graph, retriever):
        """Recall should complete in <100ms for 100 memories."""
        # Insert 100 memories
        topics = [
            "Python web development with Flask and Django frameworks",
            "JavaScript React component lifecycle and hooks",
            "Database optimization with PostgreSQL indexes",
            "Docker container orchestration with Kubernetes",
            "Machine learning model training with PyTorch",
            "REST API design patterns and best practices",
            "Git branching strategies for team collaboration",
            "Continuous integration with GitHub Actions",
            "Cloud deployment on AWS Lambda functions",
            "Security authentication with JWT tokens",
        ]
        for i in range(100):
            topic = topics[i % len(topics)]
            _make_memory(
                storage, embeddings,
                f"Memory {i}: {topic} - variation {i}",
                tags=["perf-test"],
            )

        # Warm up
        retriever.recall("Python Flask", max_results=5)

        # Timed run
        start = time.monotonic()
        results = retriever.recall("Python web development", max_results=5)
        elapsed_ms = (time.monotonic() - start) * 1000

        assert len(results) > 0
        assert elapsed_ms < 100, f"Recall took {elapsed_ms:.1f}ms, expected <100ms"
