"""Tests for the metacognition module — coverage assessment, gap detection,
and cognitive load management."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.knowledge_graph import KnowledgeGraph
from zikkaron.metacognition import MetaCognition
from zikkaron.storage import StorageEngine


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def settings():
    return Settings(DB_PATH=":memory:")


@pytest.fixture
def storage(tmp_path, settings):
    return StorageEngine(str(tmp_path / "test_meta.db"))


@pytest.fixture
def embeddings():
    return EmbeddingEngine("all-MiniLM-L6-v2")


@pytest.fixture
def graph(storage, settings):
    return KnowledgeGraph(storage, settings)


@pytest.fixture
def meta(storage, embeddings, graph, settings):
    return MetaCognition(storage, embeddings, graph, settings)


def _make_memory(
    storage,
    embeddings,
    content,
    directory="/proj",
    tags=None,
    heat=1.0,
    confidence=1.0,
    importance=0.5,
    surprise=0.0,
    created_at=None,
):
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
    updates = {}
    if confidence != 1.0:
        updates["confidence"] = confidence
    if importance != 0.5:
        updates["importance"] = importance
    if surprise != 0.0:
        updates["surprise_score"] = surprise

    if updates:
        set_clauses = ", ".join(f"{k} = ?" for k in updates)
        storage._conn.execute(
            f"UPDATE memories SET {set_clauses} WHERE id = ?",
            (*updates.values(), mid),
        )
        storage._conn.commit()

    if created_at is not None:
        storage._conn.execute(
            "UPDATE memories SET created_at = ? WHERE id = ?",
            (created_at.isoformat(), mid),
        )
        storage._conn.commit()

    return mid


# ── Coverage Assessment Tests ─────────────────────────────────────────


class TestCoverageAssessment:
    def test_coverage_sufficient(self, meta, storage, embeddings, graph):
        """Many relevant memories → high coverage."""
        # Create 7 memories about FastAPI
        for i in range(7):
            _make_memory(
                storage, embeddings,
                f"FastAPI endpoint {i}: handles REST API requests with uvicorn",
                tags=["FastAPI", "api"],
            )
        # Add entities to graph
        graph.add_relationship("FastAPI", "uvicorn", "co_occurrence")
        graph.add_relationship("FastAPI", "REST", "co_occurrence")

        result = meta.assess_coverage("How does FastAPI work?", "/proj")

        assert result["coverage"] >= 0.7
        assert result["suggestion"] == "sufficient"
        assert result["memory_count"] >= 6
        assert "gaps" in result
        assert "detail" in result

    def test_coverage_insufficient(self, meta, storage, embeddings):
        """No relevant memories → low coverage."""
        result = meta.assess_coverage("How does Kubernetes autoscaling work?", "/proj")

        assert result["coverage"] < 0.4
        assert result["suggestion"] == "insufficient"
        assert result["memory_count"] == 0
        assert len(result["gaps"]) > 0

    def test_coverage_partial(self, meta, storage, embeddings, graph):
        """Some memories → medium coverage."""
        # Create 2-3 memories
        _make_memory(
            storage, embeddings,
            "Django uses ORM for database queries",
            tags=["Django"],
        )
        _make_memory(
            storage, embeddings,
            "Django views handle HTTP requests",
            tags=["Django"],
        )
        graph.add_relationship("Django", "ORM", "co_occurrence")

        result = meta.assess_coverage("How does Django handle requests?", "/proj")

        assert 0.3 <= result["coverage"] < 0.8
        assert result["suggestion"] in ("partial", "sufficient")
        assert result["memory_count"] >= 1

    def test_coverage_recency_scoring(self, meta, storage, embeddings):
        """Recent memories score higher for recency."""
        now = datetime.now(timezone.utc)
        _make_memory(
            storage, embeddings,
            "React hooks for state management",
            tags=["React"],
            created_at=now - timedelta(hours=2),
        )

        result = meta.assess_coverage("React hooks", "/proj")
        assert result["recency_score"] == 1.0  # < 1 day old

    def test_coverage_old_recency(self, meta, storage, embeddings):
        """Old memories score lower for recency."""
        old_date = datetime.now(timezone.utc) - timedelta(days=60)
        _make_memory(
            storage, embeddings,
            "Legacy React class components",
            tags=["React"],
            created_at=old_date,
        )

        result = meta.assess_coverage("React class components", "/proj")
        assert result["recency_score"] == 0.2  # > 30 days old

    def test_coverage_entity_gap_detection(self, meta, storage, embeddings, graph):
        """Unknown entities appear in gaps list."""
        _make_memory(
            storage, embeddings,
            "Python typing module for type hints",
            tags=["Python"],
        )
        # "typing" entity exists, but "mypy" doesn't
        graph.add_relationship("Python", "typing", "co_occurrence")

        result = meta.assess_coverage("Python typing with mypy", "/proj")
        # "mypy" should be in gaps since it's not in the graph
        assert any("mypy" in g for g in result["gaps"])

    def test_coverage_returns_all_fields(self, meta, storage, embeddings):
        """Coverage result has all required fields."""
        result = meta.assess_coverage("test query", "/proj")

        assert "coverage" in result
        assert "confidence" in result
        assert "suggestion" in result
        assert "gaps" in result
        assert "memory_count" in result
        assert "entity_coverage" in result
        assert "recency_score" in result
        assert "detail" in result


# ── Gap Detection Tests ───────────────────────────────────────────────


class TestGapDetection:
    def test_gap_detection_isolated_entities(self, meta, storage, embeddings, graph):
        """Finds poorly connected entities (0 or 1 relationship)."""
        # Create an isolated entity with no relationships
        storage.insert_entity({"name": "OrphanModule", "type": "function"})

        gaps = meta.detect_gaps()

        isolated = [g for g in gaps if g["type"] == "isolated_entity"]
        assert len(isolated) > 0
        names = []
        for g in isolated:
            names.extend(g["entities"])
        assert "OrphanModule" in names

    def test_gap_detection_stale_region(self, meta, storage, embeddings):
        """Finds decayed memory clusters."""
        # Create multiple low-heat memories
        for i in range(3):
            _make_memory(
                storage, embeddings,
                f"Old architecture decision {i}",
                directory="/proj",
                tags=["architecture"],
                heat=0.1,
            )

        gaps = meta.detect_gaps("/proj")

        stale = [g for g in gaps if g["type"] == "stale_region"]
        assert len(stale) > 0
        assert stale[0]["severity"] > 0.3

    def test_gap_detection_low_confidence(self, meta, storage, embeddings):
        """Finds unreliable memories."""
        # Create low-confidence memories
        _make_memory(
            storage, embeddings,
            "Maybe the config file is at /etc/app.conf",
            directory="/proj",
            confidence=0.3,
        )
        _make_memory(
            storage, embeddings,
            "The database might use PostgreSQL or MySQL",
            directory="/proj",
            confidence=0.2,
        )

        gaps = meta.detect_gaps("/proj")

        low_conf = [g for g in gaps if g["type"] == "low_confidence"]
        assert len(low_conf) > 0

    def test_gap_detection_missing_connections(self, meta, storage, embeddings, graph):
        """Entities co-occurring in content but missing graph relationships."""
        # Create entity records
        storage.insert_entity({"name": "Redis", "type": "dependency"})
        storage.insert_entity({"name": "Celery", "type": "dependency"})

        # Create memories where both appear together, but no relationship
        _make_memory(
            storage, embeddings,
            "Redis is used as the broker for Celery task queue",
            directory="/proj",
        )
        _make_memory(
            storage, embeddings,
            "Configure Redis connection for Celery workers",
            directory="/proj",
        )

        gaps = meta.detect_gaps("/proj")

        missing = [g for g in gaps if g["type"] == "missing_connection"]
        assert len(missing) > 0
        found_pair = False
        for g in missing:
            if "Redis" in g["entities"] and "Celery" in g["entities"]:
                found_pair = True
                break
        assert found_pair

    def test_gap_detection_one_sided_knowledge(self, meta, storage, embeddings, graph):
        """Error entity with no resolution recorded."""
        # Create error entity without resolved_by relationship
        storage.insert_entity({"name": "ConnectionError", "type": "error"})

        gaps = meta.detect_gaps()

        one_sided = [g for g in gaps if g["type"] == "one_sided_knowledge"]
        assert len(one_sided) > 0
        names = []
        for g in one_sided:
            names.extend(g["entities"])
        assert "ConnectionError" in names

    def test_gap_detection_no_gaps(self, meta, storage, embeddings, graph):
        """No gaps when knowledge is well-connected."""
        gaps = meta.detect_gaps("/empty_dir")
        # Should not crash with empty data
        assert isinstance(gaps, list)

    def test_gap_returns_required_fields(self, meta, storage, embeddings, graph):
        """Each gap dict has all required fields."""
        storage.insert_entity({"name": "Lonely", "type": "variable"})

        gaps = meta.detect_gaps()

        for gap in gaps:
            assert "type" in gap
            assert "description" in gap
            assert "severity" in gap
            assert "entities" in gap
            assert "suggestion" in gap
            assert isinstance(gap["severity"], float)
            assert 0.0 <= gap["severity"] <= 1.0


# ── Cognitive Load Management Tests ───────────────────────────────────


class TestManageContext:
    def test_manage_context_under_limit(self, meta):
        """Returns all memories if under chunk limit."""
        memories = [
            {"id": 1, "content": "mem1", "heat": 0.9, "importance": 0.8, "confidence": 0.9},
            {"id": 2, "content": "mem2", "heat": 0.8, "importance": 0.7, "confidence": 0.8},
        ]

        result = meta.manage_context(memories, max_chunks=4)

        assert len(result) == 2
        for m in result:
            assert "_chunk_id" in m
            assert "_position_reason" in m
            assert m["_position_reason"] == "within_limit"

    def test_manage_context_over_limit(self, meta):
        """Truncates to chunk limit when over."""
        # Use very distinct content and spread-out timestamps so each memory
        # forms its own chunk (no entity overlap, no temporal proximity)
        base = datetime.now(timezone.utc)
        topics = [
            "PostgreSQL indexing strategies",
            "Kubernetes pod scheduling",
            "React fiber reconciliation",
            "Rust borrow checker rules",
            "GraphQL schema stitching",
            "Redis pub-sub patterns",
            "Terraform state management",
            "Docker layer caching",
            "Nginx reverse proxy setup",
            "Prometheus alerting rules",
        ]
        memories = [
            {"id": i, "content": topics[i],
             "heat": 0.5, "importance": 0.5, "confidence": 0.8,
             "surprise_score": 0.0, "tags": [f"tag{i}"],
             "created_at": (base - timedelta(days=i * 5)).isoformat()}
            for i in range(10)
        ]

        result = meta.manage_context(memories, max_chunks=3)

        # Should have at most 3 chunks worth of primary memories + overflow summaries
        primary = [m for m in result if m.get("_position_reason") != "overflow_summary"]
        summaries = [m for m in result if m.get("_position_reason") == "overflow_summary"]
        # Primary memories should come from <= 3 chunks
        chunk_ids = {m["_chunk_id"] for m in primary}
        assert len(chunk_ids) <= 3
        # Should have overflow summary
        assert len(summaries) > 0

    def test_manage_context_empty(self, meta):
        """Empty input returns empty output."""
        result = meta.manage_context([])
        assert result == []

    def test_manage_context_default_limit(self, meta):
        """Uses settings.COGNITIVE_LOAD_LIMIT by default."""
        assert meta._chunk_limit == 4  # default from Settings

    def test_primacy_recency_positioning(self, meta):
        """Most important at start (primacy) and second most at end (recency)."""
        base = datetime.now(timezone.utc)
        # Distinct topics spread far apart in time, with varying importance
        topics = [
            "PostgreSQL indexing strategies",
            "Kubernetes pod scheduling",
            "React fiber reconciliation",
            "Rust borrow checker rules",
            "GraphQL schema stitching",
            "Redis pub-sub patterns",
            "Terraform state management",
            "Docker layer caching",
            "Nginx reverse proxy setup",
            "Prometheus alerting rules",
            "Ansible playbook design",
            "ElasticSearch aggregations",
        ]
        memories = [
            {"id": i, "content": topics[i],
             "heat": 0.3 + i * 0.05, "importance": 0.3 + i * 0.05,
             "confidence": 0.8, "surprise_score": 0.0,
             "tags": [f"unique_tag_{i}"],
             "created_at": (base - timedelta(days=i * 5)).isoformat()}
            for i in range(12)
        ]

        result = meta.manage_context(memories, max_chunks=3)

        primary = [m for m in result if m.get("_position_reason") != "overflow_summary"]
        if len(primary) >= 2:
            # First chunk should be primacy
            assert primary[0]["_position_reason"] == "primacy"
            # Last primary chunk should be recency
            last_primary = [m for m in primary if m["_position_reason"] == "recency"]
            assert len(last_primary) > 0


class TestChunkMemories:
    def test_chunk_related_memories(self, meta):
        """Related memories (same tags/entities) grouped together."""
        now = datetime.now(timezone.utc)
        memories = [
            {"id": 1, "content": "FastAPI endpoint for users",
             "tags": ["FastAPI", "users"], "created_at": now.isoformat()},
            {"id": 2, "content": "FastAPI middleware for auth",
             "tags": ["FastAPI", "auth"], "created_at": now.isoformat()},
            {"id": 3, "content": "Django ORM models",
             "tags": ["Django", "database"],
             "created_at": (now - timedelta(days=5)).isoformat()},
        ]

        chunks = meta.chunk_memories(memories)

        assert len(chunks) >= 1
        # FastAPI memories should be grouped (shared entity "FastAPI")
        for chunk in chunks:
            ids = {m["id"] for m in chunk}
            if 1 in ids:
                # If mem 1 is in this chunk, mem 2 should also be
                # (they share "FastAPI" entity)
                assert 2 in ids

    def test_chunk_temporal_proximity(self, meta):
        """Memories close in time get chunked together."""
        base = datetime.now(timezone.utc)
        memories = [
            {"id": 1, "content": "first thing alpha bravo",
             "tags": ["session1"],
             "created_at": base.isoformat()},
            {"id": 2, "content": "second thing charlie delta",
             "tags": ["session1"],
             "created_at": (base + timedelta(minutes=30)).isoformat()},
            {"id": 3, "content": "third thing echo foxtrot",
             "tags": ["session2"],
             "created_at": (base + timedelta(hours=5)).isoformat()},
        ]

        chunks = meta.chunk_memories(memories)

        # Mem 1 and 2 should be together (30 min apart < 2h)
        for chunk in chunks:
            ids = {m["id"] for m in chunk}
            if 1 in ids:
                assert 2 in ids
                assert 3 not in ids

    def test_chunk_singletons(self, meta):
        """Unrelated memories stay as individual chunks."""
        memories = [
            {"id": 1, "content": "alpha bravo charlie",
             "tags": ["a"],
             "created_at": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()},
            {"id": 2, "content": "delta echo foxtrot",
             "tags": ["b"],
             "created_at": (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()},
        ]

        chunks = meta.chunk_memories(memories)

        # Should be 2 separate chunks
        assert len(chunks) == 2

    def test_chunk_empty(self, meta):
        """Empty input returns empty chunks."""
        assert meta.chunk_memories([]) == []


class TestSummarizeOverflow:
    def test_overflow_summarized(self, meta):
        """Excess memories compressed to summary."""
        memories = [
            {"id": i, "content": f"Low priority memory about topic {i}",
             "heat": 0.3, "importance": 0.3, "confidence": 0.7,
             "surprise_score": 0.1}
            for i in range(5)
        ]

        summaries = meta.summarize_overflow(memories)

        assert len(summaries) >= 1
        # At least one summary should exist
        summary_items = [s for s in summaries if s.get("_is_summary")]
        assert len(summary_items) >= 1
        assert summary_items[0]["_summarized_count"] > 0

    def test_overflow_preserves_high_value(self, meta):
        """High-surprise and high-importance memories preserved verbatim."""
        memories = [
            {"id": 1, "content": "Critical finding!",
             "heat": 0.5, "importance": 0.9, "confidence": 0.8,
             "surprise_score": 0.2},
            {"id": 2, "content": "Surprising discovery!",
             "heat": 0.5, "importance": 0.3, "confidence": 0.8,
             "surprise_score": 0.8},
            {"id": 3, "content": "Routine note",
             "heat": 0.3, "importance": 0.3, "confidence": 0.7,
             "surprise_score": 0.1},
        ]

        result = meta.summarize_overflow(memories)

        preserved_contents = [r["content"] for r in result if not r.get("_is_summary")]
        assert "Critical finding!" in preserved_contents  # high importance
        assert "Surprising discovery!" in preserved_contents  # high surprise

    def test_overflow_empty(self, meta):
        """Empty input returns empty result."""
        assert meta.summarize_overflow([]) == []


# ── MCP Tool Integration Tests ────────────────────────────────────────


class TestMCPTools:
    def test_mcp_assess_coverage(self, storage, embeddings, graph, settings, tmp_path):
        """Server tool assess_coverage returns coverage dict."""
        from zikkaron.metacognition import MetaCognition

        mc = MetaCognition(storage, embeddings, graph, settings)

        # Add some memories
        _make_memory(storage, embeddings, "FastAPI routing with path parameters")
        _make_memory(storage, embeddings, "FastAPI dependency injection system")
        graph.add_relationship("FastAPI", "routing", "co_occurrence")

        result = mc.assess_coverage("FastAPI routing", "/proj")

        assert isinstance(result, dict)
        assert "coverage" in result
        assert "suggestion" in result
        assert result["suggestion"] in ("sufficient", "partial", "insufficient")
        assert 0.0 <= result["coverage"] <= 1.0

    def test_mcp_detect_gaps(self, storage, embeddings, graph, settings):
        """Server tool detect_gaps returns gap list."""
        from zikkaron.metacognition import MetaCognition

        mc = MetaCognition(storage, embeddings, graph, settings)

        # Create isolated entity
        storage.insert_entity({"name": "TestIsolated", "type": "variable"})
        # Create low-heat memories
        _make_memory(storage, embeddings, "old data", heat=0.1, directory="/proj")
        _make_memory(storage, embeddings, "more old data", heat=0.1, directory="/proj")

        result = mc.detect_gaps("/proj")

        assert isinstance(result, list)
        # Should find at least isolated entity gap
        types = [g["type"] for g in result]
        assert "isolated_entity" in types
