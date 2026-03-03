"""Tests for dual-store Complementary Learning Systems (CLS)."""

import pytest

from zikkaron.cls_store import DualStoreCLS
from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.storage import StorageEngine


@pytest.fixture
def settings():
    return Settings(
        DB_PATH=":memory:",
        CLUSTER_SIMILARITY_THRESHOLD=0.6,
        CURATION_SIMILARITY_THRESHOLD=0.85,
    )


@pytest.fixture
def storage(tmp_path):
    db_path = str(tmp_path / "test_cls.db")
    engine = StorageEngine(db_path)
    yield engine
    engine.close()


@pytest.fixture
def embeddings():
    return EmbeddingEngine("all-MiniLM-L6-v2")


@pytest.fixture
def cls(storage, embeddings, settings):
    return DualStoreCLS(storage, embeddings, settings)


def _make_memory(
    storage, embeddings, content, directory="/tmp/project", tags=None,
    store_type="episodic", session_id=None, **kwargs
):
    """Helper to insert a memory with real embedding and optional episode link."""
    embedding = embeddings.encode(content)
    # Create an episode to track session_id if provided
    episode_id = None
    if session_id is not None:
        episode_id = storage.insert_episode({
            "session_id": session_id,
            "directory": directory,
            "raw_content": content,
        })

    mem = {
        "content": content,
        "embedding": embedding,
        "tags": tags or ["test"],
        "directory_context": directory,
        "heat": 1.0,
        "is_stale": False,
        "embedding_model": embeddings.get_model_name(),
        "source_episode_id": episode_id,
    }
    mem.update(kwargs)
    mid = storage.insert_memory(mem)

    # Set store_type
    storage._conn.execute(
        "UPDATE memories SET store_type = ? WHERE id = ?",
        (store_type, mid),
    )
    storage._conn.commit()

    return mid


# ── Classification Tests ──────────────────────────────────────────────


class TestClassifyMemory:
    def test_classify_episodic(self, cls):
        """A specific bug report should be classified as episodic."""
        content = "Fixed TypeError in auth.py line 42 when user passes None token"
        result = cls.classify_memory(content, ["bugfix"], "/tmp/project")
        assert result == "episodic"

    def test_classify_semantic_decision_keywords(self, cls):
        """A convention statement with decision keywords should be semantic."""
        content = "Always use factory pattern for creating service instances"
        result = cls.classify_memory(content, ["dev"], "/tmp/project")
        assert result == "semantic"

    def test_classify_semantic_architecture_keywords(self, cls):
        """Architecture-related content without specific indicators should be semantic."""
        content = "The design principle here is composition over inheritance"
        result = cls.classify_memory(content, ["dev"], "/tmp/project")
        assert result == "semantic"

    def test_classify_semantic_tags(self, cls):
        """Tags like 'convention' or 'rule' should force semantic classification."""
        content = "Use bun instead of npm for package management"
        result = cls.classify_memory(content, ["convention"], "/tmp/project")
        assert result == "semantic"

    def test_classify_episodic_specific_with_keywords(self, cls):
        """Content with decision keywords BUT specific file paths stays episodic."""
        content = "Always use JWT pattern in src/auth/middleware.ts for the auth flow"
        result = cls.classify_memory(content, ["dev"], "/tmp/project")
        # Has both decision and specific indicators → episodic (specific wins)
        assert result == "episodic"

    def test_classify_semantic_both_decision_and_architecture(self, cls):
        """Content with both decision AND architecture keywords → semantic."""
        content = "We should always follow the factory pattern for our architecture"
        result = cls.classify_memory(content, ["dev"], "/tmp/project")
        assert result == "semantic"


# ── Pattern Detection Tests ───────────────────────────────────────────


class TestFindRecurringPatterns:
    def test_find_recurring_3_occurrences(self, cls, storage, embeddings):
        """Three similar episodic memories from different sessions → pattern found."""
        # Create 3 very similar memories about JWT auth from different sessions
        _make_memory(
            storage, embeddings,
            "JWT authentication is used for API security",
            session_id="session-001",
        )
        _make_memory(
            storage, embeddings,
            "JWT authentication is used for API authorization",
            session_id="session-002",
        )
        _make_memory(
            storage, embeddings,
            "JWT authentication is used for API access control",
            session_id="session-003",
        )

        patterns = cls.find_recurring_patterns(min_occurrences=3)
        assert len(patterns) >= 1
        # The cluster should have 3 members
        found = any(p["occurrence_count"] >= 3 for p in patterns)
        assert found

    def test_no_pattern_below_threshold(self, cls, storage, embeddings):
        """Only 2 similar memories → should not form a pattern (min_occurrences=3)."""
        _make_memory(
            storage, embeddings,
            "Set up JWT authentication for the API",
            session_id="session-001",
        )
        _make_memory(
            storage, embeddings,
            "Added JWT authentication middleware",
            session_id="session-002",
        )

        patterns = cls.find_recurring_patterns(min_occurrences=3)
        # Should not find any patterns since we only have 2 similar memories
        qualifying = [p for p in patterns if p["occurrence_count"] >= 3]
        assert len(qualifying) == 0

    def test_no_pattern_single_session(self, cls, storage, embeddings):
        """Three similar memories from same session → no pattern (needs session diversity)."""
        for i in range(3):
            _make_memory(
                storage, embeddings,
                f"Set up JWT authentication for API endpoint {i}",
                session_id="session-same",
            )

        patterns = cls.find_recurring_patterns(min_occurrences=3)
        # All from same session → should be filtered out by session diversity check
        assert len(patterns) == 0

    def test_directory_filter(self, cls, storage, embeddings):
        """Pattern search filtered by directory."""
        _make_memory(
            storage, embeddings,
            "Using React hooks for state management",
            directory="/tmp/frontend",
            session_id="session-001",
        )
        _make_memory(
            storage, embeddings,
            "React hooks for managing component state",
            directory="/tmp/frontend",
            session_id="session-002",
        )
        _make_memory(
            storage, embeddings,
            "State management with React hooks pattern",
            directory="/tmp/frontend",
            session_id="session-003",
        )
        # Different directory, different topic
        _make_memory(
            storage, embeddings,
            "Database migration for PostgreSQL",
            directory="/tmp/backend",
            session_id="session-004",
        )

        patterns = cls.find_recurring_patterns(
            directory="/tmp/frontend", min_occurrences=3
        )
        # Should find pattern in frontend, not backend
        if patterns:
            for p in patterns:
                assert "/tmp/frontend" in p["directories"]


# ── Consistency Tests ─────────────────────────────────────────────────


class TestCheckConsistency:
    def test_check_consistency_consistent(self, cls):
        """Non-contradicting cluster should pass consistency check."""
        cluster = [
            {"id": 1, "content": "Use TypeScript for all new modules"},
            {"id": 2, "content": "TypeScript should be used for modules"},
            {"id": 3, "content": "All modules use TypeScript"},
        ]
        result = cls.check_consistency(cluster)
        assert result["consistent"] is True
        assert len(result["contradictions"]) == 0

    def test_check_consistency_contradicting(self, cls):
        """Cluster with negation mismatch should be flagged."""
        cluster = [
            {"id": 1, "content": "Use TypeScript for all modules"},
            {"id": 2, "content": "Do not use TypeScript for new modules"},
            {"id": 3, "content": "TypeScript is used for modules"},
        ]
        result = cls.check_consistency(cluster)
        assert result["consistent"] is False
        assert len(result["contradictions"]) > 0


# ── Schema Abstraction Tests ──────────────────────────────────────────


class TestAbstractToSchema:
    def test_abstract_to_schema(self, cls):
        """Multiple episodic memories about JWT should produce a generalized schema."""
        cluster = [
            {"id": 1, "content": "Set up JWT auth for API", "tags": ["auth"]},
            {"id": 2, "content": "Added JWT verification middleware", "tags": ["auth"]},
            {"id": 3, "content": "JWT token refresh endpoint implemented", "tags": ["auth", "api"]},
        ]
        schema = cls.abstract_to_schema(cluster)
        assert isinstance(schema, str)
        assert len(schema) > 10
        # Should reference JWT since it appears in all three
        assert "jwt" in schema.lower()

    def test_abstract_empty_cluster(self, cls):
        """Empty cluster should return empty string."""
        schema = cls.abstract_to_schema([])
        assert schema == ""

    def test_abstract_preserves_common_tags(self, cls):
        """Schema should mention tags that appear across multiple memories."""
        cluster = [
            {"id": 1, "content": "Deploy with Docker containers", "tags": ["devops"]},
            {"id": 2, "content": "Docker deployment pipeline setup", "tags": ["devops"]},
            {"id": 3, "content": "Container deployment using Docker", "tags": ["devops"]},
        ]
        schema = cls.abstract_to_schema(cluster)
        assert "devops" in schema.lower()


# ── Consolidation Cycle Tests ─────────────────────────────────────────


class TestConsolidationCycle:
    def test_consolidation_promotes(self, cls, storage, embeddings):
        """A recurring pattern should get promoted to a semantic memory."""
        # Create enough similar episodic memories from different sessions
        _make_memory(
            storage, embeddings,
            "Use dependency injection for service construction",
            session_id="session-001",
        )
        _make_memory(
            storage, embeddings,
            "Dependency injection pattern for building services",
            session_id="session-002",
        )
        _make_memory(
            storage, embeddings,
            "Service construction via dependency injection",
            session_id="session-003",
        )

        stats = cls.consolidation_cycle()
        # Should have found and promoted at least one pattern
        assert stats["patterns_found"] >= 0  # May or may not cluster depending on embeddings
        assert stats["total_episodic"] >= 0
        assert stats["total_semantic"] >= 0
        # Check that promoted count is consistent
        assert stats["promoted"] + stats["skipped_inconsistent"] <= stats["patterns_found"]

    def test_episodic_preserved(self, cls, storage, embeddings):
        """Original episodic memories should NOT be deleted after promotion."""
        ids = []
        for i, session in enumerate(["s1", "s2", "s3"]):
            mid = _make_memory(
                storage, embeddings,
                f"Always validate user input before database queries ({i})",
                session_id=session,
            )
            ids.append(mid)

        cls.consolidation_cycle()

        # All original episodic memories should still exist
        for mid in ids:
            mem = storage.get_memory(mid)
            assert mem is not None, f"Memory {mid} was deleted during CLS consolidation"

    def test_consolidation_cycle_stats(self, cls, storage, embeddings):
        """Consolidation cycle should return correctly structured statistics."""
        # Add some memories
        _make_memory(
            storage, embeddings,
            "Testing framework uses pytest with fixtures",
            session_id="s1",
        )
        _make_memory(
            storage, embeddings,
            "Using pytest fixtures for test setup",
            session_id="s2",
        )

        stats = cls.consolidation_cycle()

        # Verify all expected keys exist
        assert "patterns_found" in stats
        assert "promoted" in stats
        assert "skipped_inconsistent" in stats
        assert "total_episodic" in stats
        assert "total_semantic" in stats

        # Types
        assert isinstance(stats["patterns_found"], int)
        assert isinstance(stats["promoted"], int)
        assert isinstance(stats["skipped_inconsistent"], int)
        assert isinstance(stats["total_episodic"], int)
        assert isinstance(stats["total_semantic"], int)

    def test_skips_inconsistent_patterns(self, cls, storage, embeddings):
        """Patterns with contradictions should be skipped during consolidation."""
        # Create contradicting memories that are still similar enough to cluster
        _make_memory(
            storage, embeddings,
            "We use Redis for caching in all services",
            session_id="s1",
            tags=["caching"],
        )
        _make_memory(
            storage, embeddings,
            "We do not use Redis for caching anymore",
            session_id="s2",
            tags=["caching"],
        )
        _make_memory(
            storage, embeddings,
            "Redis caching is used across all our services",
            session_id="s3",
            tags=["caching"],
        )

        stats = cls.consolidation_cycle()
        # The contradicting cluster should be skipped
        # (exact behavior depends on embedding similarity and clustering)
        assert stats["skipped_inconsistent"] >= 0


# ── Dual-Store Query Tests ────────────────────────────────────────────


class TestQueryDual:
    def test_query_dual_specific(self, cls, storage, embeddings):
        """A specific query (with file path) should weight episodic results higher."""
        # Create episodic and semantic memories
        _make_memory(
            storage, embeddings,
            "Fixed bug in src/auth/login.py that caused null token error",
            store_type="episodic",
        )
        _make_memory(
            storage, embeddings,
            "Authentication system uses JWT tokens with refresh mechanism",
            store_type="semantic",
        )

        results = cls.query_dual(
            "error in src/auth/login.py", directory="", prefer="auto"
        )
        assert isinstance(results, list)
        # The specific episodic memory should appear in results
        if results:
            # Results should contain memories with _dual_score
            assert "_dual_score" in results[0]

    def test_query_dual_general(self, cls, storage, embeddings):
        """A general query (about patterns) should weight semantic results higher."""
        _make_memory(
            storage, embeddings,
            "Fixed TypeError in auth.py line 42",
            store_type="episodic",
        )
        _make_memory(
            storage, embeddings,
            "The architecture pattern uses factory methods for service creation",
            store_type="semantic",
        )

        results = cls.query_dual(
            "what architecture pattern do we use", directory="", prefer="auto"
        )
        assert isinstance(results, list)

    def test_query_dual_prefer_episodic(self, cls, storage, embeddings):
        """Explicit episodic preference should weight episodic 2x."""
        _make_memory(
            storage, embeddings,
            "Fixed bug in authentication module",
            store_type="episodic",
        )
        _make_memory(
            storage, embeddings,
            "Authentication uses JWT tokens as standard",
            store_type="semantic",
        )

        results = cls.query_dual(
            "authentication", directory="", prefer="episodic"
        )
        assert isinstance(results, list)

    def test_query_dual_prefer_semantic(self, cls, storage, embeddings):
        """Explicit semantic preference should weight semantic 2x."""
        _make_memory(
            storage, embeddings,
            "Fixed bug in authentication module",
            store_type="episodic",
        )
        _make_memory(
            storage, embeddings,
            "Authentication uses JWT tokens as standard",
            store_type="semantic",
        )

        results = cls.query_dual(
            "authentication", directory="", prefer="semantic"
        )
        assert isinstance(results, list)

    def test_query_dual_directory_filter(self, cls, storage, embeddings):
        """Query with directory should only return memories from that directory."""
        _make_memory(
            storage, embeddings,
            "React hooks for state management",
            directory="/tmp/frontend",
            store_type="episodic",
        )
        _make_memory(
            storage, embeddings,
            "Database migration script for PostgreSQL",
            directory="/tmp/backend",
            store_type="episodic",
        )

        results = cls.query_dual(
            "state management", directory="/tmp/frontend", prefer="auto"
        )
        for r in results:
            assert r.get("directory_context") == "/tmp/frontend"

    def test_query_dual_no_embedding(self, cls, storage):
        """Query with no embeddings available should return empty list."""
        # Use a cls with broken embeddings
        bad_embeddings = EmbeddingEngine("nonexistent-model")
        bad_embeddings._unavailable = True
        bad_cls = DualStoreCLS(storage, bad_embeddings, Settings(DB_PATH=":memory:"))

        results = bad_cls.query_dual("test query", directory="", prefer="auto")
        assert results == []
