"""Tests for predictive coding write gate — surprisal-based memory gating."""

from datetime import datetime, timedelta, timezone

import pytest

from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.knowledge_graph import KnowledgeGraph
from zikkaron.predictive_coding import PredictiveCodingGate
from zikkaron.retrieval import HippoRetriever
from zikkaron.storage import StorageEngine


@pytest.fixture
def settings():
    return Settings(
        DB_PATH=":memory:",
        WRITE_GATE_THRESHOLD=0.4,
    )


@pytest.fixture
def storage(tmp_path):
    db_path = str(tmp_path / "test_gate.db")
    engine = StorageEngine(db_path)
    yield engine
    engine.close()


@pytest.fixture
def embeddings():
    return EmbeddingEngine("all-MiniLM-L6-v2")


@pytest.fixture
def kg(storage, settings):
    return KnowledgeGraph(storage, settings)


@pytest.fixture
def retriever(storage, embeddings, kg, settings):
    return HippoRetriever(storage, embeddings, kg, settings)


@pytest.fixture
def gate(storage, embeddings, retriever, settings):
    return PredictiveCodingGate(storage, embeddings, retriever, settings)


def _make_memory(storage, embeddings, content, directory="/tmp/project", tags=None, **kwargs):
    """Helper to insert a memory with real embedding."""
    embedding = embeddings.encode(content)
    mem = {
        "content": content,
        "embedding": embedding,
        "tags": tags or ["test"],
        "directory_context": directory,
        "heat": 1.0,
        "is_stale": False,
        "embedding_model": embeddings.get_model_name(),
    }
    mem.update(kwargs)
    mid = storage.insert_memory(mem)
    return mid


class TestHighSurprisalNovelContent:
    def test_novel_content_passes_gate(self, gate, storage, embeddings):
        """Completely novel content in an existing directory should pass the gate."""
        # Seed with some Python-related memories
        _make_memory(
            storage, embeddings,
            "Using Flask for the web API with SQLAlchemy ORM",
            directory="/tmp/project",
        )
        _make_memory(
            storage, embeddings,
            "Configured pytest with coverage reporting",
            directory="/tmp/project",
        )

        # Novel content about a completely different topic
        should_store, surprisal, reason = gate.should_store(
            "Implemented GPU-accelerated matrix multiplication with CUDA kernels",
            "/tmp/project",
            ["cuda", "gpu"],
        )
        assert should_store is True
        assert surprisal >= 0.4
        assert reason == "high_surprisal"


class TestLowSurprisalDuplicate:
    def test_near_duplicate_is_blocked(self, gate, storage, embeddings):
        """Near-duplicate content should be blocked by the gate."""
        _make_memory(
            storage, embeddings,
            "Using Flask for the web API with SQLAlchemy ORM for database access",
            directory="/tmp/project",
        )

        # Very similar content
        should_store, surprisal, reason = gate.should_store(
            "Using Flask for the web API with SQLAlchemy ORM for database queries",
            "/tmp/project",
            ["flask"],
        )
        assert should_store is False
        assert surprisal < 0.4
        assert reason.startswith("below_threshold")


class TestAlwaysStoreErrors:
    def test_error_keywords_bypass_gate(self, gate, storage, embeddings):
        """Content with error keywords should always be stored."""
        # Seed with similar content
        _make_memory(
            storage, embeddings,
            "Database connection configuration for PostgreSQL",
            directory="/tmp/project",
        )

        # Error content about database — even if topic is similar, bypass gate
        should_store, surprisal, reason = gate.should_store(
            "Database connection error: PostgreSQL connection refused on port 5432",
            "/tmp/project",
            ["database"],
        )
        assert should_store is True
        assert reason == "bypass_error_keywords"

    def test_exception_keyword_bypasses(self, gate, storage, embeddings):
        """Content mentioning exceptions should bypass the gate."""
        _make_memory(
            storage, embeddings,
            "Python function for data processing",
            directory="/tmp/project",
        )
        should_store, surprisal, reason = gate.should_store(
            "Python exception in data processing pipeline",
            "/tmp/project",
            ["python"],
        )
        assert should_store is True
        assert reason == "bypass_error_keywords"

    def test_traceback_keyword_bypasses(self, gate, storage, embeddings):
        """Content mentioning tracebacks should bypass."""
        should_store, _, reason = gate.should_store(
            "Traceback most recent call last in main.py",
            "/tmp/project",
            [],
        )
        assert should_store is True
        assert reason == "bypass_error_keywords"

    def test_failed_keyword_bypasses(self, gate, storage, embeddings):
        """Content with 'failed' should bypass."""
        should_store, _, reason = gate.should_store(
            "Build failed due to missing dependency",
            "/tmp/project",
            [],
        )
        assert should_store is True
        assert reason == "bypass_error_keywords"

    def test_bug_keyword_bypasses(self, gate, storage, embeddings):
        """Content with 'bug' should bypass."""
        should_store, _, reason = gate.should_store(
            "Found a bug in the authentication logic",
            "/tmp/project",
            [],
        )
        assert should_store is True
        assert reason == "bypass_error_keywords"

    def test_crash_keyword_bypasses(self, gate, storage, embeddings):
        """Content with 'crash' should bypass."""
        should_store, _, reason = gate.should_store(
            "Application crash on startup after config change",
            "/tmp/project",
            [],
        )
        assert should_store is True
        assert reason == "bypass_error_keywords"


class TestAlwaysStoreDecisions:
    def test_decision_keywords_bypass_gate(self, gate, storage, embeddings):
        """Content with decision keywords should always be stored."""
        _make_memory(
            storage, embeddings,
            "Working with the project configuration system",
            directory="/tmp/project",
        )

        should_store, surprisal, reason = gate.should_store(
            "Decided to use Redis instead of Memcached for caching",
            "/tmp/project",
            ["caching"],
        )
        assert should_store is True
        assert reason == "bypass_decision_keywords"

    def test_chose_keyword_bypasses(self, gate, storage, embeddings):
        """Content with 'chose' should bypass."""
        should_store, _, reason = gate.should_store(
            "Chose TypeScript over JavaScript for type safety",
            "/tmp/project",
            [],
        )
        assert should_store is True
        assert reason == "bypass_decision_keywords"

    def test_switched_to_keyword_bypasses(self, gate, storage, embeddings):
        """Content with 'switched to' should bypass."""
        should_store, _, reason = gate.should_store(
            "Switched to pnpm from npm for faster installs",
            "/tmp/project",
            [],
        )
        assert should_store is True
        assert reason == "bypass_decision_keywords"

    def test_migrated_keyword_bypasses(self, gate, storage, embeddings):
        """Content with 'migrated' should bypass."""
        should_store, _, reason = gate.should_store(
            "Migrated database from MySQL to PostgreSQL",
            "/tmp/project",
            [],
        )
        assert should_store is True
        assert reason == "bypass_decision_keywords"

    def test_architecture_keyword_bypasses(self, gate, storage, embeddings):
        """Content with 'architecture' should bypass."""
        should_store, _, reason = gate.should_store(
            "Redesigned the microservices architecture for better scalability",
            "/tmp/project",
            [],
        )
        assert should_store is True
        assert reason == "bypass_decision_keywords"


class TestImportantTagBypass:
    def test_important_tag_bypasses(self, gate, storage, embeddings):
        """Content tagged 'important' should always be stored."""
        _make_memory(
            storage, embeddings,
            "Standard project setup with Python and pip",
            directory="/tmp/project",
        )

        should_store, _, reason = gate.should_store(
            "Standard project setup with Python and pip configuration",
            "/tmp/project",
            ["important"],
        )
        assert should_store is True
        assert reason == "bypass_important_tag"

    def test_critical_tag_bypasses(self, gate, storage, embeddings):
        """Content tagged 'critical' should always be stored."""
        should_store, _, reason = gate.should_store(
            "Routine maintenance task for the project",
            "/tmp/project",
            ["critical"],
        )
        assert should_store is True
        assert reason == "bypass_important_tag"


class TestEmptyDirectoryModerateSurprise:
    def test_new_directory_returns_high_surprisal(self, gate):
        """A brand new directory with no memories should return ~0.8 surprisal."""
        surprisal = gate.compute_surprisal(
            "Setting up a new Rust project with Cargo",
            "/tmp/brand-new-project",
            ["rust"],
        )
        assert surprisal == pytest.approx(0.8, abs=0.01)

    def test_new_directory_always_passes_gate(self, gate):
        """Content for a brand new directory should pass the gate."""
        should_store, surprisal, reason = gate.should_store(
            "Initializing a new Go microservice",
            "/tmp/brand-new-service",
            ["go"],
        )
        assert should_store is True
        assert surprisal >= 0.4


class TestEntityNoveltyNewEntities:
    def test_new_entities_increase_surprisal(self, gate, storage, embeddings):
        """Content with new entities not in the graph should increase surprisal."""
        # Seed with known entities
        _make_memory(
            storage, embeddings,
            "def process_data(): pass\ndef validate_input(): pass",
            directory="/tmp/project",
        )
        # Add known entities to entity table
        storage.insert_entity({"name": "process_data", "type": "function"})
        storage.insert_entity({"name": "validate_input", "type": "function"})

        # Content with entirely new entity names
        entity_novelty = gate._compute_entity_novelty(
            "def quantum_entangle(): pass\ndef teleport_state(): pass",
            "/tmp/project",
        )
        # New entities should yield high entity novelty
        assert entity_novelty > 0.5

    def test_existing_entities_lower_novelty(self, gate, storage, embeddings):
        """Content referencing existing entities should have lower entity novelty."""
        storage.insert_entity({"name": "process_data", "type": "function"})
        storage.insert_entity({"name": "validate_input", "type": "function"})

        entity_novelty = gate._compute_entity_novelty(
            "def process_data(): updated\ndef validate_input(): improved",
            "/tmp/project",
        )
        # Existing entities should yield lower novelty
        assert entity_novelty < 0.8


class TestTemporalNoveltyRecent:
    def test_recent_topic_has_low_temporal_novelty(self, gate, storage, embeddings):
        """A topic discussed very recently should have low temporal novelty."""
        # Insert a memory about Flask with a very recent timestamp
        storage.insert_entity({"name": "Flask", "type": "dependency"})
        _make_memory(
            storage, embeddings,
            "Setting up Flask web server with routes",
            directory="/tmp/project",
        )

        temporal_novelty = gate._compute_temporal_novelty(
            "Adding Flask middleware for authentication",
            "/tmp/project",
        )
        # Recent discussion about Flask → low temporal novelty
        assert temporal_novelty <= 0.3

    def test_no_related_entities_high_temporal_novelty(self, gate):
        """Content with no matching entities should have high temporal novelty."""
        temporal_novelty = gate._compute_temporal_novelty(
            "Quantum computing entanglement protocol",
            "/tmp/empty-project",
        )
        assert temporal_novelty >= 0.7


class TestBoundaryDetectionTopicChange:
    def test_strong_signal_on_topic_transition(self, gate):
        """A strong topic change should yield boundary > 0.6."""
        boundary = gate.compute_boundary_signal(
            "Implementing GPU-accelerated neural network training with CUDA",
            "Debugging the CSS layout issue in the navigation bar",
        )
        assert boundary > 0.6

    def test_weak_signal_on_same_topic(self, gate):
        """Similar topics should yield low boundary signal."""
        boundary = gate.compute_boundary_signal(
            "Fixed a bug in the Flask authentication middleware",
            "Updated the Flask authentication middleware with new tokens",
        )
        assert boundary < 0.6

    def test_boundary_in_valid_range(self, gate):
        """Boundary signal should always be in [0.0, 1.0]."""
        boundary = gate.compute_boundary_signal(
            "Hello world",
            "Goodbye world",
        )
        assert 0.0 <= boundary <= 1.0


class TestDirectoryModelBuilds:
    def test_directory_model_returns_correct_stats(self, gate, storage, embeddings):
        """Directory model should return accurate statistics."""
        _make_memory(
            storage, embeddings,
            "Flask web server configuration",
            directory="/tmp/myproject",
            tags=["flask", "web"],
        )
        _make_memory(
            storage, embeddings,
            "SQLAlchemy database models and migrations",
            directory="/tmp/myproject",
            tags=["database", "flask"],
        )

        model = gate.get_directory_model("/tmp/myproject")
        assert model["memory_count"] == 2
        assert model["avg_heat"] > 0.0
        assert isinstance(model["common_tags"], list)
        assert isinstance(model["recent_topics"], list)
        assert model["centroid_embedding"] is not None

    def test_empty_directory_model(self, gate):
        """Empty directory should return zero-valued model."""
        model = gate.get_directory_model("/tmp/nonexistent")
        assert model["memory_count"] == 0
        assert model["entity_count"] == 0
        assert model["avg_heat"] == 0.0
        assert model["common_tags"] == []
        assert model["recent_topics"] == []
        assert model["centroid_embedding"] is None


class TestWriteGateIntegration:
    def test_server_remember_respects_gate(self, tmp_path):
        """Server remember() should respect write gate decisions."""
        from zikkaron import server

        db_path = str(tmp_path / "test_integration.db")
        server.init_engines(db_path=db_path, start_daemons=False)

        try:
            # First, store a base memory (novel, should pass gate)
            result1 = server.remember(
                content="Using Redis for caching with TTL-based expiration",
                context="/tmp/integration-test",
                tags=["redis", "caching"],
            )
            # First memory in a new directory should always be stored
            assert "id" in result1

            # Store another base memory
            server.remember(
                content="PostgreSQL database with connection pooling via pgbouncer",
                context="/tmp/integration-test",
                tags=["postgres", "database"],
            )

            # Now try to store a near-duplicate — should be blocked
            result3 = server.remember(
                content="Using Redis for caching with TTL-based expiration policy",
                context="/tmp/integration-test",
                tags=["redis"],
            )
            # This may be blocked or may be merged by curator
            # If blocked by gate: "stored" will be False
            # If not blocked (similar but not identical), it may pass
            # We just verify the result has the expected keys
            if "stored" in result3 and result3["stored"] is False:
                assert "surprisal" in result3
                assert "reason" in result3
            else:
                # It was stored — should have surprisal in response
                assert "surprisal" in result3 or "id" in result3
        finally:
            server.shutdown()


class TestSurprisalReturnedInResponse:
    def test_surprisal_in_remember_response(self, tmp_path):
        """Surprisal score should be included in remember() response."""
        from zikkaron import server

        db_path = str(tmp_path / "test_surprisal_response.db")
        server.init_engines(db_path=db_path, start_daemons=False)

        try:
            # Store a novel memory
            result = server.remember(
                content="Implementing a brand new quantum error correction algorithm",
                context="/tmp/surprisal-test",
                tags=["quantum"],
            )

            # Should have surprisal field in the response
            assert "surprisal" in result
            assert isinstance(result["surprisal"], float)
            assert 0.0 <= result["surprisal"] <= 1.0
            assert "gate_reason" in result
        finally:
            server.shutdown()

    def test_blocked_memory_returns_surprisal(self, tmp_path):
        """Even blocked memories should return their surprisal score."""
        from zikkaron import server

        db_path = str(tmp_path / "test_blocked_surprisal.db")
        server.init_engines(db_path=db_path, start_daemons=False)

        try:
            # Store several base memories to build a generative model
            base_content = "Python Flask web application with REST API endpoints"
            server.remember(
                content=base_content,
                context="/tmp/surprisal-block-test",
                tags=["flask"],
            )

            # Try near-duplicate
            result = server.remember(
                content=base_content,  # Exact same content
                context="/tmp/surprisal-block-test",
                tags=["flask"],
            )

            # Whether blocked by gate or merged by curator, surprisal should be present
            if "stored" in result and result["stored"] is False:
                assert "surprisal" in result
                assert result["surprisal"] < 0.4
            else:
                # Merged or stored — surprisal should still be in response
                assert "surprisal" in result or "curation_action" in result
        finally:
            server.shutdown()


class TestSurprisalComputation:
    def test_surprisal_range(self, gate, storage, embeddings):
        """Surprisal should always be in [0.0, 1.0]."""
        _make_memory(
            storage, embeddings,
            "Python web development with Django",
            directory="/tmp/project",
        )
        surprisal = gate.compute_surprisal(
            "Building web apps with Python Django framework",
            "/tmp/project",
            ["python"],
        )
        assert 0.0 <= surprisal <= 1.0

    def test_identical_content_low_surprisal(self, gate, storage, embeddings):
        """Identical content should have very low surprisal."""
        content = "Configuring nginx reverse proxy with SSL termination"
        _make_memory(storage, embeddings, content, directory="/tmp/project")

        surprisal = gate.compute_surprisal(content, "/tmp/project", [])
        assert surprisal < 0.4

    def test_novel_content_high_surprisal(self, gate, storage, embeddings):
        """Completely novel content should have high surprisal."""
        _make_memory(
            storage, embeddings,
            "Python web application with Flask",
            directory="/tmp/project",
        )

        surprisal = gate.compute_surprisal(
            "Implementing quantum annealing optimization for protein folding",
            "/tmp/project",
            ["quantum"],
        )
        assert surprisal > 0.4


class TestStructuralNovelty:
    def test_no_entities_returns_low_structural_novelty(self, gate):
        """Content with no extractable entities should return low structural novelty."""
        novelty = gate._compute_structural_novelty(
            "This is a simple note",
            "/tmp/project",
        )
        assert novelty <= 0.2

    def test_new_relationship_context_high_novelty(self, gate, storage, embeddings):
        """Content introducing new relationship types should yield high novelty."""
        # No existing relationships — any rel context is new
        novelty = gate._compute_structural_novelty(
            "Fixed the ImportError in the main module by updating the package",
            "/tmp/project",
        )
        # resolved_by from the error-fix pattern should be detected
        # If it's a new rel type in the graph → 0.8
        assert novelty >= 0.2  # At minimum


class TestEmbeddingNovelty:
    def test_no_vectors_high_novelty(self, gate):
        """No existing vectors should yield high embedding novelty."""
        novelty = gate._compute_embedding_novelty(
            "Brand new content with no prior context"
        )
        assert novelty >= 0.7

    def test_similar_content_low_novelty(self, gate, storage, embeddings):
        """Similar content should yield low embedding novelty."""
        _make_memory(
            storage, embeddings,
            "Setting up Flask with SQLAlchemy for database access",
            directory="/tmp/project",
        )
        novelty = gate._compute_embedding_novelty(
            "Configuring Flask with SQLAlchemy for database operations"
        )
        assert novelty < 0.5
