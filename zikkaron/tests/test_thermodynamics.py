"""Tests for the neuroscience-inspired memory thermodynamics module."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.storage import StorageEngine
from zikkaron.thermodynamics import MemoryThermodynamics

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
    engine = StorageEngine(str(tmp_path / "test_thermo.db"), embedding_dim=384)
    yield engine
    engine.close()


@pytest.fixture
def settings():
    return Settings(
        DB_PATH=":memory:",
        DECAY_FACTOR=0.95,
        IMPORTANCE_DECAY_FACTOR=0.998,
        SURPRISE_BOOST=0.3,
        EMOTIONAL_DECAY_RESISTANCE=0.5,
        SYNAPTIC_WINDOW_MINUTES=30,
        SYNAPTIC_BOOST=0.2,
    )


@pytest.fixture
def embeddings():
    return EmbeddingEngine()


@pytest.fixture
def thermo(storage, embeddings, settings):
    return MemoryThermodynamics(storage, embeddings, settings)


def _make_embedding(dim: int = 384, seed: int = 0) -> bytes:
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    return vec.tobytes()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _minutes_ago(minutes: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()


# -- a. Surprise Scoring --


class TestSurpriseScoring:
    @requires_model
    def test_surprise_score_novel(self, thermo, storage, embeddings):
        """New content unlike anything stored should have high surprise."""
        # Store a memory about Python
        emb = embeddings.encode("Python web framework using Flask for REST APIs")
        storage.insert_memory({
            "content": "Python web framework using Flask for REST APIs",
            "embedding": emb,
            "directory_context": "/proj",
        })

        # Compute surprise for completely different content
        surprise = thermo.compute_surprise(
            "Quantum computing uses qubits for superposition calculations",
            "/proj",
        )
        # Novel content should have high surprise (> 0.3)
        assert surprise > 0.3

    @requires_model
    def test_surprise_score_familiar(self, thermo, storage, embeddings):
        """Content very similar to existing memory should have low surprise."""
        content = "Python Flask REST API development with SQLAlchemy"
        emb = embeddings.encode(content)
        storage.insert_memory({
            "content": content,
            "embedding": emb,
            "directory_context": "/proj",
        })

        # Nearly identical content
        surprise = thermo.compute_surprise(
            "Python Flask REST API development using SQLAlchemy ORM",
            "/proj",
        )
        # Familiar content should have low surprise (< 0.3)
        assert surprise < 0.3

    def test_surprise_no_memories(self, thermo):
        """With no existing memories, surprise should be moderate (0.5)."""
        surprise = thermo.compute_surprise("anything at all", "/empty")
        assert surprise == 0.5


# -- b. Importance Scoring --


class TestImportanceScoring:
    def test_error_content(self, thermo):
        score = thermo.compute_importance(
            "Got a TypeError exception in the traceback", []
        )
        assert score >= 0.2

    def test_decision_content(self, thermo):
        score = thermo.compute_importance(
            "Decided to switch from Redis to PostgreSQL", []
        )
        assert score >= 0.3

    def test_architecture_content(self, thermo):
        score = thermo.compute_importance(
            "Refactored the architecture to use a modular design pattern", []
        )
        assert score >= 0.2

    def test_many_tags(self, thermo):
        score = thermo.compute_importance(
            "Simple note", ["tag1", "tag2", "tag3"]
        )
        assert score >= 0.1

    def test_long_content(self, thermo):
        score = thermo.compute_importance("x " * 300, [])
        assert score >= 0.1

    def test_code_blocks(self, thermo):
        score = thermo.compute_importance(
            "Found in `config.py` at src/main/app.py", []
        )
        assert score >= 0.1

    def test_high_importance_combined(self, thermo):
        """Content with multiple signals should score high."""
        score = thermo.compute_importance(
            "Decided to refactor the architecture after a critical exception "
            "in the traceback. See src/server.py for the `fix_handler` code. " + "x" * 500,
            ["critical", "architecture", "refactor"],
        )
        assert score >= 0.8

    def test_importance_capped_at_one(self, thermo):
        """Importance should never exceed 1.0."""
        score = thermo.compute_importance(
            "Decided to refactor the architecture after a critical exception "
            "in the traceback. See src/server.py for details. " + "x" * 500,
            ["a", "b", "c"],
        )
        assert score <= 1.0


# -- c. Emotional Valence --


class TestEmotionalValence:
    def test_valence_frustration(self, thermo):
        """Error-laden content should produce negative valence."""
        valence = thermo.compute_valence(
            "The build failed with a crash. Multiple errors and timeout issues. "
            "Bug in the broken deployment."
        )
        assert valence < 0.0

    def test_valence_satisfaction(self, thermo):
        """Success content should produce positive valence."""
        valence = thermo.compute_valence(
            "Fixed the bug, all tests passed. Successfully deployed and merged. "
            "The feature is now shipped and approved."
        )
        assert valence > 0.0

    def test_valence_neutral(self, thermo):
        """Content with no emotional signals should be neutral."""
        valence = thermo.compute_valence(
            "The cat sat on the mat while the dog slept nearby."
        )
        assert valence == 0.0

    def test_valence_range(self, thermo):
        """Valence should always be between -1.0 and +1.0."""
        for text in [
            "error " * 100,
            "fixed " * 100,
            "neutral text",
        ]:
            v = thermo.compute_valence(text)
            assert -1.0 <= v <= 1.0


# -- d. Enhanced Decay --


class TestEnhancedDecay:
    def test_enhanced_decay_important(self, thermo):
        """High-importance memories should decay slower than normal ones."""
        hours = 24.0
        normal_mem = {"heat": 1.0, "importance": 0.3, "emotional_valence": 0.0, "confidence": 1.0}
        important_mem = {"heat": 1.0, "importance": 0.8, "emotional_valence": 0.0, "confidence": 1.0}

        normal_heat = thermo.compute_decay(normal_mem, hours)
        important_heat = thermo.compute_decay(important_mem, hours)

        # Important memory should retain more heat
        assert important_heat > normal_heat
        # Both should be less than starting heat
        assert normal_heat < 1.0
        assert important_heat < 1.0

    def test_enhanced_decay_emotional(self, thermo):
        """High-valence memories should decay slower."""
        hours = 24.0
        neutral_mem = {"heat": 1.0, "importance": 0.3, "emotional_valence": 0.0, "confidence": 1.0}
        emotional_mem = {"heat": 1.0, "importance": 0.3, "emotional_valence": -0.8, "confidence": 1.0}

        neutral_heat = thermo.compute_decay(neutral_mem, hours)
        emotional_heat = thermo.compute_decay(emotional_mem, hours)

        # Emotional memory should retain more heat
        assert emotional_heat > neutral_heat

    def test_decay_confidence_modifier(self, thermo):
        """High-confidence memories should decay slower."""
        hours = 24.0
        low_conf = {"heat": 1.0, "importance": 0.3, "emotional_valence": 0.0, "confidence": 0.2}
        high_conf = {"heat": 1.0, "importance": 0.3, "emotional_valence": 0.0, "confidence": 1.0}

        low_heat = thermo.compute_decay(low_conf, hours)
        high_heat = thermo.compute_decay(high_conf, hours)

        assert high_heat > low_heat

    def test_decay_returns_lower_heat(self, thermo):
        """Decay should always produce a lower heat value."""
        mem = {"heat": 0.8, "importance": 0.9, "emotional_valence": 0.5, "confidence": 1.0}
        new_heat = thermo.compute_decay(mem, 1.0)
        assert new_heat < 0.8


# -- e. Synaptic Boost --


class TestSynapticBoost:
    def test_synaptic_boost(self, thermo, storage):
        """Memories near a high-importance event should get boosted."""
        now = _now_iso()
        five_min_ago = _minutes_ago(5)

        # Create a nearby memory (5 min ago)
        nearby_id = storage.insert_memory({
            "content": "nearby context memory",
            "directory_context": "/proj",
            "heat": 0.5,
            "created_at": five_min_ago,
        })

        # Create the high-importance memory (now)
        target_id = storage.insert_memory({
            "content": "critical decision memory",
            "directory_context": "/proj",
            "heat": 1.0,
            "created_at": now,
        })

        boosted = thermo.synaptic_boost(target_id, 1.0)

        # The nearby memory should have been boosted
        assert boosted >= 1
        mem = storage.get_memory(nearby_id)
        assert mem["heat"] > 0.5

    def test_synaptic_boost_no_self(self, thermo, storage):
        """A memory should not boost itself."""
        now = _now_iso()
        mid = storage.insert_memory({
            "content": "self memory",
            "directory_context": "/proj",
            "heat": 0.5,
            "created_at": now,
        })
        thermo.synaptic_boost(mid, 1.0)
        mem = storage.get_memory(mid)
        # Heat should be unchanged (no self-boost)
        assert mem["heat"] == 0.5

    def test_synaptic_boost_respects_window(self, thermo, storage):
        """Memories outside the time window should not be boosted."""
        now = _now_iso()
        long_ago = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()

        # Create a distant memory (2 hours ago)
        distant_id = storage.insert_memory({
            "content": "old context memory",
            "directory_context": "/proj",
            "heat": 0.5,
            "created_at": long_ago,
        })

        # Create the high-importance memory (now)
        target_id = storage.insert_memory({
            "content": "recent event",
            "directory_context": "/proj",
            "heat": 1.0,
            "created_at": now,
        })

        thermo.synaptic_boost(target_id, 1.0)

        mem = storage.get_memory(distant_id)
        assert mem["heat"] == 0.5  # unchanged


# -- f. Metamemory --


class TestMetamemory:
    def test_metamemory_tracking(self, thermo, storage):
        """Access and usefulness tracking should work correctly."""
        mid = storage.insert_memory({
            "content": "trackable memory",
            "directory_context": "/proj",
        })

        thermo.record_access(mid, was_useful=True)
        thermo.record_access(mid, was_useful=True)
        thermo.record_access(mid, was_useful=False)

        mem = storage.get_memory(mid)
        assert mem["access_count"] == 3
        assert mem["useful_count"] == 2
        # Confidence should still be 1.0 (benefit of doubt when access_count <= 3)
        assert mem["confidence"] == 1.0

    def test_confidence_calculation(self, thermo, storage):
        """Confidence should update after 3+ accesses."""
        mid = storage.insert_memory({
            "content": "confidence test memory",
            "directory_context": "/proj",
        })

        # 4 accesses, 3 useful
        thermo.record_access(mid, was_useful=True)
        thermo.record_access(mid, was_useful=True)
        thermo.record_access(mid, was_useful=True)
        thermo.record_access(mid, was_useful=False)

        mem = storage.get_memory(mid)
        assert mem["access_count"] == 4
        assert mem["useful_count"] == 3
        # confidence = 3/4 = 0.75
        assert mem["confidence"] == pytest.approx(0.75)

    def test_reliability_benefit_of_doubt(self, thermo, storage):
        """New memories with few accesses should get benefit of the doubt."""
        mid = storage.insert_memory({
            "content": "new memory",
            "directory_context": "/proj",
        })
        assert thermo.get_reliability(mid) == 1.0

    def test_reliability_after_many_accesses(self, thermo, storage):
        """Reliability should reflect actual usefulness after enough data."""
        mid = storage.insert_memory({
            "content": "tested memory",
            "directory_context": "/proj",
        })

        # 5 accesses, 2 useful → confidence = 0.4
        for useful in [True, True, False, False, False]:
            thermo.record_access(mid, was_useful=useful)

        reliability = thermo.get_reliability(mid)
        assert reliability == pytest.approx(0.4)

    def test_record_access_nonexistent(self, thermo):
        """Recording access for a nonexistent memory should be a no-op."""
        thermo.record_access(99999, was_useful=True)  # should not raise
